import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from questionnaire.app import create_app
from questionnaire.storage import register, invite, connect, digest
from questionnaire.__main__ import export_csv
from questionnaire.schema import HEADERS

class QuestionnaireTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'videos').mkdir();(self.root/'videos'/'sample.mp4').write_bytes(b'0123456789')
        self.cfg=dict(id='test',episode='conversation_test',theme='AIー縛りなし',video='sample.mp4',start_seconds=0,end_seconds=[1,2])
        register(self.root,self.cfg)
        self.codes={str(i):invite(self.root,'test',str(i),role) for i,role in enumerate('ABC',1)}
        self.app=create_app(self.root,{'TESTING':True})
        self.client=self.app.test_client()
    def tearDown(self):self.temp.cleanup()
    def login(self,client=None,code=None):
        client=client or self.client
        csrf=client.get('/api/session').json['csrf']
        result=client.post('/api/login',json={'code':code or self.codes['1']},headers={'X-CSRF-Token':csrf})
        self.assertEqual(result.status_code,200)
        return client.get('/api/session').json['csrf']
    def post(self,path,data,csrf,client=None):return (client or self.client).post(path,json=data,headers={'X-CSRF-Token':csrf})

    def test_access_csrf_range_and_isolation(self):
        self.assertEqual(self.client.get('/media').status_code,401)
        self.assertEqual(self.client.get('/private/responses.sqlite3').status_code,404)
        csrf=self.login()
        response=self.client.get('/media',headers={'Range':'bytes=2-5'})
        self.assertEqual(response.status_code,206);self.assertEqual(response.data,b'2345');response.close()
        self.assertEqual(self.client.post('/api/answer',json={}).status_code,403)
        self.assertEqual(self.client.get('/api/session').json['participant'],'1')
        other=self.app.test_client();self.login(other,self.codes['2'])
        self.assertEqual(other.get('/api/session').json['participant'],'2')
        self.assertNotIn('answers',other.get('/api/session').json)
        self.assertIn("media-src 'self'",self.client.get('/').headers['Content-Security-Policy'])

    def test_save_resume_validate_finish_export_and_analyzer(self):
        for pid,code in self.codes.items():
            client=self.app.test_client();csrf=self.login(client,code)
            self.assertEqual(self.post('/api/answer',{'index':2,'A-B':0,'B-C':0,'C-A':0},csrf,client).status_code,409)
            self.assertEqual(self.post('/api/answer',{'index':1,'A-B':2,'B-C':0,'C-A':0},csrf,client).status_code,400)
            self.assertEqual(self.post('/api/answer',{'index':1,'A-B':None,'B-C':0,'C-A':0},csrf,client).status_code,400)
            self.assertEqual(self.post('/api/complete',{'feedback':''},csrf,client).status_code,409)
            for i in [1,2]:
                self.assertEqual(self.post('/api/answer',{'index':i,'A-B':-.3,'B-C':.2,'C-A':0},csrf,client).status_code,200)
                self.assertEqual(client.get('/api/session').json['answered'],i)
                self.assertEqual(self.post('/api/answer',{'index':i,'A-B':0,'B-C':0,'C-A':0},csrf,client).status_code,409)
            self.assertEqual(self.post('/api/complete',{'feedback':'=not-a-formula'},csrf,client).status_code,200)
            self.assertTrue(client.get('/api/session').json['completed'])
        output=self.root/'exports'/'form2.csv'
        self.assertEqual(export_csv(self.root,output),3)
        with output.open(encoding='utf-8-sig') as f:rows=list(csv.reader(f))
        self.assertEqual(rows[0],HEADERS);self.assertEqual(len(rows[0]),167)
        self.assertEqual(rows[1][-1],"'=not-a-formula")
        self.assertEqual(rows[1][10],'') # index3 not rated
        from analyze_relations import read_survey
        hs=read_survey(output,'AIー縛りなし',['1','2','3'])
        self.assertEqual(len(hs),18)
        self.assertEqual(next(r['human'] for r in hs if r['pair']=='A-B'),-.3)
        with self.assertRaisesRegex(ValueError,'回答開始済み'):register(self.root,dict(self.cfg,end_seconds=[2,3]))

    def test_boundary_and_invite_validation(self):
        for ends in [[None,2],[2,1],[0,2],[1,float('inf')]]:
            with self.assertRaises(ValueError):register(self.root,dict(self.cfg,end_seconds=ends))
        with self.assertRaises(ValueError):register(self.root,dict(self.cfg,video='../outside.mp4'))
        with self.assertRaises(Exception):invite(self.root,'test','1','A')
        csrf=self.login()
        with connect(self.root) as db:db.execute('UPDATE participants SET revoked=1 WHERE id=?',('1',))
        self.assertEqual(self.client.get('/media').status_code,401)

    def test_preview_is_not_exported(self):
        register(self.root,dict(self.cfg,id='preview',preview=True))
        code=invite(self.root,'preview','900001','A')
        csrf=self.login(code=code)
        self.assertTrue(self.client.get('/api/session').json['preview'])
        for i in [1,2]:
            self.assertEqual(self.post('/api/answer',{'index':i,'A-B':0,'B-C':0,'C-A':0},csrf).status_code,200)
        self.assertEqual(self.post('/api/complete',{'feedback':'test'},csrf).status_code,200)
        self.assertEqual(export_csv(self.root,self.root/'preview.csv'),0)

    def test_public_settings_fail_closed_and_login_rate_limit(self):
        with patch.dict(os.environ,{'QUESTIONNAIRE_PUBLIC':'1','QUESTIONNAIRE_SECRET_KEY':'','QUESTIONNAIRE_BASE_URL':''}):
            with self.assertRaises(RuntimeError):create_app(self.root)
        csrf=self.client.get('/api/session').json['csrf']
        for i in range(10):self.assertEqual(self.post('/api/login',{'code':'wrong'},csrf).status_code,401)
        self.assertEqual(self.post('/api/login',{'code':'wrong'},csrf).status_code,429)

if __name__=='__main__':unittest.main()
