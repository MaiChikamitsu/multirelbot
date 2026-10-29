import csv
import secrets
import tempfile
import unittest
from pathlib import Path
from questionnaire.app import create_app
from questionnaire.storage import connect,digest,register
from questionnaire.observer_export import export_observers
from questionnaire.__main__ import export_csv

class ObserverTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'videos').mkdir();(self.root/'videos/sample.mp4').write_bytes(b'0123456789')
        self.cfg=dict(id='muzintou_ari',theme='無人島ー縛りあり',video='sample.mp4',start_seconds=.2,end_seconds=[1,2],observer_enabled=True)
        register(self.root,self.cfg)
        self.app=create_app(self.root,{'TESTING':True})
    def tearDown(self):self.tmp.cleanup()
    def login(self,preview=False):
        code=secrets.token_urlsafe(24)
        with connect(self.root) as db:db.execute('INSERT INTO observer_invites(token_hash,preview) VALUES(?,?)',(digest(code),int(preview)))
        client=self.app.test_client();csrf=client.get('/api/session').json['csrf']
        self.assertEqual(client.post('/api/login',json=dict(code=code),headers={'X-CSRF-Token':csrf}).status_code,200)
        return client,client.get('/api/session').json['csrf'],code
    def post(self,c,t,path,payload):return c.post(path,json=payload,headers={'X-CSRF-Token':t})
    def test_profile_themes_resume_export(self):
        c,t,code=self.login();state=c.get('/api/session').json
        self.assertTrue(state['needs_profile']);self.assertNotIn('role',state)
        self.assertEqual(len(state['choices']),6)
        self.assertEqual(sum(bool(x['experiments']) for x in state['choices']),1)
        self.assertEqual(c.get('/media').status_code,409)
        self.assertEqual(self.post(c,t,'/api/profile',{'participant':'900002'}).status_code,200)
        self.assertEqual(self.post(c,t,'/api/select-theme',{'experiment':'ai_nashi'}).status_code,404)
        self.assertEqual(self.post(c,t,'/api/select-theme',{'experiment':'muzintou_ari'}).status_code,200)
        state=c.get('/api/session').json
        self.assertEqual(state['participant'],'900002');self.assertEqual(state['review_start'],.2)
        self.assertNotIn('role',state)
        for i in [1,2]:
            self.assertEqual(self.post(c,t,'/api/answer',dict(index=i,**{'A-B':-.3,'B-C':.2,'C-A':0})).status_code,200)
        self.assertEqual(self.post(c,t,'/api/complete',{'feedback':'感想'}).status_code,200)
        self.assertEqual(export_observers(self.root,self.root/'observers.csv'),1)
        self.assertEqual(export_csv(self.root,self.root/'participants.csv'),0)
        with (self.root/'observers.csv').open(encoding='utf-8-sig') as f:rows=list(csv.reader(f))
        self.assertEqual(rows[1][1],'900002');self.assertEqual(rows[1][2],'');self.assertEqual(rows[1][-1],'muzintou_ari')
        self.assertEqual(self.post(c,t,'/api/themes',{}).status_code,200)
        self.assertEqual(self.post(c,t,'/api/select-theme',{'experiment':'muzintou_ari'}).status_code,200)
        self.assertTrue(c.get('/api/session').json['completed'])
        other,token,_=self.login()
        self.assertEqual(self.post(other,token,'/api/profile',{'participant':'900002'}).status_code,409)
        self.assertEqual(self.post(other,token,'/api/profile',{'participant':'900003'}).status_code,200)
        self.assertEqual(self.post(other,token,'/api/select-theme',{'experiment':'muzintou_ari'}).status_code,200)
        self.assertEqual(other.get('/api/session').json['answered'],0)

    def test_revisit_and_update(self):
        c,t,_=self.login()
        self.post(c,t,'/api/profile',{'participant':'123'})
        self.post(c,t,'/api/select-theme',{'experiment':'muzintou_ari'})
        payload=dict(index=1,**{'A-B':-.3,'B-C':.2,'C-A':0})
        self.assertEqual(self.post(c,t,'/api/answer',payload).status_code,200)
        state=c.get('/api/session?index=1').json
        self.assertEqual(state['rating']['A-B'],-.3)
        self.assertEqual(state['segment']['start'],.2)
        self.assertEqual(c.get('/api/session?index=3').status_code,409)
        payload['A-B']=.7
        self.assertEqual(self.post(c,t,'/api/answer',payload).status_code,200)
        self.assertEqual(c.get('/api/session?index=1').json['rating']['A-B'],.7)
        self.assertEqual(c.get('/api/session').json['answered'],1)
        payload['index']=2
        self.post(c,t,'/api/answer',payload)
        self.post(c,t,'/api/complete',{'feedback':'done'})
        self.assertEqual(self.post(c,t,'/api/answer',payload).status_code,200)
        self.assertTrue(c.get('/api/session').json['completed'])
        other,token,_=self.login()
        self.post(other,token,'/api/profile',{'participant':'124'})
        self.post(other,token,'/api/select-theme',{'experiment':'muzintou_ari'})
        self.assertIsNone(other.get('/api/session?index=1').json['rating'])
        self.assertEqual(self.post(other,token,'/api/answer',payload).status_code,409)

    def test_text_mode_isolation_and_history(self):
        cfg=dict(self.cfg,id='muzintou_ari_text',modality='text',turns=[dict(speaker=x,text=str(i)) for i,x in enumerate('ABCABC')])
        cfg.pop('video')
        register(self.root,cfg)
        c,t,_=self.login()
        self.post(c,t,'/api/profile',{'participant':'222'})
        self.assertEqual(self.post(c,t,'/api/select-theme',{'experiment':'muzintou_ari_text'}).status_code,404)
        self.assertEqual(self.post(c,t,'/api/select-theme?mode=text',{'experiment':'muzintou_ari_text'}).status_code,200)
        state=c.get('/api/session?mode=text').json
        self.assertEqual(len(state['turns']),3)
        self.assertEqual(state['modality'],'text')
        self.assertIsNone(c.get('/api/session').json['experiment'])
        self.assertEqual(c.get('/media?mode=text').status_code,404)
        self.post(c,t,'/api/answer?mode=text',dict(index=1,**{'A-B':.1,'B-C':0,'C-A':-.1}))
        self.assertEqual(len(c.get('/api/session?mode=text').json['turns']),6)
        self.assertEqual(len(c.get('/api/session?mode=text&index=1').json['turns']),3)
        html=c.get('/text').get_data(as_text=True)
        self.assertIn('data-mode="text"',html)
        self.assertIn('現在読んだ会話まで',html)

    def test_preview_and_precision(self):
        c,t,_=self.login(preview=True)
        self.post(c,t,'/api/profile',{'participant':'900001'})
        self.post(c,t,'/api/select-theme',{'experiment':'muzintou_ari'})
        self.assertEqual(self.post(c,t,'/api/answer',dict(index=1,**{'A-B':.25,'B-C':0,'C-A':0})).status_code,400)
        for i in [1,2]:self.assertEqual(self.post(c,t,'/api/answer',dict(index=i,**{'A-B':0,'B-C':0,'C-A':0})).status_code,200)
        self.post(c,t,'/api/complete',{'feedback':'確認'})
        self.assertEqual(export_observers(self.root,self.root/'preview.csv'),0)

if __name__=='__main__':unittest.main()
