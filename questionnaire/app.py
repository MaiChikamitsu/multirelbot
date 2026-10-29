import json
import math
import os
import secrets
import time
from pathlib import Path
from urllib.parse import urlparse
from flask import Flask, abort, jsonify, render_template, request, send_file, session
from werkzeug.middleware.proxy_fix import ProxyFix
from .storage import connect, config_for, digest, now, validate_config


def create_app(data_dir=None, test_config=None):
    data_dir=Path(data_dir or os.environ.get('QUESTIONNAIRE_DATA_DIR',Path(__file__).parent/'private')).resolve()
    data_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
    app=Flask(__name__)
    public=os.environ.get('QUESTIONNAIRE_PUBLIC')=='1'
    key=os.environ.get('QUESTIONNAIRE_SECRET_KEY')
    base_url=os.environ.get('QUESTIONNAIRE_BASE_URL','')
    if public and (not key or len(key)<32 or key.startswith('REPLACE_') or urlparse(base_url).scheme!='https' or not urlparse(base_url).netloc):
        raise RuntimeError('公開時は32文字以上のQUESTIONNAIRE_SECRET_KEYとHTTPSのQUESTIONNAIRE_BASE_URLが必要です')
    if not key:
        keyfile=data_dir/'.session_secret'
        if not keyfile.exists():
            try:
                fd=os.open(keyfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(fd,'w') as f:f.write(secrets.token_hex(32))
            except FileExistsError:pass
        key=keyfile.read_text().strip()
    app.config.update(SECRET_KEY=key,SESSION_COOKIE_NAME='questionnaire_session',SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Strict',SESSION_COOKIE_SECURE=public,PERMANENT_SESSION_LIFETIME=60*60*12,
        MAX_CONTENT_LENGTH=16384,DATA_DIR=data_dir)
    if test_config:app.config.update(test_config)
    if public:app.wsgi_app=ProxyFix(app.wsgi_app,x_for=1,x_proto=1,x_host=1)

    def db():return connect(data_dir)
    def participant(connection):
        pid=session.get('participant')
        row=connection.execute('SELECT * FROM participants WHERE id=? AND revoked=0',(pid,)).fetchone()
        if not row or session.get('auth_version')!=row['auth_version']:abort(401)
        return row
    def observer(connection):
        from .observers import get_observer
        invite,person=get_observer(connection,session.get('observer_invite'))
        if not invite:abort(401)
        return invite,person

    def selected_key():return 'observer_text_experiment' if request.args.get('mode')=='text' else 'observer_experiment'

    def selected_observer(connection):
        invite,person=observer(connection)
        experiment=session.get(selected_key())
        if not person or not experiment:abort(409)
        item=connection.execute('SELECT * FROM observer_sessions WHERE observer=? AND experiment=?',(person['id'],experiment)).fetchone()
        if not item:abort(409)
        return person,experiment,item

    def progress(connection,pid):
        indices=[r[0] for r in connection.execute('SELECT idx FROM answers WHERE participant=? ORDER BY idx',(pid,))]
        if indices!=list(range(1,len(indices)+1)):abort(409)
        return len(indices)

    @app.before_request
    def security():
        if public and (request.host!=urlparse(base_url).netloc or not request.is_secure):abort(400)
        if request.method=='POST':
            if not request.is_json:abort(415)
            supplied=request.headers.get('X-CSRF-Token','')
            if not session.get('csrf') or not secrets.compare_digest(supplied,session['csrf']):abort(403)

    @app.after_request
    def headers(response):
        response.headers['Cache-Control']='private, no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['X-Frame-Options']='DENY'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        if public:response.headers['Strict-Transport-Security']='max-age=31536000'
        return response

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(413)
    @app.errorhandler(415)
    @app.errorhandler(429)
    def error(exc):return jsonify(error={401:'招待コードで入り直してください',409:'保存状態が変わりました。画面を再読み込みしてください',429:'試行回数が多いため、10分ほど待ってください'}.get(exc.code,'リクエストを受け付けられませんでした')),exc.code

    @app.get('/text')
    @app.get('/')
    def index():
        session.setdefault('csrf',secrets.token_urlsafe(24))
        return render_template('index.html',text_mode=request.path=='/text')

    def viewing(connection,data):
        raw=request.args.get('index')
        data['rating']=None
        if raw is None:
            if data.get('modality')=='text' and data.get('segment'):data['turns']=config_for(connection,data['experiment'])['turns'][:data['segment']['index']*3]
            return data
        try:idx=int(raw)
        except ValueError:abort(400)
        if not data.get('experiment') or not 1<=idx<=min(data['answered']+1,data['total']):abort(409)
        cfg=config_for(connection,data['experiment'])
        if data.get('observer'):
            saved=connection.execute('SELECT ab,bc,ca FROM observer_answers WHERE observer=? AND experiment=? AND idx=?',(data['participant'],data['experiment'],idx)).fetchone()
        else:
            saved=connection.execute('SELECT ab,bc,ca FROM answers WHERE participant=? AND idx=?',(data['participant'],idx)).fetchone()
        data['segment']=dict(index=idx,start=cfg.get('start_seconds',0) if idx==1 else cfg['end_seconds'][idx-2],end=cfg['end_seconds'][idx-1])
        data['rating']=dict(zip(['A-B','B-C','C-A'],saved)) if saved else None
        if data.get('modality')=='text':data['turns']=cfg['turns'][:idx*3]
        data['completed']=False
        return data

    @app.get('/api/session')
    def snapshot():
        session.setdefault('csrf',secrets.token_urlsafe(24))
        with db() as connection:
            # Migrate the known legacy preview login without touching its answers.
            if session.get('participant') and not session.get('observer_invite'):
                legacy=connection.execute('SELECT token_hash FROM participants WHERE id=?',(session['participant'],)).fetchone()
                if legacy and connection.execute('SELECT 1 FROM observer_invites WHERE token_hash=? AND revoked=0',(legacy['token_hash'],)).fetchone():
                    session.pop('participant',None);session['observer_invite']=legacy['token_hash']
            if session.get('observer_invite'):
                from .observers import snapshot
                data=snapshot(connection,session['observer_invite'],session.get(selected_key()),request.args.get('mode','video'))
                if data is None:
                    session.clear();session['csrf']=secrets.token_urlsafe(24)
                    return jsonify(authenticated=False,csrf=session['csrf'])
                return jsonify(dict(viewing(connection,data),csrf=session['csrf']))
            if not session.get('participant'):return jsonify(authenticated=False,csrf=session['csrf'])
            candidate=connection.execute('SELECT * FROM participants WHERE id=? AND revoked=0',(session['participant'],)).fetchone()
            if not candidate or candidate['auth_version']!=session.get('auth_version'):
                session.clear();session['csrf']=secrets.token_urlsafe(24)
                return jsonify(authenticated=False,csrf=session['csrf'])
            p=candidate;cfg=config_for(connection,p['experiment'])
            count=progress(connection,p['id']);total=len(cfg['end_seconds'])
            segment=None
            if count<total:
                segment=dict(index=count+1,start=cfg.get('start_seconds',0) if count==0 else cfg['end_seconds'][count-1],end=cfg['end_seconds'][count])
            return jsonify(authenticated=True,csrf=session['csrf'],participant=p['id'],role=p['role'],theme=cfg['theme'],
                experiment=p['experiment'],review_start=cfg.get('start_seconds',0),preview=bool(cfg.get('preview',False)),answered=count,total=total,segment=segment,completed=bool(p['completed_at']),feedback=p['feedback'])

    @app.post('/api/login')
    def login():
        payload=request.get_json()
        if not isinstance(payload,dict) or not isinstance(payload.get('code'),str) or len(payload['code'])>200:abort(400)
        ip=digest(request.remote_addr or 'unknown');timestamp=time.time()
        with db() as connection:
            connection.execute('DELETE FROM login_attempts WHERE at<?',(timestamp-600,))
            if connection.execute('SELECT COUNT(*) FROM login_attempts WHERE ip_hash=?',(ip,)).fetchone()[0]>=10:abort(429)
            connection.execute('INSERT INTO login_attempts VALUES(?,?)',(ip,timestamp))
            token_hash=digest(payload['code'].strip())
            invitation=connection.execute('SELECT * FROM observer_invites WHERE token_hash=? AND revoked=0',(token_hash,)).fetchone()
            p=connection.execute('SELECT * FROM participants WHERE token_hash=? AND revoked=0',(token_hash,)).fetchone()
        if invitation:
            session.clear();session['observer_invite']=token_hash;session['csrf']=secrets.token_urlsafe(24);session.permanent=True
            return jsonify(ok=True)
        # Legacy role-bound codes remain valid for existing participants.
        if not p:abort(401)
        session.clear();session['participant']=p['id'];session['auth_version']=p['auth_version'];session['csrf']=secrets.token_urlsafe(24);session.permanent=True
        return jsonify(ok=True)

    @app.post('/api/profile')
    def profile():
        body=request.get_json()
        if not isinstance(body,dict) or not isinstance(body.get('participant'),str):abort(400)
        import re
        value=body['participant'].strip()
        if not re.fullmatch('[0-9]{1,30}',value):abort(400)
        value=str(int(value))
        with db() as connection:
            connection.execute('BEGIN IMMEDIATE')
            invite,person=observer(connection)
            if person and person['id']!=value:abort(409)
            other=connection.execute('SELECT invite_hash FROM observers WHERE id=?',(value,)).fetchone()
            if other and other['invite_hash']!=session['observer_invite']:abort(409)
            if not person:connection.execute('INSERT INTO observers VALUES(?,?)',(value,session['observer_invite']))
        return jsonify(ok=True)

    @app.post('/api/select-theme')
    def select_theme():
        body=request.get_json()
        if not isinstance(body,dict) or not isinstance(body.get('experiment'),str):abort(400)
        with db() as connection:
            invite,person=observer(connection)
            if not person:abort(409)
            from .observers import available
            allowed={e['id'] for choice in available(connection,request.args.get('mode','video')) for e in choice['experiments']}
            experiment=body['experiment']
            if experiment not in allowed:abort(404)
            connection.execute('INSERT OR IGNORE INTO observer_sessions(observer,experiment,preview) VALUES(?,?,?)',(person['id'],experiment,invite['preview']))
            session[selected_key()]=experiment
        return jsonify(ok=True)

    @app.post('/api/themes')
    def themes():
        with db() as connection:observer(connection)
        session.pop(selected_key(),None)
        return jsonify(ok=True)

    @app.post('/api/logout')
    def logout():session.clear();return jsonify(ok=True)

    @app.get('/media')
    def media():
        with db() as connection:
            if session.get('observer_invite'):
                person,experiment,item=selected_observer(connection)
                cfg=config_for(connection,experiment)
                if cfg.get('modality')=='text':abort(404)
            else:
                p=participant(connection);cfg=config_for(connection,p['experiment'])
            video=validate_config(cfg,data_dir/'videos')
        return send_file(video,conditional=True,download_name='conversation'+video.suffix)

    @app.post('/api/answer')
    def answer():
        body=request.get_json()
        if not isinstance(body,dict):abort(400)
        values=[]
        for pair in ['A-B','B-C','C-A']:
            value=body.get(pair)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not -1<=value<=1 or abs(value*10-round(value*10))>1e-8:abort(400)
            values.append(value)
        idx=body.get('index')
        if isinstance(idx,bool) or not isinstance(idx,int):abort(400)
        with db() as connection:
            connection.execute('BEGIN IMMEDIATE')
            if session.get('observer_invite'):
                person,experiment,item=selected_observer(connection)
                cfg=config_for(connection,experiment)
                from .observers import progress as observer_progress
                if not 1<=idx<=min(observer_progress(connection,person['id'],experiment)+1,len(cfg['end_seconds'])):abort(409)
                connection.execute('INSERT INTO observer_answers VALUES(?,?,?,?,?,?,?) ON CONFLICT(observer,experiment,idx) DO UPDATE SET ab=excluded.ab,bc=excluded.bc,ca=excluded.ca,saved_at=excluded.saved_at',(person['id'],experiment,idx,*values,now()))
                return jsonify(ok=True)
            p=participant(connection);cfg=config_for(connection,p['experiment'])
            if p['completed_at'] or idx!=progress(connection,p['id'])+1 or idx>len(cfg['end_seconds']):abort(409)
            connection.execute('INSERT INTO answers VALUES(?,?,?,?,?,?)',(p['id'],idx,*values,now()))
        return jsonify(ok=True)

    @app.post('/api/complete')
    def complete():
        body=request.get_json()
        if not isinstance(body,dict):abort(400)
        text=body.get('feedback','')
        if not isinstance(text,str) or len(text)>5000:abort(400)
        with db() as connection:
            connection.execute('BEGIN IMMEDIATE')
            if session.get('observer_invite'):
                person,experiment,item=selected_observer(connection)
                cfg=config_for(connection,experiment)
                from .observers import progress as observer_progress
                if observer_progress(connection,person['id'],experiment)!=len(cfg['end_seconds']):abort(409)
                if not item['completed_at']:connection.execute('UPDATE observer_sessions SET feedback=?,completed_at=? WHERE observer=? AND experiment=?',(text,now(),person['id'],experiment))
                return jsonify(ok=True)
            p=participant(connection);cfg=config_for(connection,p['experiment'])
            if progress(connection,p['id'])!=len(cfg['end_seconds']):abort(409)
            if not p['completed_at']:
                connection.execute('UPDATE participants SET feedback=?,completed_at=? WHERE id=?',(text,now(),p['id']))
        return jsonify(ok=True)
    return app
