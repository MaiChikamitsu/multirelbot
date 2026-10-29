"""Third-party observers, kept separate from legacy A/B/C participants."""
import json
from .schema import THEMES
from .storage import config_for, now


def init_tables(db):
    db.executescript('''
    CREATE TABLE IF NOT EXISTS observer_invites(token_hash TEXT PRIMARY KEY, preview INTEGER NOT NULL DEFAULT 0, revoked INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS observers(id TEXT PRIMARY KEY, invite_hash TEXT UNIQUE NOT NULL REFERENCES observer_invites(token_hash));
    CREATE TABLE IF NOT EXISTS observer_sessions(observer TEXT NOT NULL REFERENCES observers(id), experiment TEXT NOT NULL REFERENCES experiments(id), preview INTEGER NOT NULL, feedback TEXT NOT NULL DEFAULT '', completed_at TEXT, PRIMARY KEY(observer,experiment));
    CREATE TABLE IF NOT EXISTS observer_answers(observer TEXT NOT NULL, experiment TEXT NOT NULL, idx INTEGER NOT NULL, ab REAL NOT NULL, bc REAL NOT NULL, ca REAL NOT NULL, saved_at TEXT NOT NULL, PRIMARY KEY(observer,experiment,idx), FOREIGN KEY(observer,experiment) REFERENCES observer_sessions(observer,experiment));
    ''')


def available(db, modality='video'):
    choices=[]
    for theme in THEMES:
        experiments=[]
        for row in db.execute('SELECT id,config FROM experiments ORDER BY id'):
            cfg=json.loads(row['config'])
            if cfg.get('modality','video')==modality and cfg.get('observer_enabled',False) and not cfg.get('preview') and cfg['theme']==theme:
                experiments.append(dict(id=row['id'],label=cfg.get('label',row['id'])))
        choices.append(dict(theme=theme,experiments=experiments))
    return choices


def progress(db, observer, experiment):
    indices=[r[0] for r in db.execute('SELECT idx FROM observer_answers WHERE observer=? AND experiment=? ORDER BY idx',(observer,experiment))]
    if indices!=list(range(1,len(indices)+1)):raise ValueError('回答番号が連続していません')
    return len(indices)


def get_observer(db, invite_hash):
    invite=db.execute('SELECT * FROM observer_invites WHERE token_hash=? AND revoked=0',(invite_hash,)).fetchone()
    if not invite:return None,None
    person=db.execute('SELECT * FROM observers WHERE invite_hash=?',(invite_hash,)).fetchone()
    return invite,person


def snapshot(db, invite_hash, experiment, modality='video'):
    invite,person=get_observer(db,invite_hash)
    if not invite:return None
    result=dict(authenticated=True,observer=True,preview=bool(invite['preview']),needs_profile=person is None,
                participant=person['id'] if person else None,choices=available(db,modality),experiment=None,segment=None)
    if not person or not experiment:return result
    item=db.execute('SELECT * FROM observer_sessions WHERE observer=? AND experiment=?',(person['id'],experiment)).fetchone()
    if not item:return result
    cfg=config_for(db,experiment)
    if cfg.get('modality','video')!=modality:return result
    count=progress(db,person['id'],experiment)
    result.update(experiment=experiment,theme=cfg['theme'],answered=count,total=len(cfg['end_seconds']),completed=bool(item['completed_at']),feedback=item['feedback'],review_start=cfg.get('start_seconds',0),modality=cfg.get('modality','video'),theme_description=cfg.get('theme_description',[]))
    if count<len(cfg['end_seconds']):
        result['segment']=dict(index=count+1,start=cfg.get('start_seconds',0) if count==0 else cfg['end_seconds'][count-1],end=cfg['end_seconds'][count])
    return result
