import hashlib
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def now(): return datetime.now(timezone.utc).isoformat()
def digest(code): return hashlib.sha256(code.encode()).hexdigest()


class Connection(sqlite3.Connection):
    def __exit__(self,*args):
        try:return super().__exit__(*args)
        finally:self.close()


def connect(data_dir):
    data_dir=Path(data_dir);data_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
    db=sqlite3.connect(data_dir/'responses.sqlite3',timeout=15,factory=Connection)
    os.chmod(data_dir/'responses.sqlite3',0o600)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY, config TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS participants(
      id TEXT PRIMARY KEY, experiment TEXT NOT NULL REFERENCES experiments(id),
      role TEXT NOT NULL CHECK(role IN ('A','B','C')), token_hash TEXT UNIQUE NOT NULL,
      revoked INTEGER NOT NULL DEFAULT 0, auth_version INTEGER NOT NULL DEFAULT 0, feedback TEXT NOT NULL DEFAULT '', completed_at TEXT,
      UNIQUE(experiment,role));
    CREATE TABLE IF NOT EXISTS answers(
      participant TEXT NOT NULL REFERENCES participants(id), idx INTEGER NOT NULL,
      ab REAL NOT NULL, bc REAL NOT NULL, ca REAL NOT NULL, saved_at TEXT NOT NULL,
      PRIMARY KEY(participant,idx));
    CREATE TABLE IF NOT EXISTS login_attempts(ip_hash TEXT NOT NULL, at REAL NOT NULL);
    ''')
    from .observers import init_tables
    init_tables(db)
    return db


def config_for(db, experiment):
    row=db.execute('SELECT config FROM experiments WHERE id=?',(experiment,)).fetchone()
    if not row: raise ValueError('実験が見つかりません')
    return json.loads(row['config'])


def validate_config(config, video_root):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',config['id']):raise ValueError('実験IDは英数字・_・-で指定してください')
    if config.get('modality')=='text':
        turns=config.get('turns',[])
        if not turns or len(turns)%3 or len(turns)//3!=len(config['end_seconds']) or any(t.get('speaker') not in 'ABC' or not t.get('text') for t in turns):raise ValueError('文字版はA/B/Cの3発話単位で登録してください')
        return None
    video_root=Path(video_root).resolve()
    video=(video_root/config['video']).resolve()
    if video_root not in video.parents or not video.is_file():raise ValueError('動画はprivate/videos配下の実在するファイルを指定してください')
    if video.suffix.lower() not in ['.mp4','.webm']:raise ValueError('ブラウザ再生用のMP4またはWebMを指定してください')
    boundaries=config['end_seconds']
    if not 1<=len(boundaries)<=54:raise ValueError('評定時点は1〜54件にしてください')
    previous=0
    import math
    for value in boundaries:
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<=previous:
            raise ValueError('end_secondsは正の秒数を昇順で入力してください。未入力の区切りは使用できません')
        previous=value
    start=config.get('start_seconds',0)
    if isinstance(start,bool) or not isinstance(start,(int,float)) or not math.isfinite(start) or start<0 or start>=boundaries[0]:raise ValueError('start_secondsが不正です')
    if not config.get('theme'):raise ValueError('テーマが必要です')
    return video


def register(data_dir, config):
    data_dir=Path(data_dir)
    validate_config(config,data_dir/'videos')
    with connect(data_dir) as db:
        old=db.execute('SELECT config FROM experiments WHERE id=?',(config['id'],)).fetchone()
        if old and db.execute('SELECT 1 FROM answers a JOIN participants p ON p.id=a.participant WHERE p.experiment=? LIMIT 1',(config['id'],)).fetchone():
            if json.loads(old['config'])!=config:raise ValueError('回答開始済みの実験設定は変更できません。別IDで登録してください')
        db.execute('INSERT INTO experiments(id,config) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET config=excluded.config',(config['id'],json.dumps(config,ensure_ascii=False)))


def invite(data_dir, experiment, participant_id, role):
    participant_id=str(participant_id).strip()
    if not re.fullmatch(r'[0-9]{1,30}',participant_id):raise ValueError('回答者IDは分析に合わせて数字で指定してください')
    participant_id=str(int(participant_id))
    if role not in 'ABC' or len(role)!=1:raise ValueError('役割はA/B/Cです')
    token=secrets.token_urlsafe(24)
    with connect(data_dir) as db:
        config_for(db,experiment)
        db.execute('INSERT INTO participants(id,experiment,role,token_hash) VALUES(?,?,?,?)',(participant_id,experiment,role,digest(token)))
    return token
