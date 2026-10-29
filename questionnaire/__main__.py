import argparse
import csv
import hashlib
import json
import re
import secrets
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from .schema import HEADERS, THEMES
from .storage import connect, config_for, invite, register


def export_csv(data_dir, output, experiment=None):
    from .schema import PAIRS
    rows=[]
    with connect(data_dir) as db:
        people=db.execute('SELECT * FROM participants WHERE completed_at IS NOT NULL ORDER BY experiment,CAST(id AS INTEGER)').fetchall()
        for p in people:
            if experiment and p['experiment']!=experiment:continue
            cfg=config_for(db,p['experiment'])
            if cfg.get('preview',False):continue
            answers={r['idx']:r for r in db.execute('SELECT * FROM answers WHERE participant=?',(p['id'],))}
            timestamp=datetime.fromisoformat(p['completed_at']).astimezone(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M:%S')
            row=[timestamp,p['id'],p['role'],cfg['theme']]
            for i in range(1,55):
                a=answers.get(i)
                row.extend([a['ab'],a['bc'],a['ca']] if a else ['','',''])
            # Spreadsheet formula injection is prevented in free-text export.
            feedback=p['feedback']
            if feedback.lstrip().startswith(('=','+','-','@')):feedback="'"+feedback
            row.append(feedback);rows.append(row)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix(output.suffix+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(HEADERS);writer.writerows(rows)
    temp.replace(output)
    return len(rows)


def main():
    p=argparse.ArgumentParser(description='動画アンケート。管理操作は端末のみ。公開Webに管理画面はありません。')
    p.add_argument('--data-dir',type=Path,default=Path(__file__).parent/'private')
    sub=p.add_subparsers(dest='action',required=True)
    prepare=sub.add_parser('prepare',help='会話から3発話ごとの区切りテンプレートを作成')
    prepare.add_argument('--id',required=True);prepare.add_argument('--video',required=True,help='private/videos内のファイル名')
    prepare.add_argument('--conversation',required=True,type=Path);prepare.add_argument('--theme',required=True,choices=THEMES)
    reg=sub.add_parser('register',help='区切り時刻を確認済みの実験を登録')
    reg.add_argument('--config',required=True,type=Path);reg.add_argument('--boundaries',required=True,type=Path)
    inv=sub.add_parser('invite');inv.add_argument('--experiment',required=True);inv.add_argument('--id',required=True);inv.add_argument('--role',required=True,choices=list('ABC'))
    ob=sub.add_parser('invite-observer',help='第三者用の招待コード。IDは本人入力、役割なし');ob.add_argument('--preview',action='store_true')
    rob=sub.add_parser('revoke-observer');rob.add_argument('--id',required=True)
    obs=sub.add_parser('export-observers',help='第三者評定を当事者とは別のCSVへ出力');obs.add_argument('--output',required=True,type=Path)
    revoke=sub.add_parser('revoke');revoke.add_argument('--id',required=True)
    rotate=sub.add_parser('rotate-code');rotate.add_argument('--id',required=True)
    exp=sub.add_parser('export');exp.add_argument('--output',required=True,type=Path);exp.add_argument('--experiment')
    serve=sub.add_parser('serve');serve.add_argument('--host',default='127.0.0.1');serve.add_argument('--port',type=int,default=8892)
    args=p.parse_args();root=args.data_dir.resolve()
    for folder in ['videos','configs','exports']:(root/folder).mkdir(parents=True,exist_ok=True)
    if args.action=='prepare':
        if not re.fullmatch('[A-Za-z0-9_-]{1,80}',args.id):p.error('idは英数字と_・-です')
        logs=[]
        for line in args.conversation.read_text(encoding='utf-8').splitlines():
            m=re.match(r'^\[(.+?)\]\s*(.+)$',line.strip())
            if m:logs.append(dict(speaker=m[1].strip(),text=m[2].strip()))
        human=[r for r in logs if r['speaker']!='ロボット']
        n=len(human)//3
        if not 1<=n<=54:p.error('この質問票は1〜54時点です')
        if set(r['speaker'] for r in human)!=set('ABC'):p.error('話者A/B/Cの会話が必要です')
        cfg=dict(id=args.id,episode=args.conversation.stem,theme=args.theme,video=args.video,
                 conversation_sha256=hashlib.sha256(args.conversation.read_bytes()).hexdigest(),
                 step=3,start_seconds=0,end_seconds=[None]*n)
        dest=root/'configs'/f'{args.id}.json'
        with dest.open('x',encoding='utf-8') as f:json.dump(cfg,f,ensure_ascii=False,indent=2)
        bounds=root/'configs'/f'{args.id}_boundaries.csv'
        with bounds.open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['index','human_end_turn','end_seconds','last_speaker','last_utterance'])
            for i in range(n):writer.writerow([i+1,(i+1)*3,'',human[(i+1)*3-1]['speaker'],human[(i+1)*3-1]['text']])
        print(f'設定: {dest}\n区切り時刻を入力: {bounds}\n冒頭に待機時間があれば設定のstart_secondsを調整してください。')
    elif args.action=='register':
        cfg=json.loads(args.config.read_text())
        with args.boundaries.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        if [int(r['index']) for r in rows]!=list(range(1,len(cfg['end_seconds'])+1)):p.error('区切り番号・件数が会話と一致しません')
        if any(int(r['human_end_turn'])!=3*int(r['index']) for r in rows):p.error('人間発話数が不正です')
        try:cfg['end_seconds']=[float(r['end_seconds']) for r in rows]
        except ValueError:p.error('全区切りのend_secondsへ秒数を入力してください')
        register(root,cfg);print('登録しました。動画の境界と音声を実際に再生して確認してください。')
    elif args.action=='invite-observer':
        from .storage import digest
        code=secrets.token_urlsafe(24)
        with connect(root) as db:db.execute('INSERT INTO observer_invites(token_hash,preview) VALUES(?,?)',(digest(code),int(args.preview)))
        print(code)
        print('1人に1コードを渡してください。IDは回答画面で本人が入力します。')
    elif args.action=='revoke-observer':
        with connect(root) as db:
            row=db.execute('SELECT invite_hash FROM observers WHERE id=?',(str(int(args.id)),)).fetchone()
            if not row:p.error('回答者IDが見つかりません')
            db.execute('UPDATE observer_invites SET revoked=1 WHERE token_hash=?',(row['invite_hash'],))
        print('第三者用コードを失効しました。回答は保持されます。')
    elif args.action=='export-observers':
        from .observer_export import export_observers
        print(f'{export_observers(root,args.output)}件の第三者評定を出力: {args.output}')
    elif args.action=='invite':
        print(invite(root,args.experiment,args.id,args.role))
        print('上の招待コードを該当参加者だけに渡してください。コードは後から再表示できません。')
    elif args.action in ['revoke','rotate-code']:
        with connect(root) as db:
            if not db.execute('SELECT 1 FROM participants WHERE id=?',(args.id,)).fetchone():p.error('IDが見つかりません')
            if args.action=='revoke':db.execute('UPDATE participants SET revoked=1 WHERE id=?',(args.id,));print('失効しました')
            else:
                from .storage import digest
                token=secrets.token_urlsafe(24)
                db.execute('UPDATE participants SET token_hash=?,revoked=0,auth_version=auth_version+1 WHERE id=?',(digest(token),args.id));print(token)
    elif args.action=='export':print(f'{export_csv(root,args.output,args.experiment)}名の完了回答を出力: {args.output}')
    elif args.action=='serve':
        from .app import create_app
        print(f'回答画面: http://{args.host}:{args.port}/')
        create_app(root).run(host=args.host,port=args.port,debug=False)

if __name__=='__main__':main()
