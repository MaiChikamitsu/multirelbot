import csv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from .schema import HEADERS
from .storage import connect,config_for


def export_observers(data_dir,output):
    rows=[]
    with connect(data_dir) as db:
        for p in db.execute('SELECT * FROM observer_sessions WHERE completed_at IS NOT NULL AND preview=0 ORDER BY experiment,observer').fetchall():
            cfg=config_for(db,p['experiment'])
            if cfg.get('preview'):continue
            values={r['idx']:r for r in db.execute('SELECT * FROM observer_answers WHERE observer=? AND experiment=?',(p['observer'],p['experiment']))}
            timestamp=datetime.fromisoformat(p['completed_at']).astimezone(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M:%S')
            row=[timestamp,p['observer'],'',cfg['theme']]
            for i in range(1,55):
                a=values.get(i);row.extend([a['ab'],a['bc'],a['ca']] if a else ['','',''])
            feedback=p['feedback']
            if feedback.lstrip().startswith(('=','+','-','@')):feedback="'"+feedback
            row.extend([feedback,p['experiment']]);rows.append(row)
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(HEADERS+['実験ID']);writer.writerows(rows)
    temp.replace(path)
    return len(rows)
