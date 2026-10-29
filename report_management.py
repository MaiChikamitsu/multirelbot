"""Persistent reversible run exclusions for generated reports."""
import html
import json
from pathlib import Path


def exclusion_path(config_path):
    path=Path(config_path).resolve()
    cfg=json.loads(path.read_text())
    return path.parent/cfg.get('exclusions_file','analysis_exclusions.json')


def load_exclusions(config_path):
    path=exclusion_path(config_path)
    if not path.exists():return []
    records=json.loads(path.read_text())
    if not isinstance(records,list) or any(not isinstance(r,dict) or not isinstance(r.get('experiment'),str) or not isinstance(r.get('run'),str) for r in records):
        raise ValueError('Exclusions must be a list of experiment/run objects')
    return records


def button(experiment,run,restore=False):
    esc=lambda s:html.escape(s,quote=True)
    return '<button class="run-action" disabled data-experiment="'+esc(experiment)+'" data-run="'+esc(run)+'" data-action="'+('restore' if restore else 'exclude')+'">'+('一覧へ戻す' if restore else '一覧・総合評価から除外')+'</button>'


def controls(excluded):
    parts=['<p id="management-status">除外・復元ボタンを使うには python3 serve_relation_reports.py --config analysis_config.example.json で開いてください。元の推定ファイルは削除しません。</p>']
    if excluded:
        parts.append('<details><summary>除外済みの実行（復元できます）</summary>')
        for r in excluded:
            parts.append('<p>'+html.escape(r['experiment']+' / '+r['run'])+' '+button(r['experiment'],r['run'],True)+'</p>')
        parts.append('</details>')
    parts.append('''<script>
(async()=>{
 if(location.protocol!=='http:')return;
 const status=document.getElementById('management-status');
 try{
  const response=await fetch('/api/status',{cache:'no-store'});
  if(!response.ok)return;
  const state=await response.json();
  status.textContent='除外すると一覧・総合評価を再集計します。元の推定ファイルは保持し、除外済みの欄から復元できます。';
  document.querySelectorAll('.run-action').forEach(button=>{
   button.disabled=false;
   button.onclick=async()=>{
    const restoring=button.dataset.action==='restore';
    if(!confirm(button.dataset.experiment+' / '+button.dataset.run+' を'+(restoring?'復元':'一覧と総合評価から除外')+'しますか？'))return;
    document.querySelectorAll('.run-action').forEach(b=>b.disabled=true);
    status.textContent='再集計しています…';
    try{
     const r=await fetch('/api/run-visibility',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':state.token},body:JSON.stringify({experiment:button.dataset.experiment,run:button.dataset.run,action:button.dataset.action})});
     const result=await r.json();
     if(!r.ok)throw new Error(result.error||'更新できませんでした');
     location.reload();
    }catch(error){status.textContent=error.message;document.querySelectorAll('.run-action').forEach(b=>b.disabled=false);}
   };
  });
 }catch(error){}
})();
</script>''')
    return ''.join(parts)
