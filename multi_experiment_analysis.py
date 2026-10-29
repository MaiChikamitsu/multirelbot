"""Explicit session/run selection and condition-specific pooled/macro evaluation."""
import html
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

SETTING_KEYS = ['model', 'step', 'window_size', 'sma_window', 'num_trials', 'gamma',
                'max_history_sessions', 'relation_temperature']


def aggregate_correlations(groups, mode, x_key='prediction', y_key='human'):
    from analyze_relations import corr, ranks
    def values(group):
        x=[r[x_key] for r in group]; y=[r[y_key] for r in group]
        return corr(x,y), corr(ranks(x),ranks(y))
    per_experiment=[values(group) for group in groups.values()]
    valid_p=[p for p,_ in per_experiment if p is not None]
    valid_s=[s for _,s in per_experiment if s is not None]
    if mode=='pooled_points':
        p,s=values([r for group in groups.values() for r in group])
    else:
        p=st.mean(valid_p) if valid_p else None
        s=st.mean(valid_s) if valid_s else None
    return dict(pearson=p,spearman=s,pearson_valid_experiments=len(valid_p),spearman_valid_experiments=len(valid_s))


def build_overall(cfg, root, humans, predictions, comparisons, changes):
    selected, selection, signatures = {}, [], {}
    for exp in cfg['experiments']:
        eid = exp['id']
        run = exp.get('aggregate_run', '')
        condition = exp.get('condition_id', '')
        base = dict(experiment=eid, episode=exp.get('episode',eid), theme=exp['theme'],
                    respondents=','.join(map(str,exp.get('respondent_ids',[]))), run=run, condition=condition)
        if not exp.get('enabled', True):
            selection.append(dict(base,status='pending_disabled')); continue
        if any(r['experiment']==eid and r['run']==run for r in cfg.get('_excluded_runs',[])):
            selection.append(dict(base,status='excluded_by_user')); continue
        if not any(p['experiment']==eid for p in predictions):
            selection.append(dict(base,status='excluded_no_predictions')); continue
        if not run:
            selection.append(dict(base,status='excluded_no_selected_run')); continue
        if not condition:
            raise ValueError(f'{eid}: condition_id is required when selecting aggregate_run')
        expected = {(h['index'],h['pair']) for h in humans if h['experiment']==eid}
        # Every rated point must have all three answers for fair overall comparison.
        raters = defaultdict(set)
        for h in humans:
            if h['experiment']==eid: raters[h['index'],h['pair']].add(h['respondent'])
        present = {(p['method'],p['index'],p['pair']) for p in predictions if p['experiment']==eid and p['run']==run}
        methods = ['Interaction Frequency','Stance-based','Sentiment-based','LLM-only','LLM + SMA','MAVeRD']
        needed = {(method,i,pair) for method in methods for i,pair in expected}
        if not expected or any(len(v)!=3 for v in raters.values()) or not needed.issubset(present):
            selection.append(dict(base,status='excluded_incomplete_run_or_ratings')); continue
        # Read metadata only from the explicitly selected run's CSV.
        import glob
        sources = []
        for file in map(Path, sorted(glob.glob(str(root/exp['predictions'])))):
            import csv
            with file.open(encoding='utf-8-sig',newline='') as f:
                if any(r['Episode']==exp.get('episode',eid) and (r.get('Run') or file.stem)==run for r in csv.DictReader(f)):
                    sources.append(file)
        metadata_paths = {p.parent/'metadata.json' for p in sources}
        if len(metadata_paths)!=1 or not all(p.exists() for p in metadata_paths):
            raise ValueError(f'{eid}: selected run needs one metadata.json alongside predictions.csv')
        meta = json.loads(next(iter(metadata_paths)).read_text())
        signature = {k:meta.get(k,1 if k=='num_trials' else None) for k in SETTING_KEYS}
        # condition_id records the prompt/experimental setting explicitly. Older
        # viewer metadata cannot automatically prove prompt identity.
        if condition in signatures and signatures[condition]!=signature:
            raise ValueError(f'{eid}: settings differ within condition {condition!r}; use separate condition_id values')
        signatures[condition]=signature
        selected[eid]=(run,condition)
        selection.append(dict(base,status='included'))
    grouped = defaultdict(list)
    for r in comparisons:
        chosen = selected.get(r['experiment'])
        if not chosen or r['run']!=chosen[0]: continue
        for pair_scope in ['all',r['pair']]:
            grouped[chosen[1],r['method'],r['target'],pair_scope].append(r)
    result=[]
    for (condition,method,target,pair),rows in sorted(grouped.items()):
        experiments=sorted({r['experiment'] for r in rows})
        # Differences were calculated within experiment/run/person/pair, so
        # neither an experiment boundary nor a missing index is bridged.
        deltas=[r for r in changes if r['experiment'] in experiments
                and r['run']==selected[r['experiment']][0] and r['method']==method
                and r['target']==target and (pair=='all' or r['pair']==pair)]
        by_exp={eid:[r for r in rows if r['experiment']==eid] for eid in experiments}
        delta_by_exp={eid:[r for r in deltas if r['experiment']==eid] for eid in experiments}
        def err(r): return r['prediction']-r['human']
        for mode in ['pooled_points','equal_experiments']:
            if mode=='pooled_points':
                mae=st.mean(abs(err(r)) for r in rows)
                mse=st.mean(err(r)**2 for r in rows)
                bias=st.mean(err(r) for r in rows)
                change_mae=st.mean(r['change_absolute_error'] for r in deltas) if deltas else None
            else:
                mae=st.mean(st.mean(abs(err(r)) for r in group) for group in by_exp.values())
                mse=st.mean(st.mean(err(r)**2 for r in group) for group in by_exp.values())
                bias=st.mean(st.mean(err(r) for r in group) for group in by_exp.values())
                available=[group for group in delta_by_exp.values() if group]
                change_mae=st.mean(st.mean(r['change_absolute_error'] for r in group) for group in available) if available else None
            associations=aggregate_correlations(by_exp,mode)
            delta_associations=aggregate_correlations(delta_by_exp,mode,'system_change','human_change')
            result.append(dict(condition=condition,method=method,target=target,pair=pair,
                aggregation=mode,n_experiments=len(experiments),n_points=len(rows),n_changes=len(deltas),
                n_change_experiments=sum(bool(g) for g in delta_by_exp.values()),mae=mae,
                change_mae=change_mae,rmse=math.sqrt(mse),bias=bias,**associations,
                change_pearson=delta_associations['pearson'],
                change_pearson_valid_experiments=delta_associations['pearson_valid_experiments']))
    return result,selection


def render_overall(out, rows, selection):
    esc=html.escape
    parts=['<!doctype html><html lang="ja"><meta charset="utf-8"><title>全実験の総合評価</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 16px;color:#182335}table{border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:9px;border:1px solid #dce2ea}th{background:#eaf0f8}section{overflow-x:auto;margin:30px 0}a{color:#2457bb}</style>',
        '<p><a href="report.html">← 実行ファイル別の分析</a></p><h1>全実験の総合評価</h1>',
        '<p>実験ごとに指定したrunを1つだけ採用し、条件別に集計しています。平均評定（mean）と各回答者（individual）は別評価です。</p>',
        '<p>全評価点（区間）で集計：各点を同じ重みで評価。会話を同じ重みで集計：会話別MAEを平均。変化量は会話間をつながず計算します。変化のない会話（有効区間0）の扱いは変化実験数で確認できます。</p>',
        '<p>旧データにプロンプト本文・ハッシュがないため、プロンプトの一致はcondition_idの指定で管理します。モデル名・推定間隔・窓・試行数・EMA等の設定差は自動検査します。</p>',
        '<p><a href="overall_metrics.csv">総合集計CSV</a> ／ <a href="experiment_selection.csv">採用・保留の一覧CSV</a></p>']
    included = [r for r in selection if r['status']=='included']
    parts.append('<section><h2>今回の総合値に含まれる実験</h2>')
    if not included:
        parts.append('<p>採用できる実験がありません。総合指標は未算出です。</p>')
    for r in included:
        parts.append('<p><strong>'+esc(r['condition'])+'</strong>：'+esc(r['episode']+'.txt')+' ／ '+esc(r['theme'])+' ／ 回答者ID '+esc(r['respondents'])+' ／ run '+esc(r['run'])+'</p>')
    parts.append('</section>')
    fields=['condition','method','target','aggregation','n_experiments','n_points','mae','change_mae','n_changes','n_change_experiments','rmse','bias','pearson','spearman','change_pearson']
    labels=['条件','手法','対象','集計方式','実験数','評価点数','MAE','変化量MAE','変化区間数','変化実験数','RMSE','bias','Pearson','Spearman','変化量Pearson']
    parts.append('<p>相関は−1〜＋1です。「全評価点」では全会話・全ペアの対応点をまとめた相関、「会話を同じ重み」では会話ごとの相関の算術平均を表示します。変化量Pearsonは前時点からの変化量同士の相関です。定数系列など未定義の会話は平均から除き、相関値の後ろに有効会話数を示します。相関は絶対値の一致を表すものではありません。</p>')
    parts.append('<section><h2>全ペアの総合指標</h2><table><tr>'+''.join('<th>'+h+'</th>' for h in labels)+'</tr>')
    for row in rows:
        if row['pair']!='all':continue
        cells=[]
        for k in fields:
            v=row[k]
            if k=='aggregation':v={'pooled_points':'全評価点・区間','equal_experiments':'会話を同じ重み'}[v]
            cell='—' if v is None else f'{v:.4f}' if isinstance(v,float) else esc(str(v))
            if k in ['pearson','spearman','change_pearson'] and row['aggregation']=='equal_experiments':
                cell += '（'+str(row[k+'_valid_experiments'])+'会話）'
            cells.append(cell)
        parts.append('<tr>'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
    parts.append('</table></section><section><h2>実験と採用runの対応</h2><table><tr><th>実験</th><th>会話ファイル名</th><th>テーマ</th><th>回答者ID</th><th>採用run</th><th>状態</th></tr>')
    status={'included':'総合評価に採用','excluded_by_user':'一覧から除外済み・総合評価に含めない','excluded_no_predictions':'評定取込済み・推定結果待ち','pending_disabled':'追加情報・データ待ち','excluded_no_selected_run':'採用run未指定','excluded_incomplete_run_or_ratings':'指定runまたは評定が未完了'}
    for r in selection:
        parts.append('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in [r['experiment'],r['episode'],r['theme'],r['respondents'],r['run'],status[r['status']]])+'</tr>')
    parts.append('</table></section></html>')
    (out/'overall_report.html').write_text(''.join(parts),encoding='utf-8')
    path=out/'report.html'
    text=path.read_text()
    text=text.replace('<h1>分析ファイル一覧</h1>', '<h1>分析ファイル一覧</h1><p><a href="overall_report.html">全実験の総合MAE・変化量MAEを見る →</a></p>',1)
    path.write_text(text,encoding='utf-8')
