"""Pool the selected AI and island runs without crossing conversation boundaries."""
import html
import statistics as st
from collections import defaultdict

from comparison_methods import COMPARISON_METHODS
from correlation_scatter import render_scatter
from correlation_overview import render_pooled_graphs

EXPERIMENTS = ('ai_nashi', 'muzintou_ari')


def build_combined(points, config, correlation, ranks):
    selected = {}
    conditions = set()
    experiments = {row['id']:row for row in config['experiments'] if row.get('enabled',True)}
    for experiment in EXPERIMENTS:
        if experiment not in experiments:
            return [],[],f'{experiment} が有効な設定にありません。'
        setting = experiments[experiment]
        if setting.get('condition_id'):
            conditions.add(setting['condition_id'])
        runs = {row['run'] for row in points if row['experiment']==experiment}
        run = setting.get('aggregate_run')
        if run is None and len(runs)==1:
            run = next(iter(runs))
        if run not in runs:
            return [],[],f'{experiment} の採用runを確認してください。除外済み・未指定・データなしの可能性があります。'
        selected[experiment] = run
    if len(conditions)>1:
        return [],[],'会話間でcondition_idが異なるため、合算比較を保留しています。'
    combined = []
    seen = set()
    for point in points:
        if selected.get(point['experiment'])!=point['run']:
            continue
        identity = tuple(point[key] for key in ['experiment','run','method','group','pair','index'])
        if identity in seen:
            raise ValueError(f'Duplicate combined point: {identity}')
        seen.add(identity)
        combined.append(dict(point,source_experiment=point['experiment'],source_run=point['run'],
                             experiment='combined_ai_muzintou',run='selected_runs'))
    grouped = defaultdict(list)
    for point in combined:
        grouped[point['method'],point['group']].append(point)
    metrics = []
    for (method,group),series in sorted(grouped.items()):
        available = {row['source_experiment'] for row in series}
        if available!=set(EXPERIMENTS):
            continue
        changes = []
        boundaries = defaultdict(list)
        for row in series:
            boundaries[row['source_experiment'],row['source_run'],row['pair']].append(row)
        for conversation in boundaries.values():
            ordered = sorted(conversation,key=lambda row:row['index'])
            for previous,current in zip(ordered,ordered[1:]):
                if current['index']==previous['index']+1:
                    changes.append(dict(human=current['human']-previous['human'],
                                        prediction=current['prediction']-previous['prediction']))
        for measure,subset in [('level',series),('change',changes)]:
            human = [row['human'] for row in subset]
            system = [row['prediction'] for row in subset]
            metrics.append(dict(experiment='combined_ai_muzintou',run='selected_runs',method=method,group=group,
                pair='all',target='mean',label='このグループの平均',measure=measure,n=len(subset),
                pearson=correlation(human,system),spearman=correlation(ranks(human),ranks(system)),
                mae=st.mean(abs(left-right) for left,right in zip(human,system)) if subset else None,
                source_runs=';'.join(experiment+':'+selected[experiment] for experiment in EXPERIMENTS)))
    valid = {(row['method'],row['group']) for row in metrics}
    combined = [row for row in combined if (row['method'],row['group']) in valid]
    return combined,metrics,''


def render_combined(points, metrics, message=''):
    if message:
        return '<section id="combined-comparison"><h2>AI＋無人島の合算比較</h2><p>'+html.escape(message)+'</p></section>'
    if not points:
        return ''
    parts = ['<section id="combined-comparison"><h2>AI＋無人島を合わせた3手法の比較</h2>',
        '<p>各会話の設定済み採用runを1つずつ使い、対応する点をまとめました。文字版・動画版・実験参加者版は別々に比較します。1点は会話・ペア・評定番号ごとの平均評定と推定値です。</p>',
        '<p>各点を同じ重みで計算するため、時点数の多い無人島の影響が大きくなります。相関係数やMAEを会話間で単純平均したものではありません。会話間・ペア間の評価水準の違いも相関に含まれます。変化量は同じ会話・同じペアの連続する番号内だけで計算します。</p>',
        '<p><a href="combined_scatter_points.csv">合算の対応点CSV</a> ／ <a href="combined_metrics.csv">合算の相関・MAE CSV</a></p>']
    sources = sorted({(row['source_experiment'],row['source_run']) for row in points})
    parts.append('<p>採用run：'+' ／ '.join(html.escape(experiment+'：'+run) for experiment,run in sources)+'</p>')
    parts.append(render_scatter(points).replace('id="scatter-correlations"','id="combined-scatter-correlations"'))
    graphs = render_pooled_graphs(metrics).replace('id="pooled-correlations"','id="combined-pooled-correlations"')
    graphs = graphs.replace('会話は別々に表示します。','ここでは2会話を合算して表示します。')
    parts.append('<h3>合算した相関係数・MAEの比較グラフ</h3>'+graphs)
    names = {'text':'文字版','video':'動画版','experiment':'実験参加者版'}
    parts.append('<h3>合算した数値の一覧</h3><div style="overflow-x:auto"><table><tr><th>対象</th><th>手法</th><th>回答グループ</th><th>点数</th><th>ピアソン r</th><th>スピアマン ρ</th><th>MAE</th></tr>')
    for measure in ['level','change']:
        for method in COMPARISON_METHODS:
            for row in metrics:
                if (row['measure'],row['method'])!=(measure,method):
                    continue
                cells = ['評定値' if measure=='level' else '変化量',method,names[row['group']],str(row['n'])]
                cells += ['—' if row[key] is None else f'{row[key]:.3f}' for key in ['pearson','spearman','mae']]
                parts.append('<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in cells)+'</tr>')
    parts.append('</table></div></section>')
    return ''.join(parts)
