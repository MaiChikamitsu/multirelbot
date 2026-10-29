"""Compact correlation overview for the selected estimation methods."""
import html
import math
from collections import defaultdict

from relation_trends import correlation_rows, series_group
from comparison_methods import COMPARISON_METHODS, METHOD_COLORS


def build_correlations(rows, observers, correlation):
    result = []
    keys = sorted({(row['experiment'], row['run'], row['method'], row['pair'])
                   for row in rows if row['method'] in COMPARISON_METHODS})
    for experiment, run, method, pair in keys:
        levels = defaultdict(dict)
        for row in rows:
            if (row['experiment'], row['run'], row['method'], row['pair']) != (experiment, run, method, pair):
                continue
            levels['system'][row['index']] = row['prediction']
            if row['target'] == 'individual':
                levels[row['respondent']][row['index']] = row['human']
        for row in observers:
            if (row['experiment'], row['pair']) == (experiment, pair):
                levels[row['label']+' 回答者ID '+row['respondent']][row['index']] = row['human']
        deltas = defaultdict(dict)
        for label, points in levels.items():
            ordered = sorted(points.items())
            for previous, current in zip(ordered, ordered[1:]):
                if current[0] == previous[0] + 1:
                    deltas[label][current[0]] = current[1] - previous[1]
        for group in ['text', 'video', 'experiment']:
            if not any(series_group(label) == group for label in levels):
                continue
            for measure, series in [('level', levels), ('change', deltas)]:
                selected = {label: list(points.items()) for label, points in series.items()
                            if series_group(label) in [group, 'system']}
                for record in correlation_rows(selected, correlation):
                    result.append(dict(experiment=experiment, run=run, method=method, pair=pair,
                        group=group, measure=measure, target='mean' if record['label'] == 'このグループの平均' else 'individual',
                        **record))
    return result


def build_pooled_correlations(rows, observers, correlation):
    result = []
    for experiment,run,method in sorted({(row['experiment'],row['run'],row['method']) for row in rows if row['method'] in COMPARISON_METHODS}):
        levels = defaultdict(dict)
        for row in rows:
            if (row['experiment'],row['run'],row['method']) != (experiment,run,method):
                continue
            key = (row['pair'],row['index'])
            levels['system'][key] = row['prediction']
            if row['target'] == 'individual':
                levels[row['respondent']][key] = row['human']
        for row in observers:
            if row['experiment'] == experiment:
                levels[row['label']+' 回答者ID '+row['respondent']][row['pair'],row['index']] = row['human']
        deltas = defaultdict(dict)
        for label,points in levels.items():
            ordered = sorted(points.items())
            for previous,current in zip(ordered,ordered[1:]):
                previous_pair,previous_index = previous[0]
                current_pair,current_index = current[0]
                if current_pair == previous_pair and current_index == previous_index+1:
                    deltas[label][current[0]] = current[1]-previous[1]
        for group in ['text','video','experiment']:
            if not any(series_group(label)==group for label in levels):
                continue
            for measure,series in [('level',levels),('change',deltas)]:
                selected = {label:list(points.items()) for label,points in series.items() if series_group(label) in [group,'system']}
                record = correlation_rows(selected,correlation)[0]
                result.append(dict(experiment=experiment,run=run,method=method,pair='all',
                    group=group,measure=measure,target='mean',**record))
    return result


def mae_axis_limit(records, measure):
    values = [row['mae'] for row in records if row['measure']==measure and row.get('mae') is not None]
    return max(.1,math.ceil(max(values,default=0)*10)/10)


def comparison_bars(records, metric, pooled=False, axis_limit=None):
    esc = html.escape
    names = {'text': '文字版', 'video': '動画版', 'experiment': '実験参加者版'}
    methods = [(method, METHOD_COLORS[method]) for method in COMPARISON_METHODS]
    keys = [(group, pair) for group in names for pair in (['all'] if pooled else ['A-B', 'B-C', 'A-C'])
            if any(row['group'] == group and row['pair'] == pair for row in records)]
    row_height = len(methods)*26+24
    height = 100 + len(keys)*row_height
    is_mae = metric == 'mae'
    limit = axis_limit or max(.1,math.ceil(max((row.get('mae') or 0 for row in records),default=0)*10)/10)
    zero = 305 if is_mae else 575
    scale = 540/limit if is_mae else 270
    title = 'LLM-only・LLM + SMA・MAVeRDの'+('MAE' if is_mae else '相関係数')+'の比較'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1020 {height}" style="width:100%;height:auto;min-width:700px" role="img" aria-label="{title}">',
             f'<rect width="1020" height="{height}" fill="white"/>']
    for value in ([0,limit/2,limit] if is_mae else [-1, -.5, 0, .5, 1]):
        position = zero+value*scale
        parts.append(f'<path d="M{position} 45 V{height-30}" stroke="{"#64748b" if value==0 else "#e2e8f0"}"/><text x="{position}" y="28" text-anchor="middle" font-size="14">{value:+g}</text>')
    for position, (group, pair) in enumerate(keys):
        top = 58 + position*row_height
        parts.append(f'<text x="12" y="{top+24}" font-size="16">{esc(names[group] if pooled else names[group]+" ／ "+pair)}</text>')
        for offset, (method, color) in enumerate(methods):
            row = next((row for row in records if (row['group'], row['pair'], row['method']) == (group, pair, method)), None)
            value = row.get(metric) if row else None
            vertical = top+offset*26
            parts.append(f'<text x="210" y="{vertical+15}" font-size="13" fill="{color}">{method}</text>')
            if value is None:
                parts.append(f'<text x="{zero+8}" y="{vertical+15}" font-size="13" fill="#64748b">算出不可</text>')
                continue
            endpoint = zero+value*scale
            left = min(zero, endpoint)
            width = abs(value)*scale
            parts.append(f'<rect class="{"mae-bar" if is_mae else "correlation-bar"}" data-method="{method}" data-value="{value}" x="{left:.2f}" y="{vertical}" width="{width:.2f}" height="20" fill="{color}"><title>{esc(names[group]+" ／ "+pair+" ／ "+method)}：{value:.3f}（比較点数 {row["n"]}）</title></rect>')
            label_position = endpoint+7 if value >= 0 else endpoint-7
            anchor = 'start' if value >= 0 else 'end'
            label = f'{value:.3f}' if is_mae else f'{value:+.3f}'
            parts.append(f'<text x="{label_position:.2f}" y="{vertical+15}" text-anchor="{anchor}" font-size="14">{label}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def render_comparison_graphs(records):
    means = [row for row in records if row['target'] == 'mean']
    parts = ['<section id="correlation-graphs"><h2>LLM-only・LLM + SMA・MAVeRDの相関比較</h2>',
             '<p><span style="color:#2563eb">青：LLM-only</span>　<span style="color:#9333ea">紫：LLM + SMA</span>　<span style="color:#ea580c">オレンジ：MAVeRD</span>。各版の平均評定と各手法との相関を、同じ尺度（−1〜＋1）で比較します。右が正の相関、左が負の相関です。算出不可は棒を描きません。</p>']
    for experiment, run in sorted({(row['experiment'], row['run']) for row in means}):
        parts.append('<h3>'+html.escape(experiment)+' ／ '+html.escape(run)+'</h3>')
        for measure, title in [('level', '評定値'), ('change', '前時点からの変化量')]:
            for metric, metric_title in [('pearson', 'ピアソンの積率相関係数（r）'), ('spearman', 'スピアマンの順位相関係数（ρ）'), ('mae','平均絶対誤差（MAE・小さいほど一致）')]:
                selected = [row for row in means if (row['experiment'], row['run'], row['measure']) == (experiment, run, measure)]
                if selected:
                    parts.append('<h4>'+title+'：'+metric_title+'</h4><div style="overflow-x:auto">'+comparison_bars(selected, metric, axis_limit=mae_axis_limit(means,measure) if metric=='mae' else None)+'</div>')
    parts.append('</section>')
    return ''.join(parts)


def render_pooled_graphs(records):
    if not records:
        return ''
    parts = ['<section id="pooled-correlations"><h2>システムと3グループの相関（全ペアまとめ）</h2>',
        '<p><span style="color:#2563eb">青：LLM-only</span>　<span style="color:#9333ea">紫：LLM + SMA</span>　<span style="color:#ea580c">オレンジ：MAVeRD</span>。文字版・動画版・実験参加者版それぞれの平均回答と、各システムの相関を比較します。</p>',
        '<p>A-B・B-C・C-Aの対応する評定点をまとめて相関を計算しています。ペア別の相関係数の平均ではありません。ペア間の評価水準の差も反映されます。会話は別々に表示します。変化量は同じペア内の連続する番号だけから計算し、ペアをまたぐ差分は作りません。</p>']
    for experiment,run in sorted({(row['experiment'],row['run']) for row in records}):
        title = {'ai_nashi':'AI・縛りなし','muzintou_ari':'無人島・縛りあり','combined_ai_muzintou':'AI＋無人島（合算）'}.get(experiment,experiment)
        parts.append('<h3>'+html.escape(title)+' ／ '+html.escape(run)+'</h3>')
        for measure,measure_title in [('level','評定値'),('change','前時点からの変化量')]:
            for metric,metric_title in [('pearson','ピアソンの積率相関係数（r）'),('spearman','スピアマンの順位相関係数（ρ）'),('mae','平均絶対誤差（MAE・小さいほど一致）')]:
                selected = [row for row in records if (row['experiment'],row['run'],row['measure'])==(experiment,run,measure)]
                parts.append('<h4>'+measure_title+'：'+metric_title+'</h4><div style="overflow-x:auto">'+comparison_bars(selected,metric,pooled=True,axis_limit=mae_axis_limit(records,measure) if metric=='mae' else None)+'</div>')
    parts.append('</section>')
    return ''.join(parts)


def render_summary(records, human_report_path='human_group_report.html', pooled_records=(), scatter_points=()):
    esc = html.escape
    names = {'text': '文字版', 'video': '動画版', 'experiment': '実験参加者版'}
    pair_graphs = render_comparison_graphs(records)
    if pooled_records:
        pair_graphs = '<details><summary>ペア別の相関グラフも見る</summary>'+pair_graphs+'</details>'
    from correlation_scatter import render_scatter
    pooled_graphs = render_pooled_graphs(pooled_records)
    if scatter_points:
        pooled_graphs = '<details><summary>相関係数とMAEの比較グラフを見る</summary><p>相関は＋1に近いほど同じ傾向、MAEは0に近いほど点数が一致します。MAEは各版の平均回答とシステムの絶対差の平均で、個人別MAEの平均ではありません。</p>'+pooled_graphs+'</details>'
    parts = [render_scatter(scatter_points), pooled_graphs, pair_graphs, '<section id="correlation-overview"><h2>相関係数の一覧（LLM-only・LLM + SMA・MAVeRD）</h2>',
        '<p><a href="'+esc(human_report_path,quote=True)+'">文字版・動画版・実験参加者版同士の違い・相関・ばらつきはこちら</a></p>',
        '<p>各版の平均評定とシステムの相関を並べています。個人の相関係数の平均ではありません。＋1に近いほど同じ傾向、−1に近いほど逆の傾向です。—はデータ不足・定数系列などで算出できない場合です。有意差検定は行っていません。</p>']
    for measure, title in [('level', '評定値'), ('change', '前時点からの変化量')]:
        parts.append('<h3>'+title+'</h3>')
        for metric, metric_title in [('pearson', 'ピアソンの積率相関係数（r）'), ('spearman', 'スピアマンの順位相関係数（ρ）'), ('mae','平均絶対誤差（MAE・小さいほど一致）')]:
            parts.append('<h4>'+metric_title+'</h4><div style="overflow-x:auto"><table><tr><th>実験</th><th>実行</th><th>手法</th><th>ペア</th><th>文字版</th><th>動画版</th><th>実験参加者版</th></tr>')
            selected = [row for row in records if row['target'] == 'mean' and row['measure'] == measure]
            keys = sorted({(row['experiment'], row['run'], row['method'], row['pair']) for row in selected})
            for key in keys:
                values = {row['group']: row for row in selected if (row['experiment'], row['run'], row['method'], row['pair']) == key}
                parts.append('<tr>'+''.join('<td>'+esc(value)+'</td>' for value in key))
                for group in names:
                    row = values.get(group)
                    value = row.get(metric) if row else None
                    count = str(row['n']) if row else '0'
                    parts.append('<td title="比較点数：'+count+'">'+('—' if value is None else f'{value:.3f}' if metric=='mae' else f'{value:+.3f}')+'</td>')
                parts.append('</tr>')
            parts.append('</table></div>')
    parts.append('<details><summary>一人ひとりの相関係数・MAEを一覧で見る</summary><div style="overflow-x:auto"><table><tr><th>実験</th><th>実行</th><th>手法</th><th>ペア</th><th>版</th><th>回答者</th><th>対象</th><th>比較点数</th><th>ピアソン r</th><th>スピアマン ρ</th><th>MAE</th></tr>')
    for row in records:
        if row['target'] != 'individual':
            continue
        cells = [row['experiment'], row['run'], row['method'], row['pair'], names[row['group']], row['label'],
                 '評定値' if row['measure'] == 'level' else '変化量', str(row['n'])]
        cells += ['—' if row[metric] is None else f'{row[metric]:+.3f}' for metric in ['pearson', 'spearman']]
        cells.append('—' if row.get('mae') is None else f'{row["mae"]:.3f}')
        parts.append('<tr>'+''.join('<td>'+esc(value)+'</td>' for value in cells)+'</tr>')
    parts.append('</table></div></details></section>')
    return ''.join(parts)


def render_page(out, records, pooled_records=(), scatter_points=(), combined_html=''):
    page = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>相関係数の一覧</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 16px;color:#182335}table{border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:8px;border:1px solid #dce2ea}th{background:#eaf0f8}h3{margin-top:36px}details{margin-top:30px}</style><p><a href="report.html">分析ファイル一覧へ</a> ／ <a href="correlation_summary.csv">相関一覧CSV</a></p>'
    (out/'correlation_report.html').write_text(page+combined_html+render_summary(records,pooled_records=pooled_records,scatter_points=scatter_points)+'</html>', encoding='utf-8')
