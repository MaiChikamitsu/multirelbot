"""Descriptive change metrics alongside existing level-based comparisons."""
import html
import statistics as st
from collections import defaultdict
from comparison_methods import COMPARISON_METHODS

EPSILON = 1e-9


def direction(value):
    return 1 if value > EPSILON else -1 if value < -EPSILON else 0


def calculate_trends(rows, correlation):
    fields = ['experiment', 'run', 'method', 'pair', 'target', 'respondent']
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in fields)].append(row)
    summaries, changes = [], []
    for key, series in sorted(grouped.items()):
        series.sort(key=lambda r: r['index'])
        base = dict(zip(fields, key))
        points = []
        for previous, current in zip(series, series[1:]):
            # Missing indices break a sequence; never bridge a missing rating.
            if current['index'] != previous['index'] + 1:
                continue
            sd = current['prediction'] - previous['prediction']
            hd = current['human'] - previous['human']
            point = dict(base, from_index=previous['index'], index=current['index'],
                         system_change=sd, human_change=hd,
                         change_absolute_error=abs(sd-hd),
                         system_direction=direction(sd), human_direction=direction(hd),
                         direction_match=int(direction(sd)==direction(hd)))
            points.append(point)
        changes.extend(points)
        human_changed = [p for p in points if p['human_direction'] != 0]
        summaries.append(dict(base, n_points=len(series), n_changes=len(points),
            level_mae=st.mean(abs(r['prediction']-r['human']) for r in series),
            level_pearson=correlation([r['prediction'] for r in series], [r['human'] for r in series]),
            change_mae=st.mean(p['change_absolute_error'] for p in points) if points else None,
            change_pearson=correlation([p['system_change'] for p in points], [p['human_change'] for p in points]),
            direction_accuracy=st.mean(p['direction_match'] for p in points) if points else None,
            human_changed_n=len(human_changed),
            direction_accuracy_when_human_changed=st.mean(p['direction_match'] for p in human_changed) if human_changed else None))
    return summaries, changes


def chart(series, change=False):
    people = sorted(k for k in series if k != 'system')
    colors = {person: '#dc2626' if person.startswith('動画版 ') else '#059669' if person.startswith('文字版 ') else '#2563eb'
              for person in people}
    colors['system'] = '#111827'
    keys = people + (['system'] if 'system' in series else [])
    indices = [i for points in series.values() for i, _ in points]
    if not indices: return '<p>比較できる連続区間がありません。</p>'
    maximum = max(indices)
    limit = 2 if change else 1
    x = lambda i: 55 + (i-1)*1000/max(1,maximum-1)
    y = lambda v: 155 - v*120/limit
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 325" role="img" aria-label="'+('前時点からの変化量' if change else '回答者別の評定推移')+'"><rect width="1100" height="325" fill="white"/>']
    for v in [-limit, 0, limit]:
        parts.append(f'<path d="M55 {y(v)} H1055" stroke="#dce2ea"/><text x="5" y="{y(v)+5}" font-size="14">{v}</text>')
    for i in range(1,maximum+1):
        if i==1 or i%5==0 or i==maximum: parts.append(f'<text x="{x(i)-5}" y="305" font-size="13">{i}</text>')
    for k in keys:
        color = colors.get(k, '#9333ea')
        points = sorted(series[k])
        # Split lines at gaps to avoid suggesting interpolation across missing data.
        segments = []
        for i, v in points:
            if not segments or i != segments[-1][-1][0]+1: segments.append([])
            segments[-1].append((i,v))
        for segment in segments:
            coords = ' '.join(f'{x(i):.2f},{y(v):.2f}' for i,v in segment)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{3 if k=="system" else 1.8}" points="{coords}"/>')
            for i,v in segment: parts.append(f'<circle cx="{x(i):.2f}" cy="{y(v):.2f}" r="2" fill="{color}"/>')
    legend = [('system', '#111827', '黒：システム'), ('experiment', '#2563eb', '青：実験参加者'),
              ('video', '#dc2626', '赤：動画版'), ('text', '#059669', '緑：文字版')]
    visible = {series_group(person) for person in keys}
    parts.append('</svg><p style="display:flex;flex-wrap:wrap;gap:8px 20px">'+''.join(
        '<span style="color:'+color+'">'+label+'</span>' for group, color, label in legend if group in visible)+'</p>')
    parts.append('<p style="display:flex;flex-wrap:wrap;gap:8px 20px">' + ''.join('<span style="color:'+colors[k]+'">'+html.escape(k if '回答者ID ' in k else '実験 回答者ID '+k)+'</span>' for k in people) + '</p>')
    return ''.join(parts)


def series_group(label):
    if label == 'system':
        return 'system'
    if label.startswith('動画版 '):
        return 'video'
    if label.startswith('文字版 '):
        return 'text'
    return 'experiment'


def correlation_rows(series, correlation):
    from analyze_relations import ranks
    system = dict(series.get('system', []))
    people = {label: dict(points) for label, points in series.items() if label != 'system'}
    by_index = defaultdict(list)
    for points in people.values():
        for index, value in points.items():
            by_index[index].append(value)
    average = {index: st.mean(values) for index, values in by_index.items()}
    result = []
    for label, points in [('このグループの平均', average)] + sorted(people.items()):
        indices = sorted(system.keys() & points.keys())
        predicted = [system[index] for index in indices]
        human = [points[index] for index in indices]
        result.append(dict(label=label, n=len(indices),
                           pearson=correlation(predicted, human),
                           spearman=correlation(ranks(predicted), ranks(human)),
                           mae=st.mean(abs(left-right) for left,right in zip(predicted,human)) if indices else None))
    return result


def correlation_table(series, correlation):
    parts = ['<h5>システムとの相関</h5><table><tr><th>比較対象</th><th>比較点数</th><th>Pearson r</th><th>Spearman ρ</th></tr>']
    for row in correlation_rows(series, correlation):
        label = row['label'] if '回答者ID ' in row['label'] or row['label'] == 'このグループの平均' else '実験参加者 回答者ID '+row['label']
        parts.append('<tr><td>'+html.escape(label)+'</td><td>'+str(row['n'])+'</td>')
        for metric in ['pearson', 'spearman']:
            value = row[metric]
            parts.append('<td>'+('—（算出不可）' if value is None else f'{value:+.3f}')+'</td>')
        parts.append('</tr>')
    parts.append('</table><p><small>＋1に近いほど同じ向き、−1に近いほど逆向き、0付近はその指標での関連が弱いことを表します。Pearsonは値の動き、Spearmanは順位の一致を見ます。相関が高くても評定値自体が一致するとは限りません。定数系列や点数不足は算出不可です。統計的な有意差の判定ではありません。平均は各時点の回答済みの人から計算します。</small></p>')
    return ''.join(parts)


def comparison_charts(levels, deltas, split=False, correlation=None):
    parts = []
    groups = [('text', '文字版とシステム'), ('video', '動画版とシステム'),
              ('experiment', '実験参加者版とシステム')] if split else [(None, '')]
    for group, title in groups:
        if group is not None:
            parts.append('<section class="group-comparison" data-group="'+group+'"><h3>'+title+'</h3>')
        for series, change, heading in [(levels, False, '評定値の推移'), (deltas, True, '前時点からの変化量')]:
            selected = {label: list(points.items()) for label, points in series.items()
                        if group is None or series_group(label) in [group, 'system']}
            parts.append('<h4>'+heading+'</h4>'+chart(selected, change))
            if split and correlation is not None:
                parts.append(correlation_table(selected, correlation))
        if group is not None:
            parts.append('</section>')
    return ''.join(parts)


def render_trends(rows, correlation, observers=()):
    summaries, changes = calculate_trends(rows, correlation)
    esc = html.escape
    parts = ['<section id="individual-trends"><h2>一人ひとりの評価の流れ</h2>',
        '<p>手法・ペアごとに開いて比較できます。上の図は評定値（−1〜＋1）、下の図は前時点からの変化量（−2〜＋2）です。横軸は評定番号。変化量の番号tはt−1→tの変化を表します。</p>',
        '<p>方向は上昇・変化なし・低下の3分類。一致率は0〜100％です。計算誤差を除くため±1e−9以内のみ変化なしとします。欠測をまたぐ差分は計算しません。相関が未定義の場合や比較区間がない場合は「—」です。</p>',
        '<p>変化量MAEは小さいほど、方向一致率・相関は大きいほど一致します。人が変化した区間だけの一致率も併記します。これらは記述指標であり、時系列の独立性を仮定した有意差検定は行っていません。</p>']
    groups = sorted({(r['method'],r['pair']) for r in rows})
    if observers:
        parts.append('<p>すべて実線で、4グループの色を統一しています。青：実験、赤：動画版、緑：文字版、黒：システム。同じグループの各回答者は同じ色で表示します。</p>')
        parts.append('<p>LLM-only・LLM + SMA・MAVeRDは、文字版・動画版・実験参加者版をそれぞれシステムと比較する3つのグラフに分けています。</p>')
    for method,pair in groups:
        subset = [r for r in rows if r['method']==method and r['pair']==pair and r['target']=='individual']
        if not subset: continue
        parts.append('<details style="border:1px solid #dce2ea;padding:16px;margin:16px 0"><summary style="cursor:pointer;font-weight:bold">'+esc(method+' ／ '+pair)+'</summary>')
        levels = defaultdict(dict)
        for r in subset:
            levels['system'][r['index']] = r['prediction']
            levels[r['respondent']][r['index']] = r['human']
        for rating in observers:
            if rating['pair'] == pair and rating['experiment'] in {r['experiment'] for r in subset}:
                label = rating['label']+' 回答者ID '+rating['respondent']
                levels[label][rating['index']] = rating['human']
        deltas = defaultdict(dict)
        for r in changes:
            if r['method']==method and r['pair']==pair and r['target']=='individual':
                deltas['system'][r['index']] = r['system_change']
                deltas[r['respondent']][r['index']] = r['human_change']
        for label, values in levels.items():
            if label == 'system' or '回答者ID ' not in label:
                continue
            points = sorted(values.items())
            for previous, current in zip(points, points[1:]):
                if current[0] == previous[0] + 1:
                    deltas[label][current[0]] = current[1] - previous[1]
        parts.append(comparison_charts(levels, deltas, bool(observers) and method in COMPARISON_METHODS, correlation))
        parts.append('<div style="overflow-x:auto"><table><tr><th>比較対象</th><th>時点数</th><th>値のMAE</th><th>値の相関</th><th>変化区間数</th><th>変化量MAE</th><th>変化量の相関</th><th>方向一致率</th><th>人が変化した区間数</th><th>その区間の方向一致率</th></tr>')
        for r in summaries:
            if r['method']!=method or r['pair']!=pair: continue
            label = '3人平均' if r['target']=='mean' else '回答者'+r['respondent']
            cells = [esc(label)]
            for field in ['n_points','level_mae','level_pearson','n_changes','change_mae','change_pearson','direction_accuracy','human_changed_n','direction_accuracy_when_human_changed']:
                v=r[field]
                cells.append('—' if v is None else f'{v*100:.1f}％' if 'accuracy' in field else str(v) if isinstance(v,int) else f'{v:.4f}')
            parts.append('<tr>'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
        parts.append('</table></div></details>')
    parts.append('</section>')
    return ''.join(parts)
