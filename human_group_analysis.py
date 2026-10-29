"""Descriptive comparisons of human rating groups, independent of system runs."""
import html
import math
import statistics as st
from collections import defaultdict
from itertools import combinations

GROUPS = {'text': ('文字版', '#059669'), 'video': ('動画版', '#dc2626'),
          'experiment': ('実験参加者版', '#2563eb')}


def analyze_groups(humans, observers, correlation, ranks):
    series = defaultdict(dict)
    for source, records in [('experiment', humans), (None, observers)]:
        for row in records:
            group = source or {'文字版': 'text', '動画版': 'video'}.get(row['label'])
            if group is None:
                continue
            key = (row['experiment'], row['pair'], group, row['respondent'])
            index = int(row['index'])
            if index in series[key]:
                raise ValueError(f'Duplicate human rating: {key} / {index}')
            series[key][index] = float(row['human'])
    by_point = defaultdict(dict)
    individuals = []
    for (experiment, pair, group, respondent), values in sorted(series.items()):
        scores = list(values.values())
        individuals.append(dict(experiment=experiment, pair=pair, group=group, respondent=respondent,
            n=len(scores), mean=st.mean(scores), temporal_sd=st.stdev(scores) if len(scores)>1 else None))
        for index, value in values.items():
            by_point[experiment, pair, group, index][respondent] = value
    points = []
    for (experiment, pair, group, index), values in sorted(by_point.items()):
        scores = list(values.values())
        points.append(dict(experiment=experiment, pair=pair, group=group, index=index,
            n=len(scores), mean=st.mean(scores), sd=st.stdev(scores) if len(scores)>1 else None,
            minimum=min(scores), maximum=max(scores), range=max(scores)-min(scores)))
    stats = []
    for experiment, pair, group in sorted({key[:3] for key in series}):
        people = {key[3]: values for key, values in series.items() if key[:3] == (experiment, pair, group)}
        scores = [score for values in people.values() for score in values.values()]
        subset = [point for point in points if (point['experiment'], point['pair'], point['group']) == (experiment, pair, group)]
        sds = [point['sd'] for point in subset if point['sd'] is not None]
        means = [point['mean'] for point in subset]
        stats.append(dict(experiment=experiment, pair=pair, group=group, n_people=len(people),
            respondent_ids=','.join(sorted(people)), n_indices=len(subset), n_ratings=len(scores),
            mean=st.mean(scores), median=st.median(scores), minimum=min(scores), maximum=max(scores),
            pooled_sd=st.stdev(scores) if len(scores)>1 else None,
            mean_within_sd=st.mean(sds) if sds else None, n_sd_indices=len(sds),
            mean_within_range=st.mean(point['range'] for point in subset),
            temporal_mean_sd=st.stdev(means) if len(means)>1 else None))
    comparisons, rater_correlations = [], []
    for experiment, pair in sorted({key[:2] for key in series}):
        means = {group: {point['index']: point['mean'] for point in points
                        if (point['experiment'], point['pair'], point['group']) == (experiment, pair, group)}
                 for group in GROUPS}
        for first, second in combinations(GROUPS, 2):
            common = sorted(means[first].keys() & means[second].keys())
            if not common:
                continue
            left = [means[first][index] for index in common]
            right = [means[second][index] for index in common]
            differences = [a-b for a,b in zip(left, right)]
            intervals = [(previous, current) for previous,current in zip(common, common[1:]) if current == previous+1]
            left_change = [means[first][current]-means[first][previous] for previous,current in intervals]
            right_change = [means[second][current]-means[second][previous] for previous,current in intervals]
            comparisons.append(dict(experiment=experiment, pair=pair, first=first, second=second, n=len(common),
                first_mean=st.mean(left), second_mean=st.mean(right), bias=st.mean(differences),
                mae=st.mean(abs(value) for value in differences), rmse=math.sqrt(st.mean(value**2 for value in differences)),
                pearson=correlation(left, right), spearman=correlation(ranks(left), ranks(right)),
                n_changes=len(intervals), change_pearson=correlation(left_change, right_change),
                change_spearman=correlation(ranks(left_change), ranks(right_change))))
        people = [(key[2], key[3], values) for key, values in series.items() if key[:2] == (experiment, pair)]
        for (first_group, first_person, left), (second_group, second_person, right) in combinations(sorted(people), 2):
            common = sorted(left.keys() & right.keys())
            xs, ys = [left[index] for index in common], [right[index] for index in common]
            rater_correlations.append(dict(experiment=experiment, pair=pair, first_group=first_group,
                first_person=first_person, second_group=second_group, second_person=second_person,
                n=len(common), pearson=correlation(xs, ys), spearman=correlation(ranks(xs), ranks(ys))))
    return dict(points=points, statistics=stats, comparisons=comparisons,
                individuals=individuals, rater_correlations=rater_correlations)


def formatted(value):
    return '—' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)


def table(headers, rows):
    return '<div class="table-scroll"><table><tr>'+''.join('<th>'+html.escape(header)+'</th>' for header in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(formatted(value))+'</td>' for value in row)+'</tr>' for row in rows)+'</table></div>'


def timeline(points, spread=False):
    maximum = max((point['index'] for point in points), default=1)
    x = lambda index: 65+(index-1)*900/max(1,maximum-1)
    y = (lambda value: 265-value*200/1.5) if spread else (lambda value: 155-value*105)
    title = '同じ時点での回答者間の標準偏差' if spread else '平均評定と回答者の最小〜最大'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1020 320" role="img" aria-label="{title}"><rect width="1020" height="320" fill="white"/>']
    for value in ([0, .5, 1, 1.5] if spread else [-1, -.5, 0, .5, 1]):
        parts.append(f'<path d="M65 {y(value)} H965" stroke="#e2e8f0"/><text x="12" y="{y(value)+5}" font-size="14">{value:g}</text>')
    for index in range(1, maximum+1):
        if index==1 or index==maximum or index%5==0:
            parts.append(f'<text x="{x(index)-6}" y="295" font-size="13">{index}</text>')
    for group, (name, color) in GROUPS.items():
        selected = sorted([point for point in points if point['group']==group and (not spread or point['sd'] is not None)], key=lambda point: point['index'])
        segments = []
        for point in selected:
            if not segments or point['index'] != segments[-1][-1]['index']+1:
                segments.append([])
            segments[-1].append(point)
        for segment in segments:
            if not spread:
                bounds = [(point['index'], point['minimum']) for point in segment]+[(point['index'], point['maximum']) for point in reversed(segment)]
                coords = ' '.join(f'{x(index):.2f},{y(value):.2f}' for index,value in bounds)
                parts.append(f'<polygon points="{coords}" fill="{color}" fill-opacity="0.09"/>')
            coords = ' '.join(f'{x(point["index"]):.2f},{y(point["sd"] if spread else point["mean"]):.2f}' for point in segment)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for point in segment:
                value = point['sd'] if spread else point['mean']
                parts.append(f'<circle cx="{x(point["index"]):.2f}" cy="{y(value):.2f}" r="2.5" fill="{color}"><title>{name}：評定番号 {point["index"]}、値 {value:.3f}、人数 {point["n"]}</title></circle>')
    parts.append('</svg>')
    return ''.join(parts)


def histogram(individual_series):
    counts = {group: [0]*10 for group in GROUPS}
    for group, values in individual_series:
        for value in values:
            counts[group][min(9, max(0, math.floor((value+1)*5+1e-9)))]+=1
    totals = {group: sum(values) for group,values in counts.items()}
    rates = {group: [count/totals[group] if totals[group] else 0 for count in values] for group,values in counts.items()}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1020 330" role="img" aria-label="個人と時点をまとめた評定値の分布"><rect width="1020" height="330" fill="white"/>']
    for rate in [0, .25, .5, .75, 1]:
        vertical = 260-rate*220
        parts.append(f'<path d="M65 {vertical} H965" stroke="#e2e8f0"/><text x="10" y="{vertical+5}" font-size="13">{rate*100:g}%</text>')
    for bucket in range(10):
        for position,(group,(name,color)) in enumerate(GROUPS.items()):
            height = rates[group][bucket]*220
            parts.append(f'<rect x="{72+bucket*89+position*23}" y="{260-height:.2f}" width="20" height="{height:.2f}" fill="{color}"><title>{name}：{counts[group][bucket]}件 / {totals[group]}件（{rates[group][bucket]*100:.1f}%）</title></rect>')
        parts.append(f'<text x="{72+bucket*89}" y="286" font-size="11">{-1+bucket*.2:.1f}〜</text>')
    parts.append('<text x="65" y="315" font-size="13">横軸：評定値（幅0.2、最終区間のみ＋1を含む）　縦軸：各グループ内の回答割合</text></svg>')
    return ''.join(parts)


def overview_plot(rows, minimum, maximum, ticks, label, colors, dots=False):
    height = 65+len(rows)*42
    start, end = 225, 760
    x = lambda value: start+(value-minimum)*(end-start)/(maximum-minimum)
    zero = x(0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 {height}" role="img" aria-label="{html.escape(label)}"><rect width="860" height="{height}" fill="white"/>']
    for tick in ticks:
        position = x(tick)
        parts.append(f'<path d="M{position:.2f} 35 V{height-15}" stroke="{"#94a3b8" if tick==0 else "#e2e8f0"}"/><text x="{position:.2f}" y="22" text-anchor="middle" font-size="14">{tick:g}</text>')
    for index,row in enumerate(rows):
        vertical = 51+index*42
        parts.append(f'<text x="8" y="{vertical+6}" font-size="16">{html.escape(row["label"])}</text>')
        value = row['value']
        if value is None:
            parts.append(f'<text x="{zero+8:.2f}" y="{vertical+6}" font-size="14">算出不可</text>')
            continue
        color = colors[row['group']]
        position = x(value)
        if dots:
            parts.append(f'<circle class="overview-value" data-value="{value}" cx="{position:.2f}" cy="{vertical}" r="7" fill="{color}"/>')
        else:
            parts.append(f'<rect class="overview-value" data-value="{value}" x="{min(zero,position):.2f}" y="{vertical-9}" width="{abs(position-zero):.2f}" height="18" fill="{color}"/>')
        parts.append(f'<text x="{position+12:.2f}" y="{vertical+6}" font-size="15">{value:.3f}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def render_visual_overview(data):
    colors = {group:color for group,(_,color) in GROUPS.items()}
    comparisons = [('text','video','文字版 × 動画版'),('text','experiment','文字版 × 参加者'),('video','experiment','動画版 × 参加者')]
    comparison_colors = {label:'#7c3aed' for _,_,label in comparisons}
    sd_limit = max(.5,math.ceil(max((row['mean_within_sd'] or 0 for row in data['statistics']),default=0)*4)/4)
    mae_limit = max(.5,math.ceil(max((row['mae'] for row in data['comparisons']),default=0)*4)/4)
    parts = ['<section id="human-visual-overview"><h2>まずは、この4つの図で比較</h2>',
             '<p><span style="color:#059669">緑＝文字版</span>　<span style="color:#dc2626">赤＝動画版</span>　<span style="color:#2563eb">青＝実験参加者版</span>。すべて各会話のA-B・B-C・A-Cを分けて表示しています。</p>']
    for experiment in sorted({row['experiment'] for row in data['statistics']}):
        name = {'ai_nashi':'AI・縛りなし','muzintou_ari':'無人島・縛りあり'}.get(experiment,experiment)
        parts.append('<h3>'+html.escape(name)+'</h3><div class="visual-grid">')
        means,spreads,correlations,differences = [],[],[],[]
        for pair in ['A-B','B-C','A-C']:
            for group,(group_name,_) in GROUPS.items():
                row = next((row for row in data['statistics'] if (row['experiment'],row['pair'],row['group'])==(experiment,pair,group)),None)
                if row:
                    label = pair+'　'+group_name
                    means.append(dict(label=label,group=group,value=row['mean']))
                    spreads.append(dict(label=label,group=group,value=row['mean_within_sd']))
            for first,second,label in comparisons:
                row = next((row for row in data['comparisons'] if (row['experiment'],row['pair'],row['first'],row['second'])==(experiment,pair,first,second)),None)
                if row:
                    correlations.append(dict(label=pair+'　'+label,group=label,value=row['pearson']))
                    differences.append(dict(label=pair+'　'+label,group=label,value=row['mae']))
        for title,description,plot in [
            ('① どのグループが高めに評価？','点が右ほど高い評価、左ほど低い評価。各グループの全回答の平均です。',overview_plot(means,-1,1,[-1,-.5,0,.5,1],'平均評価の比較',colors,dots=True)),
            ('② 同じグループ内で意見が割れている？','棒が長いほど回答者の点数がばらばら。短いほどそろっています（同じ時点の標準偏差の平均）。',overview_plot(spreads,0,sd_limit,[0,sd_limit/2,sd_limit],'回答者間のばらつき',colors)),
            ('③ グループ同士で同じように動く？','右の＋1に近いほど同じ動き。0付近は関連が弱く、左の−1に近いほど逆の動き（平均評定のピアソン相関）。',overview_plot(correlations,-1,1,[-1,-.5,0,.5,1],'グループ平均同士の相関',comparison_colors)),
            ('④ 実際の点数はどれくらい違う？','棒が短いほど点数が近く、長いほどずれています（平均評定同士のMAE）。相関が高くてもずれは残ります。',overview_plot(differences,0,mae_limit,[0,mae_limit/2,mae_limit],'グループ平均同士の点数のずれ',comparison_colors))]:
            parts.append('<article class="visual-card"><h4>'+title+'</h4><p>'+description+'</p>'+plot+'</article>')
        parts.append('</div>')
    parts.append('<p>グループ平均の比較です。個人差や時間ごとの推移は下の詳細にあります。回答者構成も異なるため、差を提示形式だけの影響とは断定できません。</p></section>')
    return ''.join(parts)


def render_report(out, data, humans, observers):
    parts = ['<!doctype html><html lang="ja"><meta charset="utf-8"><title>文字版・動画版・実験参加者版の比較</title>',
        '<style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 18px;color:#182335}table{border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:9px;border:1px solid #dce2ea;white-space:nowrap}th{background:#eef2f7}.table-scroll{overflow-x:auto}svg{width:100%;height:auto}section{margin:45px 0}details{margin:20px 0}p{line-height:1.7}a{color:#2457bb}</style>',
        '<p><a href="report.html">分析ファイル一覧へ</a> ／ <a href="correlation_report.html">システムとの相関比較へ</a></p>',
        '<h1>文字版・動画版・実験参加者版の比較</h1>',
        '<p><a href="correlation_report.html#scatter-correlations">システムと3グループの相関の散布図（全ペアまとめ）を見る →</a></p>',
        '<p><a href="correlation_report.html#combined-comparison">AI＋無人島を合わせた3手法の散布図・相関・MAEを見る →</a></p>',
        '<style>.visual-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.visual-card{border:1px solid #dce2ea;border-radius:12px;padding:18px}.visual-card h4{font-size:19px;margin:0 0 10px}.visual-card p{min-height:70px;margin:0;font-size:15px}@media(max-width:1000px){.visual-grid{grid-template-columns:1fr}}</style>',
        render_visual_overview(data),
        '<p>システム推定値を使わず、人の回答同士を同じ会話・ペア・評定番号で比較します。青：実験参加者版、赤：動画版、緑：文字版。</p>',
        '<p>相関は動きの類似、MAEは点数のずれ、標準偏差は回答者間のばらつきを表します。グループ平均の相関が高くても、個人同士が一致するとは限りません。時点を独立した標本とみなす有意差検定は行っていません。</p>',
        '<p>文字版・動画版・実験参加者版は回答者の構成も異なります。差には提示形式・回答者個人・会話への参加などが含まれ、提示形式だけの効果とは断定できません。欠測は補完しません。各時点の人数はCSVとグラフの点の説明で確認できます。</p>',
        '<p>CSV：'+ ' ／ '.join('<a href="human_group_'+key+'.csv">'+name+'</a>' for key,name in [('statistics','ばらつき集計'),('comparisons','グループ間比較'),('points','時点別平均・標準偏差'),('individuals','個人別集計'),('rater_correlations','個人間相関')])+'</p>']
    comparisons = data['comparisons']
    valid_correlations = [row['pearson'] for row in comparisons if row['pearson'] is not None]
    parts.append('<details><summary>数値の一覧表を開く</summary><h2>結果の一覧</h2>')
    if valid_correlations:
        positive = sum(value>0 for value in valid_correlations)
        parts.append('<p>グループ平均同士のピアソン相関は、算出できた'+str(len(valid_correlations))+'組中'+str(positive)+'組が正の値でした（範囲 '+formatted(min(valid_correlations))+'〜'+formatted(max(valid_correlations))+'）。正の相関があっても同じ点数を付けているとは限らないため、MAEも併記します。</p>')
    overview = []
    spread_overview = []
    for experiment,pair in sorted({(row['experiment'],row['pair']) for row in data['statistics']}):
        cells = [experiment,pair]
        for first,second in combinations(GROUPS,2):
            row = next((row for row in comparisons if (row['experiment'],row['pair'],row['first'],row['second'])==(experiment,pair,first,second)),None)
            cells.extend([row['pearson'],row['mae']] if row else [None,None])
        overview.append(cells)
        spreads = {row['group']:row['mean_within_sd'] for row in data['statistics'] if (row['experiment'],row['pair'])==(experiment,pair)}
        spread_overview.append([experiment,pair]+[spreads.get(group) for group in GROUPS])
    parts.append('<h3>平均評定同士の相関とずれ</h3>'+table(['実験','ペア','文字×動画 r','文字×動画 MAE','文字×参加者 r','文字×参加者 MAE','動画×参加者 r','動画×参加者 MAE'],overview))
    parts.append('<h3>回答者間のばらつき（時点別SDの平均）</h3>'+table(['実験','ペア','文字版','動画版','実験参加者版'],spread_overview)+'<p>小さいほど同じ時点での個人回答がそろっています。時間的な変化の大きさとは別の指標です。</p></details>')
    for experiment in sorted({row['experiment'] for row in data['statistics']}):
        parts.append('<h2>'+html.escape(experiment)+'</h2>')
        for pair in ['A-B','B-C','A-C']:
            points = [row for row in data['points'] if (row['experiment'],row['pair']) == (experiment,pair)]
            if not points:
                continue
            stats = [row for row in data['statistics'] if (row['experiment'],row['pair']) == (experiment,pair)]
            comparisons = [row for row in data['comparisons'] if (row['experiment'],row['pair']) == (experiment,pair)]
            parts.append('<details><summary>'+pair+'：時系列・分布・個人差の詳細を見る</summary><section><h3>'+pair+'</h3><h4>このペアで見られる違い</h4><ul>')
            for row in comparisons:
                parts.append('<li>'+GROUPS[row['first']][0]+' − '+GROUPS[row['second']][0]+'：平均差 '+f'{row["bias"]:+.3f}'+
                    '、平均のずれ（MAE） '+formatted(row['mae'])+'、平均評定同士のピアソン相関 '+formatted(row['pearson'])+'。</li>')
            valid = [row for row in stats if row['mean_within_sd'] is not None]
            if valid:
                largest = max(valid,key=lambda row:row['mean_within_sd'])
                parts.append('<li>回答者間SDの時点平均が最も大きいのは'+GROUPS[largest['group']][0]+'（'+formatted(largest['mean_within_sd'])+'）。</li>')
            parts.append('</ul><h4>グループ平均の推移と回答の範囲</h4><p>線は各時点の平均、薄い帯は個人回答の最小〜最大です。信頼区間ではありません。</p>'+timeline(points))
            parts.append('<h4>グループ間の相関と点数の差</h4>')
            parts.append(table(['比較（左−右）','共通時点数','平均差','MAE','RMSE','ピアソン r','スピアマン ρ','変化区間数','変化量 r','変化量 ρ'],
                [[GROUPS[row['first']][0]+' − '+GROUPS[row['second']][0]]+[row[key] for key in ['n','bias','mae','rmse','pearson','spearman','n_changes','change_pearson','change_spearman']] for row in comparisons]))
            parts.append('<p>平均差が正なら左のグループが高め、負なら右が高めです。各比較は共通する評定番号だけを使います。変化量は連続する番号の平均評定の差分で、欠測をまたぎません。</p>')
            parts.append('<h4>同じ時点での回答者間のばらつき</h4>'+timeline(points,spread=True))
            parts.append(table(['版','回答者数','回答者ID','時点数','回答数','全回答の平均','中央値','回答者間SDの時点平均','SD算出時点数','最小〜最大幅の時点平均','平均評定の時間方向SD','全回答を混ぜたSD'],
                [[GROUPS[row['group']][0]]+[row[key] for key in ['n_people','respondent_ids','n_indices','n_ratings','mean','median','mean_within_sd','n_sd_indices','mean_within_range','temporal_mean_sd','pooled_sd']] for row in stats]))
            parts.append('<p>回答者間SDは各時点の個人差（標本標準偏差、分母n−1）で、小さいほど点数がそろっています。1人しかいない時点は算出不可です。「平均評定の時間方向SD」は時間的な動き、「全回答を混ぜたSD」は個人差と時間変化の両方を含みます。</p>')
            raw = []
            for group, source in [('experiment',humans),(None,observers)]:
                for row in source:
                    if (row['experiment'],row['pair']) == (experiment,pair):
                        selected_group = group or {'文字版':'text','動画版':'video'}.get(row['label'])
                        if selected_group:
                            raw.append((selected_group,[float(row['human'])]))
            parts.append('<h4>評定値の分布</h4><p>各グループ内の割合で表示します。個人と時点をまとめた分布なので、回答者間のばらつきだけを示す図ではありません。</p>'+histogram(raw))
            raters = [row for row in data['rater_correlations'] if (row['experiment'],row['pair']) == (experiment,pair)]
            parts.append('<details><summary>個人同士にも相関があるかを見る</summary><p>個人ペアごとの時系列相関の分布です。グループ内・グループ間を分けています。これは記述的な相関で、評価の一致度や信頼性の検定ではありません。</p>')
            summaries = []
            for first,second in [(group,group) for group in GROUPS]+list(combinations(GROUPS,2)):
                selected = [row for row in raters if sorted([row['first_group'],row['second_group']])==sorted([first,second])]
                values = [row['pearson'] for row in selected if row['pearson'] is not None]
                summaries.append([GROUPS[first][0]+' × '+GROUPS[second][0],len(selected),len(values),st.median(values) if values else None,min(values) if values else None,max(values) if values else None])
            parts.append(table(['個人ペアの所属','個人ペア数','算出できたペア数','ピアソン相関の中央値','最小','最大'],summaries)+'</details>')
            people = [row for row in data['individuals'] if (row['experiment'],row['pair']) == (experiment,pair)]
            parts.append('<details><summary>一人ひとりの平均と時間方向のばらつき</summary>'+table(['版','回答者ID','時点数','平均','時間方向SD'],[[GROUPS[row['group']][0]]+[row[key] for key in ['respondent','n','mean','temporal_sd']] for row in people])+'</details></section></details>')
    parts.append('</html>')
    (out/'human_group_report.html').write_text(''.join(parts),encoding='utf-8')


def build_report(out, humans, observers, correlation, ranks, write_csv):
    data = analyze_groups(humans, observers, correlation, ranks)
    for key, rows in data.items():
        write_csv(out/('human_group_'+key+'.csv'),rows)
    render_report(out,data,humans,observers)
    return data
