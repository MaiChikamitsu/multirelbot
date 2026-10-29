"""Scatter plots of paired human group means and system predictions."""
import hashlib
import html
import statistics as st
from collections import defaultdict
from comparison_methods import COMPARISON_METHODS

GROUPS = {'text': ('文字版', '#059669'), 'video': ('動画版', '#dc2626'),
          'experiment': ('実験参加者版', '#2563eb')}


def build_scatter_points(rows, observers):
    people = defaultdict(dict)
    systems = {}
    for row in rows:
        if row['method'] not in COMPARISON_METHODS:
            continue
        systems[row['experiment'],row['run'],row['method'],row['pair'],row['index']] = row['prediction']
        if row['target']=='individual':
            people[row['experiment'],'experiment',row['pair'],row['index']][row['respondent']] = row['human']
    for row in observers:
        group = {'文字版':'text','動画版':'video'}.get(row['label'])
        if group:
            people[row['experiment'],group,row['pair'],row['index']][row['respondent']] = row['human']
    result = []
    for (experiment,run,method,pair,index),prediction in sorted(systems.items()):
        for group in GROUPS:
            ratings = people.get((experiment,group,pair,index))
            if ratings:
                result.append(dict(experiment=experiment,run=run,method=method,group=group,pair=pair,
                    index=index,human=st.mean(ratings.values()),prediction=prediction,n_raters=len(ratings)))
    return result


def scatter_chart(points, correlation, ranks):
    name,color = GROUPS[points[0]['group']]
    human = [row['human'] for row in points]
    system = [row['prediction'] for row in points]
    pearson,spearman = correlation(human,system),correlation(ranks(human),ranks(system))
    x = lambda value: 75+(value+1)*150
    y = lambda value: 45+(1-value)*150
    identity = '|'.join(str(points[0][key]) for key in ['experiment','run','method','group'])
    clip = 'scatter-'+hashlib.sha256(identity.encode()).hexdigest()[:16]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 450 460" style="width:100%;height:auto" role="img" aria-label="'+html.escape(name+'と'+points[0]['method']+'の相関散布図')+'"><rect width="450" height="460" fill="white"/>',
        '<text x="225" y="24" text-anchor="middle" font-size="17">'+html.escape(name+' ／ '+points[0]['method'])+'</text>',
        f'<defs><clipPath id="{clip}"><rect x="75" y="45" width="300" height="300"/></clipPath></defs>']
    for value in [-1,-.5,0,.5,1]:
        parts.append(f'<path d="M{x(value):.2f} 45 V345 M75 {y(value):.2f} H375" stroke="#e2e8f0"/><text x="{x(value):.2f}" y="365" text-anchor="middle" font-size="13">{value:g}</text><text x="64" y="{y(value)+5:.2f}" text-anchor="end" font-size="13">{value:g}</text>')
    parts.append('<path d="M75 45 V345 H375" fill="none" stroke="#475569"/><text x="225" y="390" text-anchor="middle" font-size="15">人の平均評定</text><text transform="translate(20 195) rotate(-90)" text-anchor="middle" font-size="15">システム推定値</text>')
    parts.append(f'<g clip-path="url(#{clip})"><path d="M75 345 L375 45" stroke="#cbd5e1" stroke-width="1.5"/>')
    for row in points:
        source = row.get('source_experiment',row['experiment'])
        title = f'{source}／{row["pair"]}／評定番号 {row["index"]}：人 {row["human"]:.3f}、システム {row["prediction"]:.3f}（回答者 {row["n_raters"]}人）'
        parts.append(f'<circle class="scatter-point" data-human="{row["human"]}" data-system="{row["prediction"]}" cx="{x(row["human"]):.2f}" cy="{y(row["prediction"]):.2f}" r="4" fill="{color}" fill-opacity="0.55"><title>{html.escape(title)}</title></circle>')
    if len(set(human))>1:
        mean_x,mean_y = st.mean(human),st.mean(system)
        slope = sum((left-mean_x)*(right-mean_y) for left,right in zip(human,system))/sum((value-mean_x)**2 for value in human)
        intercept = mean_y-slope*mean_x
        parts.append(f'<path class="regression-line" d="M75 {y(intercept-slope):.2f} L375 {y(intercept+slope):.2f}" stroke="#111827" stroke-width="2" fill="none"/>')
    parts.append('</g>')
    r = '算出不可' if pearson is None else f'{pearson:+.3f}'
    rho = '算出不可' if spearman is None else f'{spearman:+.3f}'
    mae = st.mean(abs(left-right) for left,right in zip(human,system))
    parts.append(f'<text x="225" y="419" text-anchor="middle" font-size="14">n={len(points)}　ピアソン r={r}　スピアマン ρ={rho}</text><text x="225" y="445" text-anchor="middle" font-size="14">MAE={mae:.3f}（小さいほど一致）</text></svg>')
    return ''.join(parts)


def render_scatter(points):
    if not points:
        return ''
    from analyze_relations import corr,ranks
    parts = ['<section id="scatter-correlations"><h2>システムと人の評価の相関：散布図</h2>',
        '<p>横軸は人の平均評定、縦軸はシステム推定値。1点は「1つのペア・1つの評定番号」で、A-B・B-C・C-Aを同じ図にまとめています。</p>',
        '<p>右上がりなら正の相関、右下がりなら負の相関です。黒線は回帰直線、薄い灰色の斜線は人とシステムの点数が等しい位置（y=x）。点の色は文字版＝緑、動画版＝赤、実験参加者版＝青です。点は重なる場合があります。</p>',
        '<style>.scatter-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.scatter-card{border:1px solid #dce2ea;border-radius:10px;padding:12px}.scatter-card h4{margin:0;text-align:center}@media(max-width:1000px){.scatter-grid{grid-template-columns:1fr}}</style>']
    for experiment,run in sorted({(row['experiment'],row['run']) for row in points}):
        title = {'ai_nashi':'AI・縛りなし','muzintou_ari':'無人島・縛りあり','combined_ai_muzintou':'AI＋無人島（合算）'}.get(experiment,experiment)
        parts.append('<h3>'+html.escape(title)+' ／ '+html.escape(run)+'</h3>')
        for method in COMPARISON_METHODS:
            parts.append('<h3>'+method+'</h3><div class="scatter-grid">')
            for group,(name,_) in GROUPS.items():
                selected = [row for row in points if (row['experiment'],row['run'],row['method'],row['group'])==(experiment,run,method,group)]
                if selected:
                    parts.append('<article class="scatter-card"><h4>'+name+'</h4>'+scatter_chart(selected,corr,ranks)+'</article>')
            parts.append('</div>')
    parts.append('<p>各図はグループ平均との相関です。ペア間の評価水準の差も含まれ、個人同士の一致や統計的な有意性を示すものではありません。</p></section>')
    return ''.join(parts)
