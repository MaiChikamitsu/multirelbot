"""Offline questionnaire comparison. Standard library only; never calls an API."""
from __future__ import annotations
import argparse
import csv
import glob
import hashlib
import html
import json
import math
import re
import statistics as st
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

METHODS = ['Interaction Frequency', 'Stance-based', 'Sentiment-based', 'LLM-only', 'LLM + SMA', 'MAVeRD']
PAIRS = ['A-B', 'B-C', 'A-C']


def number(value):
    value = unicodedata.normalize('NFKC', str(value)).strip().replace('−', '-')
    result = float(value)
    if not math.isfinite(result) or not -1 <= result <= 1:
        raise ValueError(f'Expected finite score in [-1,1]: {value!r}')
    return result


def pair(value):
    letters = re.findall('[ABC]', value)
    if len(letters) != 2 or letters[0] == letters[1]:
        raise ValueError(f'Invalid pair: {value}')
    return '-'.join(sorted(letters))


def normalize_id(value):
    value = str(value).strip()
    try:
        numeric = float(value)
        if numeric.is_integer(): return str(int(numeric))
    except ValueError:
        pass
    return value


def _survey_rows(path, sheet_name='フォームの回答 2'):
    if Path(path).suffix.lower() == '.csv':
        def column_name(index):
            value=''
            while index:
                index,remainder=divmod(index-1,26)
                value=chr(65+remainder)+value
            return value
        with Path(path).open(encoding='utf-8-sig',newline='') as f:
            return [{column_name(i):v for i,v in enumerate(row,1)} for row in csv.reader(f) if row]
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        sheets = ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet', ns)
        sheet = next(s for s in sheets if s.attrib['name'] == sheet_name)
        rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        target = next(r.attrib['Target'] for r in rels if r.attrib['Id'] == rid)
        target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
        rows = []
        for row in ET.fromstring(z.read(target)).findall('m:sheetData/m:row', ns):
            cells = {}
            for c in row.findall('m:c', ns):
                v = c.find('m:v', ns)
                value = v.text if v is not None else ''
                if c.attrib.get('t') == 's': value = strings[int(value)]
                if c.attrib.get('t') == 'inlineStr': value = ''.join(c.find('m:is', ns).itertext())
                cells[re.sub(r'\d', '', c.attrib['r'])] = value
            rows.append(cells)
    return rows


def read_survey(path, theme, respondent_ids=None, roles=None):
    rows = _survey_rows(path)
    headers = rows[0]
    columns = {}
    for col, label in headers.items():
        match = re.fullmatch(r'(\d+)\.\s*([ABC])と([ABC])', label)
        if match: columns[col] = (int(match[1]), pair(match[2] + match[3]))
    if not columns: raise ValueError('No time-series rating headers found')
    selected = {normalize_id(v) for v in respondent_ids} if respondent_ids is not None else None
    if selected is not None and len(selected) != 3:
        raise ValueError('respondent_ids must contain exactly three distinct IDs')
    result, seen = [], set()
    for row in rows[1:]:
        if row.get('D') != theme: continue
        respondent = normalize_id(row['B'])
        if selected is not None and respondent not in selected: continue
        role = row.get('C', '').strip()
        if role not in 'ABC' or len(role) != 1: raise ValueError('Invalid role')
        if roles is not None and normalize_id(roles.get(role, '')) != respondent:
            raise ValueError(f'Role mapping mismatch: {theme} ID={respondent} role={role}')
        identity = (respondent, role, theme)
        if identity in seen: raise ValueError(f'Duplicate respondent: {identity}')
        seen.add(identity)
        for col, (index, p) in columns.items():
            value = row.get(col, '')
            if str(value).strip() == '': continue
            result.append(dict(respondent=respondent, role=role, theme=theme, index=index,
                               pair=p, human=number(value), source_cell=f'{col}{rows.index(row)+1}'))
    if not result: raise ValueError(f'No ratings for theme {theme!r}')
    found = {r['respondent'] for r in result}
    if selected is not None and found != selected:
        raise ValueError(f'Missing respondents for {theme!r}: {sorted(selected-found)}')
    if len(found) != 3 or {role for _,role,_ in seen} != set('ABC') or len(seen) != 3:
        raise ValueError(f'{theme!r}: select exactly one A/B/C group with respondent_ids; found {len(seen)} responses')
    return result


def write_csv(path, rows, fields=None):
    rows = list(rows)
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]) if rows else fields or ['status'])
        writer.writeheader()
        writer.writerows(rows)


def corr(x, y):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2: return None
    mx, my = st.mean(x), st.mean(y)
    return sum((a-mx)*(b-my) for a,b in zip(x,y)) / math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))


def ranks(x):
    return [1 + sum(v < a for v in x) + (sum(v == a for v in x)-1)/2 for a in x]


def metrics(rows):
    x, y = [r['prediction'] for r in rows], [r['human'] for r in rows]
    return dict(n=len(rows), mae=st.mean(abs(a-b) for a,b in zip(x,y)),
                rmse=math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y))),
                bias=st.mean(a-b for a,b in zip(x,y)), pearson=corr(x,y), spearman=corr(ranks(x), ranks(y)))


def analyze(config_path):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    root = config_path.parent
    from report_management import load_exclusions
    exclusions = load_exclusions(config_path)
    excluded_keys = {(r['experiment'], r['run']) for r in exclusions}
    cfg['_excluded_runs'] = exclusions
    out = root / cfg.get('output', 'analysis_output')
    out.mkdir(parents=True, exist_ok=True)
    humans, predictions, comparisons, coverage, provenance = [], [], [], [], []
    enabled = [e for e in cfg['experiments'] if e.get('enabled', True)]
    if len({e['id'] for e in cfg['experiments']}) != len(cfg['experiments']):
        raise ValueError('Experiment IDs must be unique')
    assigned = set()
    for exp in enabled:
        eid = exp['id']
        survey = root / exp['survey']
        hs = read_survey(survey, exp['theme'], exp.get('respondent_ids'), exp.get('roles'))
        for respondent, role in {(h['respondent'], h['role']) for h in hs}:
            identity = (str(survey.resolve()), exp['theme'], respondent, role)
            if identity in assigned:
                raise ValueError(f'The same response was assigned to multiple sessions: {identity}')
            assigned.add(identity)
        for h in hs: h['experiment'] = eid
        humans.extend(hs)
        groups = defaultdict(list)
        for h in hs: groups[h['index'],h['pair']].append(h)
        files = [Path(p) for p in sorted(glob.glob(str(root / exp['predictions'])))]
        seen = set()
        for file in files:
            provenance.append(dict(path=str(file), sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            with file.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    if row['Episode'] != exp.get('episode', eid): continue
                    method = row['Method']
                    if method not in METHODS: raise ValueError(f'Unknown method: {method}')
                    index = int(row['Round']) + exp.get('index_offset', 0)
                    p = pair(row['Pair'])
                    run = row.get('Run') or file.stem
                    if (eid, run) in excluded_keys:
                        continue
                    key = (method,run,index,p)
                    if key in seen: raise ValueError(f'Duplicate prediction: {eid} {key}')
                    seen.add(key)
                    prediction = number(row['Prediction'])
                    base = dict(experiment=eid, method=method, run=run, index=index, pair=p, prediction=prediction)
                    predictions.append(base)
                    if (index,p) not in groups:
                        coverage.append(dict(experiment=eid, method=method,run=run,index=index,pair=p,status='no_human_rating'))
                        continue
                    vals = groups[index,p]
                    targets = [('mean', 'all', st.mean(v['human'] for v in vals))]
                    targets += [('individual', v['respondent'], v['human']) for v in vals]
                    for target, respondent, human in targets:
                        comparisons.append(dict(base,target=target,respondent=respondent,human=human,n_raters=len(vals),
                                                error=prediction-human,absolute_error=abs(prediction-human)))
        for method in METHODS:
            runs = {key[1] for key in seen if key[0] == method} or {'missing'}
            for run in sorted(runs):
                for index,p in sorted(groups):
                    if (method,run,index,p) not in seen:
                        coverage.append(dict(experiment=eid,method=method,run=run,index=index,pair=p,status='no_prediction'))
        provenance.append(dict(path=str(survey),sha256=hashlib.sha256(survey.read_bytes()).hexdigest()))
    summary = []
    # Per run only: repeated runs are never silently averaged before MAE.
    for dimensions in [('experiment','method','run','target'), ('experiment','method','run','target','pair'), ('experiment','method','run','target','respondent')]:
        grouped = defaultdict(list)
        for row in comparisons:
            # Mean targets have no respondent; their overall metrics already exist.
            if 'respondent' in dimensions and row['target'] != 'individual':
                continue
            grouped[tuple(row[k] for k in dimensions)].append(row)
        for key, vals in sorted(grouped.items()):
            summary.append(dict(dict.fromkeys(['experiment','method','run','target','pair','respondent'], 'all'), **dict(zip(dimensions,key)), **metrics(vals)))
    from relation_trends import calculate_trends
    trend_summary, trend_points = calculate_trends(comparisons, corr)
    # Aggregate interval errors, weighting each valid interval equally.
    for aggregate in summary:
        relevant = [point for point in trend_points
                    if all(point[k] == aggregate[k] for k in ['experiment', 'method', 'run', 'target'])
                    and (aggregate['pair'] == 'all' or point['pair'] == aggregate['pair'])
                    and (aggregate['respondent'] == 'all' or point['respondent'] == aggregate['respondent'])]
        aggregate['change_mae'] = st.mean(point['change_absolute_error'] for point in relevant) if relevant else None
        aggregate['n_changes'] = len(relevant)
    write_csv(out/'human_ratings.csv', humans)
    write_csv(out/'predictions.csv', predictions)
    write_csv(out/'point_comparisons.csv', comparisons)
    write_csv(out/'metrics.csv', summary)
    write_csv(out/'coverage.csv', coverage)
    (out/'provenance.json').write_text(json.dumps(dict(config=cfg,sources=provenance),ensure_ascii=False,indent=2))
    write_csv(out/'trend_metrics.csv', trend_summary)
    write_csv(out/'trend_changes.csv', trend_points)
    from observer_ratings import load_observers
    observers = load_observers(cfg.get('observer_surveys', []), root)
    write_csv(out/'observer_ratings.csv', observers)
    from human_group_analysis import build_report
    build_report(out, humans, observers, corr, ranks, write_csv)
    detail_links = render_index_tables(out, humans, predictions)
    from multi_experiment_analysis import build_overall, render_overall
    overall_rows, selection = build_overall(cfg, root, humans, predictions, comparisons, trend_points)
    write_csv(out/'overall_metrics.csv', overall_rows)
    write_csv(out/'experiment_selection.csv', selection)
    render(out, comparisons, summary, coverage, detail_links, exclusions, observers, cfg)
    render_overall(out, overall_rows, selection)
    print(f'{len(humans)} ratings, {len(predictions)} predictions, {len(comparisons)} comparisons; {out}')
    return out



def render_index_tables(out, humans, predictions):
    """One HTML/CSV per experiment and run; retain every surveyed index."""
    esc = html.escape
    human_groups = defaultdict(dict)
    identities = defaultdict(set)
    for row in humans:
        identity = (row['respondent'], row['role'])
        identities[row['experiment']].add(identity)
        human_groups[row['experiment'], row['index'], row['pair']][identity] = row['human']
    runs = defaultdict(list)
    for row in predictions:
        runs[row['experiment'], row['run']].append(row)
    links = []
    detail_dir = out / 'index_tables'
    detail_dir.mkdir(exist_ok=True)
    for (experiment, run), run_rows in sorted(runs.items()):
        people = sorted(identities[experiment])
        if len(people) != 3:
            raise ValueError(f'{experiment}: six-column view requires exactly three respondents')
        key = experiment + '_' + run
        stem = re.sub(r'[^\w.-]', '_', key) + '_' + hashlib.sha256(key.encode()).hexdigest()[:8]
        pred = {(r['method'], r['index'], r['pair']): r['prediction'] for r in run_rows}
        groups = sorted((index, p) for eid, index, p in human_groups if eid == experiment)
        records = []
        headers = ['インデックス', 'システム値', '3人の回答の平均', '絶対誤差']
        headers += [f'回答者ID {rid}（役割{role}）' for rid, role in people]
        parts = ['<!doctype html><html lang="ja"><meta charset="utf-8"><title>'+esc(key)+' 各点比較</title>',
            '<style>body{font:16px system-ui;color:#182335;max-width:1100px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px 14px;border:1px solid #dce2ea;text-align:right}th{background:#eaf0f8;position:sticky;top:0}th:first-child,td:first-child{text-align:center}tbody tr:nth-child(even){background:#f6f8fb}td:nth-child(4){background:#fff2dc;font-weight:600}section{margin:40px 0 60px}a{color:#2457bb}.missing{color:#8b5b18}nav a{display:inline-block;margin:5px 12px 5px 0}h2{scroll-margin-top:70px}</style>',
            '<p><a href="../report.html">← 集計レポート</a></p><h1>インデックスごとの比較</h1>',
            '<p>実験：'+esc(experiment)+' ／ 実行：'+esc(run)+'</p>',
            '<p>列順：システム値 → 3人の平均 → 絶対誤差 → 各回答者の値。絶対誤差＝|システム値−平均|。数値は表示のみ小数4桁に丸めています。</p>',
            '<p>このページにはこの実行だけを表示しています。未保存の推定は「未保存」、空欄の回答は「未回答」。3人揃わない点は3人平均・誤差を算出しません。</p>',
            '<p><a href="'+esc(stem)+'.csv">この実行のCSVを開く</a></p><nav>']
        for mi, method in enumerate(METHODS):
            parts.append(f'<a href="#method-{mi}">'+esc(method)+'</a>')
        parts.append('</nav>')
        for mi, method in enumerate(METHODS):
            parts.append(f'<h2 id="method-{mi}">'+esc(method)+'</h2>')
            for p in PAIRS:
                parts.append('<section><h3>'+esc(p)+'</h3><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>')
                errors = []
                for index, gp in groups:
                    if gp != p: continue
                    answers = human_groups[experiment,index,p]
                    values = [answers.get(person) for person in people]
                    mean = st.mean(values) if all(v is not None for v in values) else None
                    system = pred.get((method,index,p))
                    error = abs(system-mean) if system is not None and mean is not None else None
                    if error is not None: errors.append(error)
                    record = {'experiment':experiment, 'run':run, 'method':method, 'pair':p, 'index':index,
                              'system':system, 'human_mean':mean, 'absolute_error':error}
                    record.update({f'respondent_{rid}_role_{role}':v for (rid,role),v in zip(people,values)})
                    records.append(record)
                    cells = [str(index)]
                    for j, v in enumerate([system,mean,error]+values):
                        cells.append(f'{v:.4f}' if v is not None else ('未保存' if j==0 else '未回答' if j>=3 else '—'))
                    parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in cells)+'</tr>')
                parts.append('</tbody></table>')
                parts.append('<p>このペアのMAE（上の絶対誤差の平均）：'+(f'{st.mean(errors):.4f}' if errors else '—')+f' ／ 比較できた時点数：{len(errors)}</p></section>')
        parts.append('</html>')
        (detail_dir / (stem+'.html')).write_text(''.join(parts), encoding='utf-8')
        write_csv(detail_dir / (stem+'.csv'), records)
        count = len({r['index'] for r in run_rows})
        links.append((f'{experiment} / {run}（保存済み{count}時点）', 'index_tables/'+stem+'.html'))
    return links



def render(out, rows, summary, coverage, detail_links=(), exclusions=(), observers=(), config=None):
    """Landing page selects one run; each run has its own complete report."""
    esc = html.escape
    from correlation_overview import build_correlations, build_pooled_correlations, render_page, render_summary
    correlations = build_correlations(rows, observers, corr)
    write_csv(out/'correlation_summary.csv', correlations)
    pooled_correlations = build_pooled_correlations(rows, observers, corr)
    write_csv(out/'correlation_pooled.csv', pooled_correlations)
    from correlation_scatter import build_scatter_points
    scatter_points = build_scatter_points(rows, observers)
    write_csv(out/'correlation_scatter_points.csv', scatter_points)
    combined_html = ''
    if config is not None:
        from combined_comparison import build_combined, render_combined
        combined_points, combined_metrics, combined_message = build_combined(scatter_points, config, corr, ranks)
        write_csv(out/'combined_scatter_points.csv', combined_points)
        write_csv(out/'combined_metrics.csv', combined_metrics)
        combined_html = render_combined(combined_points, combined_metrics, combined_message)
    render_page(out, correlations, pooled_correlations, scatter_points, combined_html)
    run_keys = sorted({(r['experiment'], r['run']) for r in rows})
    pages = []
    for experiment, run in run_keys:
        key = experiment + '_' + run
        stem = re.sub(r'[^\w.-]', '_', key) + '_' + hashlib.sha256(key.encode()).hexdigest()[:8]
        folder = out / 'run_reports' / stem
        folder.mkdir(parents=True, exist_ok=True)
        selected_rows = [r for r in rows if (r['experiment'], r['run']) == (experiment, run)]
        selected_summary = [r for r in summary if (r['experiment'], r['run']) == (experiment, run)]
        selected_coverage = [r for r in coverage if (r['experiment'], r['run']) == (experiment, run)]
        selected_links = [(label, '../../'+filename) for label, filename in detail_links
                          if Path(filename).stem == stem]
        selected_observers = [r for r in observers if r['experiment'] == experiment]
        render_run(folder, selected_rows, selected_summary, selected_coverage, selected_links, selected_observers)
        report_path = folder / 'report.html'
        content = report_path.read_text()
        selected_correlations = [r for r in correlations if (r['experiment'], r['run']) == (experiment, run)]
        selected_pooled = [r for r in pooled_correlations if (r['experiment'], r['run']) == (experiment, run)]
        selected_scatter = [r for r in scatter_points if (r['experiment'], r['run']) == (experiment, run)]
        content = content.replace('<h2>インデックスごとの比較', render_summary(selected_correlations, '../../human_group_report.html', selected_pooled, selected_scatter)+'<h2>インデックスごとの比較', 1)
        from observer_ratings import render_observers
        content += render_observers(folder, selected_rows, observers)
        if any(r['experiment'] == experiment for r in observers):
            content = content.replace('<section id="individual-trends">', '<p><a href="#observer-trends">動画版・文字版の一人ひとりの回答へ ↓</a></p><section id="individual-trends">', 1)
        navigation = '<p><a href="../../report.html">← 分析ファイル一覧へ</a>　<a href="#individual-trends">一人ひとりの評価の流れへ ↓</a></p><h2>'+esc(experiment)+' ／ '+esc(run)+'</h2>'
        navigation += '<p><a href="../../human_group_report.html">文字版・動画版・実験参加者版の違い・相関・ばらつきを見る →</a></p>'
        navigation += '<p><a href="../../correlation_report.html#combined-comparison">AI＋無人島を合わせた3手法の比較を見る →</a></p>'
        content = content.replace('<h1>関係性推定と人の評定</h1>', '<h1>関係性推定と人の評定</h1>'+navigation, 1)
        report_path.write_text(content, encoding='utf-8')
        count = len({r['index'] for r in selected_rows})
        pages.append((experiment, run, count, report_path.relative_to(out).as_posix()))
    parts = ['<!doctype html><html lang="ja"><meta charset="utf-8"><title>分析ファイル一覧</title>',
             '<style>body{font:16px system-ui;color:#182335;max-width:1000px;margin:40px auto;padding:0 20px}article{padding:20px;margin:18px 0;border:1px solid #dce2ea;border-radius:10px;background:#f7f9fc}a{color:#2457bb}p{line-height:1.6}</style>',
             '<h1>分析ファイル一覧</h1><p>見たい実行ファイルを選んでください。集計表・グラフ・対応状況は、実行ごとの別ページに表示します。</p>']
    from report_management import button, controls
    parts.append('<p><a href="correlation_report.html">LLM-only・LLM + SMA・MAVeRDの相関係数をまとめて見る →</a></p>')
    parts.append('<p><a href="correlation_report.html#combined-comparison">AI＋無人島を合わせた3手法の散布図・相関・MAEを見る →</a></p>')
    parts.append('<p><a href="correlation_report.html#scatter-correlations">システムと3グループの相関の散布図（全ペアまとめ）を見る →</a></p>')
    parts.append('<p><a href="human_group_report.html">文字版・動画版・実験参加者版の違い・相関・ばらつきを見る →</a></p>')
    parts.append(controls(exclusions))
    manifest = [dict(experiment=e, run=r, count=n, path=p) for e,r,n,p in pages]
    (out/'report_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for experiment, run, count, path in pages:
        parts.append('<article><h2>'+esc(experiment)+'</h2><p>実行：'+esc(run)+f' ／ 保存済み{count}時点</p><a href="'+esc(path, quote=True)+'">このファイルの分析を開く →</a><p>'+button(experiment, run)+'</p></article>')
    if not pages:
        parts.append('<p>分析できる保存済み推定値がありません。</p>')
    parts.append('</html>')
    (out/'report.html').write_text(''.join(parts), encoding='utf-8')


def render_run(out, rows, summary, coverage, detail_links=(), observers=()):
    colors = ['#8b5cf6','#ef4444','#06b6d4','#f59e0b','#22c55e','#2563eb']
    esc = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><title>関係性推定の比較</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;color:#182335}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}svg{width:100%;height:auto}section{margin:30px 0}small{color:#555}</style><h1>関係性推定と人の評定</h1><p>横軸は3人間発話ごとの評定番号。青灰色帯は回答者の最小〜最大、黒線は平均評定。平滑化や時刻補間は加えていません。</p>']
    parts.append('<h2>インデックスごとの比較（実行ファイル別）</h2><p>この実行の各点比較表を別ページで開きます。</p><ul>')
    for label, filename in detail_links:
        parts.append('<li><a href="'+esc(filename, quote=True)+'">'+esc(label)+'</a></li>')
    parts.append('</ul>')
    overall = [r for r in summary if r['pair']=='all' and r['respondent']=='all']
    parts.append('<h2>集計（runごと）</h2><table><tr><th>実験</th><th>手法</th><th>run</th><th>対象</th><th>n</th><th>MAE</th><th>変化量MAE</th><th>変化区間数</th><th>RMSE</th><th>bias</th><th>Pearson</th><th>Spearman</th></tr>')
    for r in overall:
        parts.append('<tr>'+''.join('<td>'+esc(f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k]) if r[k] is not None else '—')+'</td>' for k in ['experiment','method','run','target','n','mae','change_mae','n_changes','rmse','bias','pearson','spearman'])+'</tr>')
    parts.append('</table><p>MAE・変化量MAEは小さいほど一致。変化量MAEは前時点からの変化量の誤差の平均で、欠測をまたぐ区間は除きます。41時点が揃った場合、meanは120区間、individualは360区間です。biasは推定−評定。定数系列の相関は未定義（—）。individualは個人評定との誤差、meanは各点の平均評定との誤差です。</p>')
    for eid in sorted({r['experiment'] for r in rows}):
      for p in PAIRS:
        subset = [r for r in rows if r['experiment']==eid and r['pair']==p and r['target']=='mean']
        if not subset: continue
        maximum = max(r['index'] for r in subset)
        x = lambda i: 60 + (i-1)*1000/max(1,maximum-1)
        y = lambda v: 170-v*125
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 350"><rect width="1120" height="350" fill="white"/>']
        for v in [-1,-.5,0,.5,1]: svg.append(f'<path d="M60 {y(v)} H1060" stroke="#ddd"/><text x="10" y="{y(v)+5}" font-size="14">{v}</text>')
        indices = sorted({r['index'] for r in subset})
        for i in indices:
            if i==1 or i%5==0 or i==maximum: svg.append(f'<text x="{x(i)-5}" y="320" font-size="13">{i}</text>')
        def line(data, color, width):
            return '<polyline fill="none" stroke="'+color+'" stroke-width="'+str(width)+'" points="'+' '.join(f'{x(i):.2f},{y(v):.2f}' for i,v in data)+'"/>'
        human = {r['index']:r['human'] for r in subset}
        individual = [r for r in rows if r['experiment']==eid and r['pair']==p and r['target']=='individual']
        bounds = {i:[r['human'] for r in individual if r['index']==i] for i in indices}
        coords = [(i,min(bounds[i])) for i in indices]+[(i,max(bounds[i])) for i in reversed(indices)]
        svg.append('<polygon fill="#e2e8f0" points="'+' '.join(f'{x(i)},{y(v)}' for i,v in coords)+'"/>')
        svg.append(line(sorted(human.items()), '#111827', 3))
        for method,color in zip(METHODS,colors):
          for run in sorted({r['run'] for r in subset if r['method']==method}):
            data = sorted((r['index'],r['prediction']) for r in subset if r['method']==method and r['run']==run)
            if data: svg.append(line(data,color,1.8))
        svg.append('</svg>')
        filename = re.sub(r'[^\w.-]', '_', eid)+'_'+p+'.svg'
        (out/filename).write_text(''.join(svg))
        parts.append('<section><h2>'+esc(eid+' / '+p)+'</h2>'+''.join(svg)+'<p>'+ '　'.join(f'<span style="color:{c}">{esc(m)}</span>' for m,c in zip(METHODS,colors))+'</p></section>')
    from relation_trends import render_trends, calculate_trends
    local_trends, local_changes = calculate_trends(rows, corr)
    write_csv(out/'trend_metrics.csv', local_trends)
    write_csv(out/'trend_changes.csv', local_changes)
    parts.append('<p><a href="trend_metrics.csv">この実行の変化指標CSV</a> ／ <a href="trend_changes.csv">区間別の変化量CSV</a></p>')
    parts.append(render_trends(rows, corr, observers))
    counts = defaultdict(int)
    for row in coverage: counts[row['experiment'],row['method'],row['status']] += 1
    parts.append('<h2>対応漏れ</h2><pre>'+esc('\n'.join(f'{e} / {m}: {s} = {n}' for (e,m,s),n in sorted(counts.items())))+'</pre><p>欠測は0補完せず除外。詳細はcoverage.csv。母集団や時系列の独立性を仮定した有意差検定は行いません。</p>')
    (out/'report.html').write_text(''.join(parts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    analyze(args.config)
