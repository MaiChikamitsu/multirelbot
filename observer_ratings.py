"""Supplementary observer ratings, kept separate from participant aggregates."""
import hashlib
import html
import re
from collections import defaultdict

from analyze_relations import _survey_rows, normalize_id, number, pair, corr, write_csv
from relation_trends import render_trends


def load_observers(sources, root):
    ratings = []
    identities = set()
    for source in sources:
        path = root / source['survey']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = _survey_rows(path, source.get('sheet', 'フォームの回答 1'))
        headers = rows[0]
        id_column = next(col for col, label in headers.items() if label == '回答者ID')
        theme_column = next(col for col, label in headers.items() if label == 'テーマ')
        columns = {}
        for column, label in headers.items():
            match = re.fullmatch(r'(\d+)\.\s*([ABC])と([ABC])', label)
            if match:
                columns[column] = (int(match[1]), pair(match[2] + match[3]))
        if not columns:
            raise ValueError(f'No rating columns: {path}')
        for row_number, row in enumerate(rows[1:], 2):
            if row.get(theme_column) != source['theme']:
                continue
            respondent = normalize_id(row.get(id_column, ''))
            identity = (source['experiment'], source['label'], respondent)
            if not respondent or identity in identities:
                raise ValueError(f'Missing or duplicate observer: {identity}')
            identities.add(identity)
            for column, (index, rating_pair) in columns.items():
                value = row.get(column, '')
                if not str(value).strip():
                    continue
                ratings.append(dict(experiment=source['experiment'], label=source['label'],
                    respondent=respondent, index=index, pair=rating_pair, human=number(value),
                    source=str(path), source_cell=f'{column}{row_number}',
                    sha256=digest))
        if not any(r['experiment'] == source['experiment'] and r['label'] == source['label'] for r in ratings):
            raise ValueError(f'No observer ratings: {path}')
    return ratings


def render_observers(folder, rows, ratings):
    experiment = rows[0]['experiment'] if rows else None
    selected = [rating for rating in ratings if rating['experiment'] == experiment]
    if not selected:
        return ''
    systems = {(row['method'], row['index'], row['pair']): row for row in rows}
    comparisons = []
    for rating in selected:
        for (method, index, rating_pair), system in systems.items():
            if (index, rating_pair) != (rating['index'], rating['pair']):
                continue
            comparisons.append(dict(experiment=experiment, run=system['run'], method=method,
                index=index, pair=rating_pair, prediction=system['prediction'],
                human=rating['human'], respondent=rating['respondent'], target='individual',
                label=rating['label']))
    write_csv(folder/'observer_comparisons.csv', comparisons)
    parts = ['<section id="observer-trends"><h2>一人ひとりの評価の流れ：動画版・文字版</h2>',
        '<p>追加の回答は版・回答者ごとに表示します。参加者3人の平均と総合評価には含めていません。</p>',
        '<p><a href="observer_comparisons.csv">追加回答の比較CSV</a></p>']
    grouped = defaultdict(list)
    for row in selected:
        grouped[row['label'], row['respondent']].append(row)
    for position, ((label, respondent), answers) in enumerate(sorted(grouped.items())):
        subset = [row for row in comparisons if (row['label'], row['respondent']) == (label, respondent)]
        parts.append('<details><summary>'+html.escape(label+' ／ 回答者ID '+respondent)+'</summary>')
        values = {(row['index'], row['pair']): row['human'] for row in answers}
        parts.append('<h3>回答値一覧</h3><table><tr><th>評定番号</th><th>A-B</th><th>B-C</th><th>A-C</th></tr>')
        for index in sorted({row['index'] for row in answers}):
            cells = [str(index)] + [str(values.get((index, rating_pair), '未回答')) for rating_pair in ['A-B', 'B-C', 'A-C']]
            parts.append('<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in cells)+'</tr>')
        parts.append('</table>')
        labeled = [dict(row, respondent=label+' 回答者ID '+respondent) for row in subset]
        content = render_trends(labeled, corr).replace('id="individual-trends"', f'id="observer-{position}"')
        parts.append(content+'</details>')
    parts.append('</section>')
    return ''.join(parts)
