import unittest
from relation_trends import calculate_trends, direction, render_trends
from analyze_relations import corr

class TrendTest(unittest.TestCase):
    def row(self,index,system,human):
        return dict(experiment='test',run='run1',method='LLM-only',pair='A-C',target='individual',respondent='1',index=index,prediction=system,human=human)

    def test_example(self):
        rows=[self.row(1,.1,-.3),self.row(2,.5,.2),self.row(3,.5,.2)]
        summaries,changes=calculate_trends(rows,corr)
        r=summaries[0]
        self.assertEqual(r['n_changes'],2)
        self.assertAlmostEqual(changes[0]['system_change'],.4)
        self.assertAlmostEqual(changes[0]['human_change'],.5)
        self.assertAlmostEqual(r['change_mae'],.05)
        self.assertEqual(r['direction_accuracy'],1)
        self.assertEqual(r['human_changed_n'],1)
        self.assertEqual(r['direction_accuracy_when_human_changed'],1)
        self.assertIn('一人ひとりの評価の流れ',render_trends(rows,corr))

    def test_gap_and_constant(self):
        r,changes=calculate_trends([self.row(1,0,0),self.row(3,0,.5)],corr)
        self.assertEqual(changes,[])
        self.assertIsNone(r[0]['change_mae'])
        self.assertIsNone(r[0]['level_pearson'])
        self.assertEqual(direction(1e-15),0)
        r,_=calculate_trends([self.row(1,0,0),self.row(2,0,0)],corr)
        self.assertEqual(r[0]['direction_accuracy'],1)
        self.assertIsNone(r[0]['direction_accuracy_when_human_changed'])

    def test_observers_overlay_existing_chart(self):
        from xml.etree import ElementTree
        import re
        rows = [self.row(1, 0, 0), self.row(2, .2, .1), self.row(3, .4, .2)]
        rows = [dict(row, method='Stance-based') for row in rows]
        observers = [dict(experiment='test', label=label, respondent='9001', pair='A-C',
                          index=index, human=value)
                     for label in ['動画版', '文字版']
                     for index, value in [(1, -.1), (2, .3), (3, .2)]]
        observers.append(dict(experiment='other', label='動画版', respondent='wrong', pair='A-C', index=1, human=1))
        content = render_trends(rows, corr, observers)
        charts = re.findall(r'<svg\b.*?</svg>', content, re.S)
        self.assertEqual(len(charts), 2)
        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        for chart in charts:
            lines = ElementTree.fromstring(chart).findall('svg:polyline', namespace)
            self.assertEqual(len(lines), 4)
            self.assertEqual(len({line.attrib['stroke'] for line in lines}), 4)
            self.assertEqual({line.attrib.get('stroke-dasharray') for line in lines}, {None})
        self.assertIn('動画版 回答者ID 9001', content)
        self.assertIn('文字版 回答者ID 9001', content)
        self.assertNotIn('wrong', content)

    def test_selected_methods_split_groups_with_system(self):
        from xml.etree import ElementTree
        import re
        for method in ['MAVeRD', 'LLM-only', 'LLM + SMA']:
            rows = [dict(self.row(index, value, value), method=method)
                    for index, value in [(1, 0), (2, .2)]]
            observers = [dict(experiment='test', label=label, respondent='9001', pair='A-C',
                              index=index, human=value)
                         for label in ['文字版', '動画版'] for index, value in [(1, .1), (2, .3)]]
            content = render_trends(rows, corr, observers)
            charts = re.findall(r'<svg\b.*?</svg>', content, re.S)
            self.assertEqual(len(charts), 6)
            for position, chart in enumerate(charts):
                lines = ElementTree.fromstring(chart).findall('{http://www.w3.org/2000/svg}polyline')
                self.assertEqual([line.attrib['stroke'] for line in lines],
                                 [['#059669', '#111827'], ['#dc2626', '#111827'],
                                  ['#2563eb', '#111827']][position // 2])
            self.assertIn('文字版とシステム', content)
            self.assertIn('動画版とシステム', content)
            self.assertIn('実験参加者版とシステム', content)

    def test_group_colors_are_shared_and_stable(self):
        from relation_trends import chart
        from xml.etree import ElementTree
        import re
        series = {key: [(1, 0), (2, .2)] for key in [
            '1', '2', '動画版 回答者ID 9001', '動画版 回答者ID 9002',
            '文字版 回答者ID 9001', '文字版 回答者ID 9002', 'system']}
        for change in [False, True]:
            content = chart(series, change)
            svg = ElementTree.fromstring(re.search(r'<svg\b.*?</svg>', content, re.S).group())
            lines = svg.findall('{http://www.w3.org/2000/svg}polyline')
            self.assertEqual([line.attrib['stroke'] for line in lines],
                             ['#2563eb', '#2563eb', '#dc2626', '#dc2626', '#059669', '#059669', '#111827'])
            self.assertIn('青：実験', content)
            self.assertIn('赤：動画版', content)
            self.assertIn('緑：文字版', content)

    def test_correlations_align_points_and_average_people(self):
        from relation_trends import correlation_rows, correlation_table
        series = {'system': [(1, 0), (2, .5), (3, 1)],
                  '1': [(1, 0), (2, .5), (3, 1)],
                  '2': [(1, 1), (2, .5), (3, 0)],
                  '3': [(2, .2), (3, .4), (4, 1)]}
        result = correlation_rows(series, corr)
        self.assertEqual(result[0]['label'], 'このグループの平均')
        self.assertEqual(result[1]['pearson'], 1)
        self.assertEqual(result[2]['pearson'], -1)
        self.assertAlmostEqual(result[3]['pearson'], 1)
        self.assertEqual(result[3]['n'], 2)
        constant = {'system': [(1, 0), (2, 1)], '1': [(1, .5), (2, .5)]}
        self.assertIsNone(correlation_rows(constant, corr)[0]['pearson'])
        self.assertIn('算出不可', correlation_table(constant, corr))
        self.assertAlmostEqual(correlation_rows(constant,corr)[0]['mae'],.5)

if __name__=='__main__':unittest.main()
