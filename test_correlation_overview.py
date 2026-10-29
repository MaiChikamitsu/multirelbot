import unittest

from analyze_relations import corr
from correlation_overview import build_correlations, render_summary


class CorrelationOverviewTests(unittest.TestCase):
    def test_group_means_individuals_and_gaps(self):
        rows = [dict(experiment='test', run='run', method=method, pair='A-B',
                     target='individual', respondent='1', index=index, prediction=value, human=value)
                for method in ['MAVeRD', 'LLM-only', 'LLM + SMA', 'Stance-based']
                for index, value in [(1, 0), (2, .2), (4, .6)]]
        observers = [dict(experiment='test', pair='A-B', label=label, respondent='9001',
                          index=index, human=-value)
                     for label in ['文字版', '動画版'] for index, value in [(1, 0), (2, .2), (4, .6)]]
        result = build_correlations(rows, observers, corr)
        self.assertEqual(len(result), 36)
        self.assertEqual({row['method'] for row in result}, {'MAVeRD', 'LLM-only', 'LLM + SMA'})
        for row in result:
            if row['measure'] == 'change':
                self.assertEqual(row['n'], 1)
                self.assertIsNone(row['pearson'])
            else:
                self.assertAlmostEqual(row['pearson'], 1 if row['group'] == 'experiment' else -1)
        content = render_summary(result)
        self.assertIn('ピアソンの積率相関係数', content)
        self.assertIn('スピアマンの順位相関係数', content)
        self.assertIn('一人ひとりの相関係数', content)
        self.assertTrue(all(row['mae'] is not None for row in result))

    def test_signed_bars_and_undefined_values(self):
        from xml.etree import ElementTree
        from correlation_overview import comparison_bars
        records = [dict(group='text', pair='A-B', method=method, pearson=value, n=23)
                   for method, value in [('LLM-only', -.4), ('MAVeRD', .8)]]
        svg = ElementTree.fromstring(comparison_bars(records, 'pearson'))
        bars = [node for node in svg if node.attrib.get('class') == 'correlation-bar']
        self.assertEqual(len(bars), 2)
        self.assertAlmostEqual(float(bars[0].attrib['x']) + float(bars[0].attrib['width']), 575)
        self.assertAlmostEqual(float(bars[1].attrib['x']), 575)
        self.assertAlmostEqual(float(bars[1].attrib['width']), 216)
        records[1]['pearson'] = None
        content = comparison_bars(records, 'pearson')
        svg = ElementTree.fromstring(content)
        self.assertEqual(len([node for node in svg if node.attrib.get('class') == 'correlation-bar']), 1)
        self.assertIn('算出不可', content)

    def test_pooled_points_not_average_correlations_and_no_cross_pair_changes(self):
        from correlation_overview import build_pooled_correlations, render_pooled_graphs
        rows = [dict(experiment='test',run='run',method=method,pair=pair,target='individual',
                     respondent='1',index=index,prediction=value,human=value)
                for method in ['MAVeRD','LLM-only','LLM + SMA']
                for pair,points in [('A-B',[(1,0),(2,0)]),('B-C',[(1,.5),(2,.5)])]
                for index,value in points]
        records=build_pooled_correlations(rows,[],corr)
        self.assertEqual(len(records),6)
        for row in records:
            if row['measure']=='level':
                self.assertEqual(row['n'],4)
                self.assertAlmostEqual(row['pearson'],1)
            else:
                self.assertEqual(row['n'],2)
                self.assertIsNone(row['pearson'])
        self.assertIn('全ペアまとめ',render_pooled_graphs(records))

    def test_mae_bars_start_at_zero_and_have_shared_scale(self):
        from xml.etree import ElementTree
        from correlation_overview import comparison_bars
        records=[dict(group='text',pair='all',method=method,mae=value,n=10)
                 for method,value in [('LLM-only',.4),('LLM + SMA',.2),('MAVeRD',0)]]
        svg=ElementTree.fromstring(comparison_bars(records,'mae',pooled=True,axis_limit=.5))
        bars=[node for node in svg if node.attrib.get('class')=='mae-bar']
        self.assertEqual(len(bars),3)
        self.assertTrue(all(float(bar.attrib['x'])==305 for bar in bars))
        self.assertEqual([float(bar.attrib['width']) for bar in bars],[432,216,0])


if __name__ == '__main__':
    unittest.main()
