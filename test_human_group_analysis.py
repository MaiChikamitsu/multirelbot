import math
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from analyze_relations import corr, ranks, write_csv
from human_group_analysis import analyze_groups, build_report, timeline, histogram


class HumanGroupAnalysisTests(unittest.TestCase):
    def fixtures(self):
        humans = [dict(experiment='test',pair='A-B',respondent=person,index=index,human=value+offset)
                  for person,offset in [('1',0),('2',.2)]
                  for index,value in [(1,-.5),(2,-.1),(4,.3)]]
        observers = [dict(experiment='test',pair='A-B',label=label,respondent=person,index=index,human=value+offset)
                     for label,offset in [('文字版',.4),('動画版',0)]
                     for person in ['9001','9002'] for index,value in [(1,-.5),(2,-.1),(4,.3)]]
        return humans,observers

    def test_difference_correlation_and_within_variation(self):
        humans,observers=self.fixtures()
        data=analyze_groups(humans,observers,corr,ranks)
        comparison=next(row for row in data['comparisons'] if (row['first'],row['second'])==('text','video'))
        self.assertAlmostEqual(comparison['bias'],.4)
        self.assertAlmostEqual(comparison['mae'],.4)
        self.assertAlmostEqual(comparison['pearson'],1)
        self.assertAlmostEqual(comparison['spearman'],1)
        self.assertEqual(comparison['n_changes'],1)
        self.assertIsNone(comparison['change_pearson'])
        participant=next(row for row in data['statistics'] if row['group']=='experiment')
        self.assertAlmostEqual(participant['mean_within_sd'],math.sqrt(.02))
        self.assertEqual(participant['n_people'],2)
        text=next(row for row in data['statistics'] if row['group']=='text')
        self.assertAlmostEqual(text['mean_within_sd'],0)
        self.assertGreater(text['temporal_mean_sd'],0)
        self.assertEqual(len(data['rater_correlations']),15)

    def test_missing_singleton_constant_and_duplicate(self):
        humans=[dict(experiment='test',pair='A-B',respondent='1',index=1,human=0)]
        observers=[dict(experiment='test',pair='A-B',label='動画版',respondent='9001',index=1,human=.2)]
        data=analyze_groups(humans,observers,corr,ranks)
        self.assertTrue(all(row['mean_within_sd'] is None for row in data['statistics']))
        self.assertIsNone(data['comparisons'][0]['pearson'])
        self.assertEqual(data['comparisons'][0]['n_changes'],0)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            analyze_groups(humans+humans,observers,corr,ranks)

    def test_report_and_svg(self):
        humans,observers=self.fixtures()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=build_report(root,humans,observers,corr,ranks,write_csv)
            self.assertIn('回答者間SD',(root/'human_group_report.html').read_text())
            self.assertTrue((root/'human_group_comparisons.csv').exists())
            ElementTree.fromstring(timeline(data['points']))
            ElementTree.fromstring(timeline(data['points'],spread=True))
            svg=ElementTree.fromstring(histogram([('text',[-1,0,1]),('video',[.2])]))
            bars=svg.findall('{http://www.w3.org/2000/svg}rect')[1:]
            self.assertEqual(len(bars),30)
            self.assertAlmostEqual(sum(float(bar.attrib['height']) for bar in bars),440,places=1)
            boundary=ElementTree.fromstring(histogram([('text',[-.8])]))
            boundary_bars=boundary.findall('{http://www.w3.org/2000/svg}rect')[1:]
            self.assertEqual(float(boundary_bars[0].attrib['height']),0)
            self.assertEqual(float(boundary_bars[3].attrib['height']),220)

    def test_visual_overview_values_and_missing_correlations(self):
        import re
        from human_group_analysis import render_visual_overview
        humans,observers=self.fixtures()
        data=analyze_groups(humans,observers,corr,ranks)
        content=render_visual_overview(data)
        charts=re.findall(r'<svg\b.*?</svg>',content,re.S)
        self.assertEqual(len(charts),4)
        for chart in charts:
            svg=ElementTree.fromstring(chart)
            values=[node for node in svg if node.attrib.get('class')=='overview-value']
            self.assertEqual(len(values),3)
        data['comparisons'][0]['pearson']=None
        content=render_visual_overview(data)
        self.assertIn('算出不可',content)


if __name__=='__main__':
    unittest.main()
