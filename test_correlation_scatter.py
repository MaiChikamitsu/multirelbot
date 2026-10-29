import unittest
from xml.etree import ElementTree

from analyze_relations import corr,ranks
from correlation_scatter import build_scatter_points,scatter_chart


class ScatterTests(unittest.TestCase):
    def test_mean_points_axes_and_repeated_system_rows(self):
        rows = [dict(experiment='test',run='run',method=method,pair='A-B',index=index,
                     target='individual',respondent=person,prediction=value,human=value+offset)
                for method in ['MAVeRD','LLM-only','LLM + SMA']
                for index,value in [(1,-.5),(2,.5)] for person,offset in [('1',0),('2',.2)]]
        observers = [dict(experiment='test',label='文字版',pair='A-B',index=1,respondent='9001',human=0)]
        points = build_scatter_points(rows,observers)
        self.assertEqual(len(points),9)
        self.assertEqual({row['method'] for row in points},{'MAVeRD','LLM-only','LLM + SMA'})
        selected = [row for row in points if row['group']=='experiment' and row['method']=='MAVeRD']
        self.assertAlmostEqual(selected[0]['human'],-.4)
        self.assertEqual(selected[0]['n_raters'],2)
        svg = ElementTree.fromstring(scatter_chart(selected,corr,ranks))
        circles = svg.findall('.//{http://www.w3.org/2000/svg}circle')
        self.assertEqual(len(circles),2)
        self.assertAlmostEqual(float(circles[0].attrib['cx']),165)
        self.assertAlmostEqual(float(circles[0].attrib['cy']),270)
        self.assertIn('r=+1.000',scatter_chart(selected,corr,ranks))

    def test_constant_human_has_no_regression_line(self):
        points = [dict(experiment='test',run='run',method='LLM-only',group='video',pair='A-B',index=1,
                       human=0,prediction=.5,n_raters=1)]
        content = scatter_chart(points,corr,ranks)
        ElementTree.fromstring(content)
        self.assertNotIn('class="regression-line"',content)
        self.assertIn('算出不可',content)


if __name__=='__main__':
    unittest.main()
