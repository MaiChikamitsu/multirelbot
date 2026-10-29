import unittest

from analyze_relations import corr,ranks
from combined_comparison import build_combined,render_combined


class CombinedTests(unittest.TestCase):
    def fixture(self):
        config = dict(experiments=[dict(id=experiment,aggregate_run='chosen',condition_id='same')
                                   for experiment in ['ai_nashi','muzintou_ari']])
        points = [dict(experiment=experiment,run=run,method=method,group=group,pair=pair,index=index,
                       human=value,prediction=value+.1,n_raters=3)
                  for experiment,values in [('ai_nashi',[(1,-.5),(2,-.3)]),('muzintou_ari',[(1,.1),(2,.3),(4,.6)])]
                  for run in ['chosen','unused'] for method in ['LLM-only','LLM + SMA','MAVeRD']
                  for group in ['text','video','experiment'] for pair in ['A-B','A-C'] for index,value in values]
        return points,config

    def test_pool_selected_runs_without_boundary_changes(self):
        points,config=self.fixture()
        combined,metrics,message=build_combined(points,config,corr,ranks)
        self.assertEqual(message,'')
        self.assertEqual(len(combined),90)
        self.assertEqual(len(metrics),18)
        self.assertEqual({row['source_run'] for row in combined},{'chosen'})
        for row in metrics:
            self.assertEqual(row['n'],10 if row['measure']=='level' else 4)
            if row['measure']=='level':
                self.assertAlmostEqual(row['pearson'],1)
                self.assertAlmostEqual(row['mae'],.1)
        self.assertIn('AI＋無人島',render_combined(combined,metrics))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            build_combined(points+points[:1],config,corr,ranks)

    def test_missing_selection_and_condition_mismatch(self):
        points,config=self.fixture()
        config['experiments'][0]['aggregate_run']='missing'
        self.assertTrue(build_combined(points,config,corr,ranks)[2])
        config['experiments'][0]['aggregate_run']='chosen'
        config['experiments'][0]['condition_id']='different'
        self.assertIn('condition_id',build_combined(points,config,corr,ranks)[2])


if __name__=='__main__':
    unittest.main()
