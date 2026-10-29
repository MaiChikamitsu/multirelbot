import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from analyze_relations import analyze
from report_management import load_exclusions
from serve_relation_reports import ReportManager
from multi_experiment_analysis import SETTING_KEYS

class ManagementTest(unittest.TestCase):
    def test_exclude_restore_and_overall_membership(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);experiments=[];source_bytes={}
            for n in [1,2]:
                eid=f'experiment{n}';folder=root/eid;folder.mkdir()
                survey=folder/'survey.xlsx';survey.write_bytes(b'fixture')
                pred=folder/'predictions.csv'
                with pred.open('w',newline='') as f:
                    writer=csv.writer(f);writer.writerow(['Method','Episode','Run','Round','Pair','Prediction'])
                    for method in ['Interaction Frequency','Stance-based','Sentiment-based','LLM-only','LLM + SMA','MAVeRD']:
                        for i in [1,2]:
                            for pair in ['A-B','A-C','B-C']:writer.writerow([method,eid,'run'+str(n),i,pair,n*.1])
                source_bytes[pred]=pred.read_bytes()
                (folder/'metadata.json').write_text(json.dumps({k:1 for k in SETTING_KEYS}))
                experiments.append(dict(id=eid,theme=eid,survey=str(survey),episode=eid,predictions=str(pred),aggregate_run='run'+str(n),condition_id='same'))
            config=root/'config.json';config.write_text(json.dumps(dict(experiments=experiments)))
            def fake_survey(path,theme,*args):
                n=int(theme[-1])
                return [dict(respondent=str((n-1)*3+j),role=role,theme=theme,index=i,pair=pair,human=0,source_cell='fixture') for j,role in enumerate('ABC',1) for i in [1,2] for pair in ['A-B','A-C','B-C']]
            with patch('analyze_relations.read_survey',side_effect=fake_survey):
                output=analyze(config)
                def csv_rows(name):
                    with (output/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
                self.assertEqual({r['n_experiments'] for r in csv_rows('overall_metrics.csv')},{'2'})
                manager=ReportManager(config)
                manager.update('experiment2','run2','exclude')
                self.assertEqual(len(load_exclusions(config)),1)
                self.assertEqual({r['experiment'] for r in json.loads((output/'report_manifest.json').read_text())},{'experiment1'})
                self.assertEqual({r['n_experiments'] for r in csv_rows('overall_metrics.csv')},{'1'})
                self.assertEqual(next(r['status'] for r in csv_rows('experiment_selection.csv') if r['experiment']=='experiment2'),'excluded_by_user')
                included=(output/'overall_report.html').read_text().split('今回の総合値に含まれる実験</h2>')[1].split('</section>')[0]
                self.assertIn('experiment1.txt',included);self.assertNotIn('experiment2.txt',included)
                analyze(config) # Exclusion survives a fresh command.
                self.assertEqual({r['experiment'] for r in csv_rows('predictions.csv')},{'experiment1'})
                manager.update('experiment2','run2','restore')
                self.assertEqual(load_exclusions(config),[])
                self.assertEqual({r['n_experiments'] for r in csv_rows('overall_metrics.csv')},{'2'})
                with self.assertRaisesRegex(ValueError,'Unknown'):manager.update('unknown','run1','exclude')
                for file,expected in source_bytes.items():self.assertEqual(file.read_bytes(),expected)
                failing=ReportManager(config,analyze=lambda path:(_ for _ in ()).throw(ValueError('simulated error')))
                with self.assertRaises(ValueError):failing.update('experiment2','run2','exclude')
                self.assertEqual(load_exclusions(config),[])

if __name__=='__main__':unittest.main()
