import csv
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import text_relation_viewer as viewer

class ViewerAutosaveTest(unittest.TestCase):
    def test_saves_six_methods_each_round_and_keeps_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            source=root/'sample.txt'
            source.write_text('[A]いい\n[B]なるほど\n[C]ありがとう\n[A]はい\n[B]いいね\n[C]うん\n')
            raw={('A','B'):.3,('A','C'):.6,('B','C'):-.3}
            with patch.object(viewer.estimator,'estimate_relation_once',return_value=raw), patch.object(viewer.estimator,'DEBUG',False), patch.object(viewer.config,'get_config',return_value=SimpleNamespace(llm=SimpleNamespace(relation_model='test',azure_model='test',relation_temperature=.3))):
                for _ in range(2):
                    app=viewer.create_app(source,interval=0,output_dir=root/'runs')
                    client=app.test_client()
                    self.assertEqual(client.post('/control/start').status_code,200)
                    deadline=time.monotonic()+3
                    while not client.get('/state').json['done'] and time.monotonic()<deadline: time.sleep(.01)
                    state=client.get('/state').json
                    self.assertTrue(state['done'])
                    run=Path(state['output'])
                    with (run/'predictions.csv').open() as f: rows=list(csv.DictReader(f))
                    self.assertEqual(len(rows),36)
                    self.assertEqual({r['Method'] for r in rows},set(viewer.METHODS))
                    self.assertEqual({r['Human_Turn'] for r in rows},{'3','6'})
                    self.assertEqual({r['Log_Index'] for r in rows},{'2','5'})
                    self.assertEqual(json.loads((run/'metadata.json').read_text())['window_size'],6)
                self.assertEqual(len(list((root/'runs').glob('*/predictions.csv'))),2)

class ViewerTrialsTest(unittest.TestCase):
    def run_viewer(self, root, responses, num_trials):
        source = root / 'sample.txt'
        source.write_text('[A]はい\n[B]なるほど\n[A]いいね\n[C]うん\n[B]そうですね\n[A]はい\n')
        cfg = SimpleNamespace(llm=SimpleNamespace(relation_model='test', azure_model='test', relation_temperature=.3))
        with patch.object(viewer.estimator, 'estimate_relation_once', side_effect=responses) as call, patch.object(viewer.estimator, 'DEBUG', False), patch.object(viewer.config, 'get_config', return_value=cfg):
            app = viewer.create_app(source, interval=0, output_dir=root/'runs', num_trials=num_trials)
            client = app.test_client()
            self.assertEqual(call.call_count, 0)
            client.post('/control/start')
            deadline = time.monotonic() + 5
            while not client.get('/state').json['done'] and time.monotonic() < deadline:
                time.sleep(.01)
            state = client.get('/state').json
            self.assertTrue(state['done'])
            count = call.call_count
        return Path(state['output']), state, count

    def test_ten_independent_trials_and_analysis_compatibility(self):
        from statistics import mean
        pairs = [('A','B'), ('A','C'), ('B','C')]
        responses = [{p: (i-5)/10 for p in pairs} for i in range(10)]
        responses += [{p: (4-i)/10 for p in pairs} for i in range(10)]
        with tempfile.TemporaryDirectory() as td:
            run, state, count = self.run_viewer(Path(td), responses, 10)
            self.assertEqual(count, 20)
            self.assertEqual(state['status'], '完了')
            with (run/'predictions.csv').open() as f: rows = list(csv.DictReader(f))
            with (run/'trial_predictions.csv').open() as f: trials = list(csv.DictReader(f))
            self.assertEqual(len(rows), 36)
            self.assertEqual(len(trials), 180)
            self.assertEqual(json.loads((run/'metadata.json').read_text())['num_trials'], 10)
            for row in rows:
                matching = [float(t['Prediction']) for t in trials if t['Method']==row['Method'] and t['Round']==row['Round'] and t['Pair']==row['Pair']]
                if matching:
                    self.assertEqual(len(matching), 10)
                    self.assertAlmostEqual(float(row['Prediction']), mean(matching))
                    self.assertEqual(row['Num_Trials'], '10')
                else:
                    self.assertEqual(row['Num_Trials'], '1')
            # Independently computed EMA: first-round C count is zero; in
            # round two A-C/B-C have no historical joint utterance weight.
            second = [r for r in rows if r['Method']=='MAVeRD' and r['Round']=='2' and r['Pair']=='A–C'][0]
            self.assertAlmostEqual(float(second['Prediction']), -.05)
            raw_second = next(t for t in trials if t['Method']=='MAVeRD' and t['Round']=='2' and t['Pair']=='A–B' and t['Trial']=='1')
            self.assertAlmostEqual(float(raw_second['Prediction']), (1-1/1.8)*(-.5)+(1/1.8)*.4)
            import analyze_relations as analysis
            self.assertAlmostEqual(analysis.number(second['Prediction']), -.05)

    def test_failure_does_not_publish_partial_trial_average(self):
        raw = {('A','B'):-.3, ('A','C'):.2, ('B','C'):.1}
        with tempfile.TemporaryDirectory() as td:
            # Complete round 1, then fail after one successful trial in round 2.
            run, state, count = self.run_viewer(Path(td), [raw,raw,raw,RuntimeError('test failure')], 2)
            self.assertIn('推定失敗', state['status'])
            self.assertEqual(count, 4)
            with (run/'predictions.csv').open() as f: rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 18)
            self.assertEqual({r['Round'] for r in rows}, {'1'})
            with (run/'trial_predictions.csv').open() as f: trials = list(csv.DictReader(f))
            self.assertEqual(len(trials), 27)

if __name__=='__main__': unittest.main()
