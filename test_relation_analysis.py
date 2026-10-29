"""Regression tests: parsing, independent metrics, missing data, checkpointed six-method run."""
import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import analyze_relations as a


class AnalysisTests(unittest.TestCase):
    def test_numeric_pair_and_metrics(self):
        self.assertEqual(a.number('＋0.3'), .3)
        self.assertEqual(a.pair('C–A'), 'A-C')
        for value in ['nan','inf','2','bad']:
            with self.assertRaises(ValueError): a.number(value)
        m=a.metrics([dict(prediction=0,human=1),dict(prediction=1,human=0)])
        self.assertEqual(m['mae'],1)
        self.assertEqual(m['pearson'],-1)
        self.assertIsNone(a.corr([0,0],[1,2]))
        self.assertEqual(a.ranks([1,1,3]), [1.5,1.5,3])

    def test_real_survey_and_six_method_checkpoint(self):
        import collect_relation_predictions as c
        from argparse import Namespace
        import relation_estimator_from_txt as estimator
        survey=Path('/Users/kinmai/Downloads/【修正版】（直後）予備実験アンケート （回答）.xlsx')
        if not survey.exists(): self.skipTest('Local fixture not available')
        ratings=a.read_survey(survey,'無人島ー縛りあり')
        self.assertEqual(len(ratings),369)
        self.assertEqual(max(r['index'] for r in ratings),41)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            conversation=root/'conversation.txt'
            conversation.write_text('[A]いいですね\n[B]そうですね\n[C]ありがとう\n[A]でも違う\n[B]なるほど\n[C]いいね\n')
            output=root/'run.csv'
            metadata=dict(input_sha256=c.hashlib.sha256(conversation.read_bytes()).hexdigest(),step=3,window=6,episode='test',run='r1',gamma=estimator.GAMMA,history=estimator.MAX_HISTORY_SESSIONS)
            output.with_suffix('.raw.json').write_text(json.dumps(dict(metadata=metadata,raw=[{'A-B':.3,'A-C':.6,'B-C':-.3},{'A-B':.9,'A-C':0,'B-C':.3}])))
            args=Namespace(input=str(conversation),output=str(output),episode='test',run='r1',step=3,window=6,offline_only=False)
            c.collect(args)
            before=output.read_bytes()
            c.collect(args)
            self.assertEqual(before,output.read_bytes())
            with output.open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),36)
            sma=next(float(r['Prediction']) for r in rows if r['Method']=='LLM + SMA' and r['Round']=='2' and r['Pair']=='A-B')
            self.assertAlmostEqual(sma,.6)
            cfg=root/'config.json'
            cfg.write_text(json.dumps(dict(experiments=[dict(id='test',survey=str(survey),theme='無人島ー縛りあり',predictions='run.csv',episode='test')])) )
            out=a.analyze(cfg)
            with (out/'coverage.csv').open(encoding='utf-8-sig') as f: coverage=list(csv.DictReader(f))
            self.assertEqual(len(coverage),6*39*3)
            with (out/'metrics.csv').open(encoding='utf-8-sig') as f: metrics=list(csv.DictReader(f))
            self.assertEqual({r['method'] for r in metrics},set(a.METHODS))
            keys = [tuple(r[k] for k in ['experiment','method','run','target','pair','respondent']) for r in metrics]
            self.assertEqual(len(keys), len(set(keys)))
            overall = [r for r in metrics if r['pair']=='all' and r['respondent']=='all']
            self.assertEqual(len(overall), 12)
            with output.open('a') as f: f.write('LLM-only,test,r1,1,A-B,0.3\n')
            with self.assertRaisesRegex(ValueError,'Duplicate prediction'): a.analyze(cfg)


if __name__=='__main__': unittest.main()
