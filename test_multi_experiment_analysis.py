import csv
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from unittest.mock import patch

from analyze_relations import read_survey, corr
from relation_trends import calculate_trends
from multi_experiment_analysis import build_overall, aggregate_correlations

METHODS=['Interaction Frequency','Stance-based','Sentiment-based','LLM-only','LLM + SMA','MAVeRD']
PAIRS=['A-B','A-C','B-C']

class MultiExperimentTest(unittest.TestCase):
    def test_pooled_and_equal_conversation_correlations(self):
        groups={'a':[dict(prediction=0,human=10),dict(prediction=1,human=11)],
                'b':[dict(prediction=10,human=0),dict(prediction=11,human=1)]}
        pooled=aggregate_correlations(groups,'pooled_points')
        macro=aggregate_correlations(groups,'equal_experiments')
        self.assertLess(pooled['pearson'],0)
        self.assertEqual(macro['pearson'],1)
        self.assertEqual(macro['spearman'],1)
        self.assertEqual(macro['pearson_valid_experiments'],2)
        constant={'a':[dict(prediction=0,human=0),dict(prediction=0,human=1)]}
        values=aggregate_correlations(constant,'equal_experiments')
        self.assertIsNone(values['pearson'])
        self.assertEqual(values['pearson_valid_experiments'],0)

    def test_same_theme_two_groups_and_real_latest(self):
        source=Path('/Users/kinmai/Downloads/【修正版】（直後）予備実験アンケート （回答） (1).xlsx')
        if not source.exists():self.skipTest('Local questionnaire unavailable')
        self.assertEqual(len(read_survey(source,'AIー縛りなし',['4','5','6'],{'A':'4','B':'5','C':'6'})),207)
        with tempfile.TemporaryDirectory() as td:
            fixture=Path(td)/'six_same_theme.xlsx'
            ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            with ZipFile(source) as original,ZipFile(fixture,'w') as updated:
                for item in original.infolist():
                    data=original.read(item.filename)
                    if item.filename=='xl/worksheets/sheet1.xml':
                        tree=ET.fromstring(data)
                        cells={c.attrib['r']:c for c in tree.findall('.//m:c',ns)}
                        theme_index=cells['D2'].find('m:v',ns).text
                        for r in ['5','6','7']:cells['D'+r].find('m:v',ns).text=theme_index
                        data=ET.tostring(tree)
                    updated.writestr(item,data)
            first=read_survey(fixture,'無人島ー縛りあり',['1','2','3'])
            second=read_survey(fixture,'無人島ー縛りあり',['4','5','6'])
            self.assertEqual(len(first),369);self.assertEqual(len(second),207)
            self.assertEqual({r['respondent'] for r in second},{'4','5','6'})
            with self.assertRaisesRegex(ValueError,'select exactly'):read_survey(fixture,'無人島ー縛りあり')
            with self.assertRaisesRegex(ValueError,'Role mapping'):read_survey(fixture,'無人島ー縛りあり',['4','5','6'],{'A':'5','B':'4','C':'6'})

    def test_unequal_lengths_and_boundaries_and_run_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);experiments=[];humans=[];predictions=[];comparisons=[]
            for eid,ys,system in [('one',[0,0],.2),('two',[0,.5,0],.6)]:
                folder=root/eid;folder.mkdir()
                (folder/'metadata.json').write_text(json.dumps(dict(model='test',step=3,window_size=6,num_trials=10,gamma=.8,max_history_sessions=3,sma_window=3,relation_temperature=.3)))
                p=folder/'predictions.csv'
                p.write_text('Episode,Run\n'+eid+',chosen\n')
                experiments.append(dict(id=eid,theme='theme',episode=eid,predictions=str(p),aggregate_run='chosen',condition_id='same'))
                for i,y in enumerate(ys,1):
                    for pair in PAIRS:
                        for person in ['1','2','3']:humans.append(dict(experiment=eid,index=i,pair=pair,respondent=person))
                        for method in METHODS:
                            base=dict(experiment=eid,run='chosen',method=method,index=i,pair=pair,prediction=system)
                            predictions.append(base)
                            for target,respondent in [('mean','all'),('individual','1'),('individual','2'),('individual','3')]:
                                comparisons.append(dict(base,target=target,respondent=respondent,human=y))
                            # Old runs must not be counted again.
                            comparisons.append(dict(base,run='old',target='mean',respondent='all',human=-1))
            _,changes=calculate_trends(comparisons,corr)
            rows,status=build_overall({'experiments':experiments},root,humans,predictions,comparisons,changes)
            self.assertEqual([s['status'] for s in status],['included','included'])
            select=lambda mode:next(r for r in rows if r['method']=='MAVeRD' and r['target']=='mean' and r['pair']=='all' and r['aggregation']==mode)
            pooled=select('pooled_points');macro=select('equal_experiments')
            self.assertEqual(pooled['n_points'],15)
            self.assertEqual(pooled['n_changes'],9) # (2-1 + 3-1)*3, no cross-session transitions
            self.assertAlmostEqual(pooled['mae'],.34)
            self.assertAlmostEqual(macro['mae'],(.2+1.3/3)/2)
            self.assertAlmostEqual(pooled['change_mae'],1/3)
            self.assertAlmostEqual(macro['change_mae'],.25)
            meta=root/'two/metadata.json';content=json.loads(meta.read_text());content['gamma']=.6;meta.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError,'settings differ'):
                build_overall({'experiments':experiments},root,humans,predictions,comparisons,changes)
            experiments[1]['condition_id']='different'
            rows,_=build_overall({'experiments':experiments},root,humans,predictions,comparisons,changes)
            self.assertEqual({r['n_experiments'] for r in rows},{1})

if __name__=='__main__':unittest.main()
