"""Run the viewer's six methods with durable per-round checkpoints."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

FIELDS = ['Method','Episode','Run','Round','Pair','Prediction']


def collect(args):
    import config
    import relation_estimator_from_txt as estimator
    from baselines.interaction_frequency import estimate_interaction_frequency_with_metadata
    from baselines.sentiment import estimate_sentiment
    from baselines.stance import estimate_stance
    from baselines.sma import SimpleMovingAverageScorer
    from log_filtering import filter_logs_by_human_count
    estimator.DEBUG = False
    logs = estimator.parse_conversation_file(args.input)
    participants = estimator.detect_participants(logs)
    if participants != ['A','B','C']: raise ValueError('This analysis expects A/B/C speakers')
    rounds = estimator.split_into_rounds(logs, participants, args.step)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = output.with_suffix('.raw.json')
    metadata = dict(input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                    step=args.step,window=args.window,episode=args.episode,run=args.run,
                    gamma=estimator.GAMMA,history=estimator.MAX_HISTORY_SESSIONS)
    saved = dict(metadata=metadata,raw=[])
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved['metadata'] != metadata: raise ValueError('Checkpoint settings differ. Use another output path.')
    if len(saved['raw']) > len(rounds): raise ValueError('Too many cached rounds')
    sma = SimpleMovingAverageScorer(3)
    ema = estimator.EMAScorer(True, estimator.GAMMA, estimator.MAX_HISTORY_SESSIONS)
    records = []
    def persist():
        temp = output.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(records)
        temp.replace(output)
    cfg = None
    for index,end in enumerate(rounds):
        window = filter_logs_by_human_count(logs[:end+1], args.window)
        frequency,_ = estimate_interaction_frequency_with_metadata(window, participants)
        methods = {'Interaction Frequency':frequency,'Stance-based':estimate_stance(window,participants),
                   'Sentiment-based':estimate_sentiment(window,participants)}
        for name,values in methods.items():
            for p,value in values.items(): records.append(dict(zip(FIELDS,[name,args.episode,args.run,index+1,'-'.join(p),value])))
    persist()  # Offline results survive any later API error.
    if args.offline_only: return
    for index,end in enumerate(rounds):
        if index < len(saved['raw']):
            raw = {tuple(k.split('-')):v for k,v in saved['raw'][index].items()}
        else:
            if cfg is None: cfg = config.get_config()
            window = filter_logs_by_human_count(logs[:end+1], args.window)
            raw = estimator.estimate_relation_once(window,participants,args.window,estimator.LLM_MODEL,cfg)
            if set(raw) != {('A','B'),('A','C'),('B','C')}: raise ValueError('Incomplete raw LLM scores')
            saved['raw'].append({'-'.join(k):v for k,v in raw.items()})
            temp = cache.with_suffix('.tmp')
            temp.write_text(json.dumps(saved,ensure_ascii=False,indent=2)); temp.replace(cache)
        counts = Counter(l['speaker'] for l in logs[:end+1] if l['speaker'] in participants)
        methods = {'LLM-only':raw, 'LLM + SMA':{p:sma.update(p,v) for p,v in raw.items()},
                   'MAVeRD':{p:ema.update(p,v,counts) for p,v in raw.items()}}
        ema.finalize_round(counts)
        for name,values in methods.items():
            for p,value in values.items(): records.append(dict(zip(FIELDS,[name,args.episode,args.run,index+1,'-'.join(p),value])))
        persist()
        print(f'round {index+1}/{len(rounds)} saved',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--episode',required=True)
    p.add_argument('--run',default='run1')
    p.add_argument('--step',type=int,default=3)
    p.add_argument('--window',type=int,default=6)
    p.add_argument('--offline-only',action='store_true')
    args=p.parse_args()
    if args.step<1 or args.window<1: p.error('step/window must be positive')
    collect(args)
