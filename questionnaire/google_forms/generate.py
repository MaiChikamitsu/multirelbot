"""Bundle registered text questionnaires into a paste-ready Google Apps Script."""
import argparse
import json
from pathlib import Path

ISLAND = '今回は15分間、次のテーマについて3人に話し合ってもらいました。\n3人が同じ無人島で救助を待つとします。持っていける道具は、ナイフ・ライター・テントのうち1つだけです。島には飲み水があり、救助までは3日間かかるとします。どの道具を持っていくかについて話し合ってください。\n最初はディベート形式で議論し、最終的には15分間で3人としての案を1つに決めてもらいました。'

def generate(configs, output):
    experiments=[]
    for path in configs:
        cfg=json.loads(Path(path).read_text())
        if cfg.get('modality')!='text' or not cfg['turns'] or len(cfg['turns'])%3:
            raise ValueError('3発話単位の文字版設定が必要です')
        if any(t['speaker'] not in ('A','B','C') or not t['text'] for t in cfg['turns']):
            raise ValueError('原稿の話者または本文が不正です')
        experiments.append(dict(id=cfg['id'],theme=cfg['theme'],turns=cfg['turns'],description='\n\n'.join(cfg.get('theme_description',[])) or ISLAND))
    script=Path(__file__).with_name('template.gs').read_text().replace('__EXPERIMENT_DATA__',json.dumps(experiments,ensure_ascii=False))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(script)
    print(str(output)+': '+', '.join('{} {}区間'.format(e['id'],len(e['turns'])//3) for e in experiments))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,action='append',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();generate(args.config,args.output)
