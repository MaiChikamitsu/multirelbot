"""Replay a text conversation with live relation graphs: python3 text_relation_viewer.py."""
import argparse
import copy
import csv
import hashlib
import json
import math
from statistics import mean
from datetime import datetime
import threading
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

from flask import Flask, jsonify, render_template
import config
import relation_estimator_from_txt as estimator
from azure_clients import format_azure_error
from baselines.interaction_frequency import estimate_interaction_frequency_with_metadata
from baselines.sentiment import estimate_sentiment
from baselines.stance import estimate_stance
from baselines.sma import SimpleMovingAverageScorer
from log_filtering import filter_logs_by_human_count

METHODS = ['Interaction Frequency', 'Stance-based', 'Sentiment-based', 'LLM-only', 'LLM + SMA', 'MAVeRD']


def create_app(input_file, interval=1.0, step=3, window_size=6, output_dir=None, num_trials=1):
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1")
    logs = estimator.parse_conversation_file(input_file)
    participants = estimator.detect_participants(logs)
    if len(participants) < 2:
        raise ValueError('話者が2人以上いる [話者] 発言 形式のテキストが必要です。')
    cfg = config.get_config()
    output = Path(output_dir or 'estimation_accuracy/viewer_runs') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        input=str(Path(input_file).resolve()),
        input_sha256=hashlib.sha256(Path(input_file).read_bytes()).hexdigest(),
        step=step, window_size=window_size, sma_window=3, num_trials=num_trials,
        aggregation="mean across independent trials after per-trial smoothing",
        gamma=estimator.GAMMA, max_history_sessions=estimator.MAX_HISTORY_SESSIONS,
        model=estimator.LLM_MODEL or cfg.llm.relation_model or cfg.llm.azure_model,
        relation_temperature=cfg.llm.relation_temperature, score_range=[-1, 1],
        index_definition='Round=turn/step; log_index is zero-based parsed conversation position; no recording time is available',
    )
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'推定結果の保存先: {output.resolve()}', flush=True)
    pairs = list(combinations(participants, 2))
    smas = [SimpleMovingAverageScorer(3) for _ in range(num_trials)]
    scorers = [estimator.EMAScorer(True, estimator.GAMMA, estimator.MAX_HISTORY_SESSIONS) for _ in range(num_trials)]
    lock = threading.Lock()
    running = threading.Event()
    state = dict(status='開始待ち', participants=participants, conversation=[], history=[],
                 methods=METHODS, method_scores={}, interaction_intensity={}, scores={}, total=len(logs), position=0, step=step, window_size=window_size, input=str(input_file), output=str(output.resolve()), num_trials=num_trials, done=False)
    app = Flask(__name__)

    def replay():
        count = 0
        for index, log in enumerate(logs):
            running.wait()
            with lock:
                state['conversation'].append(log)
                state['position'] = index + 1
                state['status'] = 'テキスト再生中'
            if log['speaker'] in participants:
                count += 1
            if log['speaker'] in participants and count % step == 0:
                with lock:
                    state['status'] = '関係値を推定中…'
                try:
                    window = filter_logs_by_human_count(logs[:index + 1], window_size)
                    counts = Counter(item['speaker'] for item in logs[:index + 1] if item['speaker'] in participants)
                    trial_values = {'LLM-only': [], 'LLM + SMA': [], 'MAVeRD': []}
                    for trial_index, (sma, scorer) in enumerate(zip(smas, scorers)):
                        with lock:
                            state['status'] = f'関係値を推定中…（試行 {trial_index + 1}/{num_trials}）'
                        raw = estimator.estimate_relation_once(window, participants,
                            window_size, estimator.LLM_MODEL, cfg)
                        if any(pair not in raw or not math.isfinite(raw[pair]) or not -1 <= raw[pair] <= 1 for pair in pairs):
                            raise ValueError('推定結果に必要なペアの有効なスコアがありません。')
                        # Each trial retains its own temporal history across rounds.
                        trial_methods = {
                            'LLM-only': raw,
                            'LLM + SMA': {pair: sma.update(pair, raw[pair]) for pair in pairs},
                            'MAVeRD': {pair: scorer.update(pair, raw[pair], counts) for pair in pairs},
                        }
                        scorer.finalize_round(counts)
                        trial_file = output / 'trial_predictions.csv'
                        new_trial_file = not trial_file.exists()
                        with trial_file.open('a', encoding='utf-8', newline='') as f:
                            writer = csv.writer(f)
                            if new_trial_file:
                                writer.writerow(['Method', 'Episode', 'Run', 'Round', 'Trial', 'Pair', 'Prediction'])
                            for name, values in trial_methods.items():
                                trial_values[name].append(values)
                                for pair in pairs:
                                    writer.writerow([name, Path(input_file).stem, output.name,
                                                     count // step, trial_index + 1, '–'.join(pair), values[pair]])
                    frequency, metadata = estimate_interaction_frequency_with_metadata(window, participants)
                    method_results = {
                        'Interaction Frequency': frequency,
                        'Stance-based': estimate_stance(window, participants),
                        'Sentiment-based': estimate_sentiment(window, participants),
                    }
                    for name, trials in trial_values.items():
                        method_results[name] = {pair: mean(values[pair] for values in trials) for pair in pairs}
                    method_scores = {name: {'–'.join(pair): values[pair] for pair in pairs}
                                     for name, values in method_results.items()}
                    scores = method_scores['MAVeRD']
                    prediction_file = output / "predictions.csv"
                    new_file = not prediction_file.exists()
                    estimated_at = datetime.now().astimezone().isoformat()
                    with prediction_file.open("a", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        if new_file: writer.writerow(["Method", "Episode", "Run", "Round", "Pair", "Prediction", "Human_Turn", "Log_Index", "Window_Start_Log_Index", "Estimated_At", "Interaction_Intensity", "Num_Trials"])
                        for name, values in method_scores.items():
                            for label, value in values.items():
                                writer.writerow([name, Path(input_file).stem, output.name, count // step, label, value, count, index, index + 1 - len(window), estimated_at, metadata[tuple(label.split("–"))]["interaction_intensity"] if name == "Interaction Frequency" else "", num_trials if name in trial_values else 1])
                    with lock:
                        state['scores'] = scores
                        state['method_scores'] = method_scores
                        state['interaction_intensity'] = {'–'.join(pair): metadata[pair]['interaction_intensity'] for pair in pairs}
                        state['history'].append(dict(turn=count, log_index=index, scores=scores, methods=method_scores))
                        state['status'] = '関係図を更新しました'
                except Exception as exc:
                    detail = format_azure_error(exc, cfg.llm)
                    with lock:
                        state['status'] = f'推定失敗（発話 {index + 1}）: {detail}'
                        state['done'] = True
                    print(f'推定失敗（発話 {index + 1}）: {detail}', flush=True)
                    return
            # A paused replay resumes at the next utterance. An in-flight API call finishes.
            time.sleep(interval)
        with lock:
            state['done'] = True
            state['status'] = '完了（最後の端数発話は推定対象外）' if count % step else '完了'

    thread = threading.Thread(target=replay, daemon=True)
    thread.start()

    @app.get('/')
    def index():
        return render_template('text_relation_viewer.html')

    @app.get('/state')
    def snapshot():
        with lock:
            return jsonify(copy.deepcopy(state))

    @app.post('/control/<action>')
    def control(action):
        if action not in ('start', 'pause'):
            return jsonify(error='unknown action'), 400
        with lock:
            if state['done']:
                return jsonify(error='再生は終了しています。再生し直すにはサーバを再起動してください。'), 409
            if action == 'start':
                running.set()
                state['status'] = 'テキスト再生中'
            else:
                running.clear()
                state['status'] = '一時停止（実行中の推定は完了します）'
        return jsonify(ok=True)

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='テキストを順に再生し、関係図と関係値の推移を表示します。')
    parser.add_argument('--input', required=True, type=Path, help='再生する会話テキストのパス')
    parser.add_argument('--interval', type=float, default=1.0, help='発話間の待ち時間（秒）')
    parser.add_argument('--step', type=int, default=3, help='何人間発話ごとに推定するか')
    parser.add_argument('--window-size', type=int, default=6, help='推定に使う直近の人間発話数')
    parser.add_argument('--output-dir', type=Path, default=None, help='実行ごとのCSV保存先の親フォルダ')
    parser.add_argument('--num-trials', type=int, default=1, help='各時点のLLM試行数。各試行で平滑化した後に平均（論文条件は10）')
    parser.add_argument('--port', type=int, default=8889)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'入力ファイルが見つかりません: {args.input}')
    if args.step < 1 or args.window_size < 1 or args.interval < 0 or args.num_trials < 1:
        parser.error('--step・--window-size・--num-trials は1以上、--interval は0以上にしてください。')
    print(f'各推定時点でAzure APIを{args.num_trials}回呼び、試行別に平滑化して平均します。開始ボタンを押すまでAPIは呼びません。')
    create_app(args.input, args.interval, args.step, args.window_size, args.output_dir, args.num_trials).run(host='127.0.0.1', port=args.port, threaded=True)
