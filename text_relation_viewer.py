"""Replay a text conversation with live relation graphs: python3 text_relation_viewer.py."""
import argparse
import copy
import threading
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

from flask import Flask, jsonify, render_template
import config
import relation_estimator_from_txt as estimator
from baselines.interaction_frequency import estimate_interaction_frequency_with_metadata
from baselines.sentiment import estimate_sentiment
from baselines.stance import estimate_stance
from baselines.sma import SimpleMovingAverageScorer
from log_filtering import filter_logs_by_human_count

METHODS = ['Interaction Frequency', 'Stance-based', 'Sentiment-based', 'LLM-only', 'LLM + SMA', 'MAVeRD']


def create_app(input_file, interval=1.0, step=3, window_size=6):
    logs = estimator.parse_conversation_file(input_file)
    participants = estimator.detect_participants(logs)
    if len(participants) < 2:
        raise ValueError('話者が2人以上いる [話者] 発言 形式のテキストが必要です。')
    cfg = config.get_config()
    pairs = list(combinations(participants, 2))
    sma = SimpleMovingAverageScorer(3)
    scorer = estimator.EMAScorer(True, estimator.GAMMA, estimator.MAX_HISTORY_SESSIONS)
    lock = threading.Lock()
    running = threading.Event()
    state = dict(status='開始待ち', participants=participants, conversation=[], history=[],
                 methods=METHODS, method_scores={}, interaction_intensity={}, scores={}, total=len(logs), position=0, step=step, window_size=window_size, input=str(input_file), done=False)
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
                    raw = estimator.estimate_relation_once(window, participants,
                        window_size, estimator.LLM_MODEL, cfg)
                    if any(pair not in raw for pair in pairs):
                        raise ValueError('推定結果に必要なペアのスコアがありません。')
                    counts = Counter(item['speaker'] for item in logs[:index + 1] if item['speaker'] in participants)
                    scores = {'–'.join(pair): scorer.update(pair, raw[pair], counts) for pair in pairs}
                    scorer.finalize_round(counts)
                    frequency, metadata = estimate_interaction_frequency_with_metadata(window, participants)
                    method_results = {
                        'Interaction Frequency': frequency,
                        'Stance-based': estimate_stance(window, participants),
                        'Sentiment-based': estimate_sentiment(window, participants),
                        'LLM-only': raw,
                        'LLM + SMA': {pair: sma.update(pair, raw[pair]) for pair in pairs},
                    }
                    method_scores = {name: {'–'.join(pair): values[pair] for pair in pairs}
                                     for name, values in method_results.items()}
                    method_scores['MAVeRD'] = scores
                    with lock:
                        state['scores'] = scores
                        state['method_scores'] = method_scores
                        state['interaction_intensity'] = {'–'.join(pair): metadata[pair]['interaction_intensity'] for pair in pairs}
                        state['history'].append(dict(turn=count, log_index=index, scores=scores, methods=method_scores))
                        state['status'] = '関係図を更新しました'
                except Exception as exc:
                    with lock:
                        state['status'] = f'推定失敗 ({type(exc).__name__})。ターミナルを確認してください。'
                        state['done'] = True
                    # SDK errors can contain request information; show only the error type.
                    print(f'推定失敗: {type(exc).__name__}', flush=True)
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
    parser.add_argument('--input', default=estimator.INPUT_FILE)
    parser.add_argument('--interval', type=float, default=1.0, help='発話間の待ち時間（秒）')
    parser.add_argument('--step', type=int, default=3, help='何人間発話ごとに推定するか')
    parser.add_argument('--window-size', type=int, default=6, help='推定に使う直近の人間発話数')
    parser.add_argument('--port', type=int, default=8889)
    args = parser.parse_args()
    if args.step < 1 or args.window_size < 1 or args.interval < 0:
        parser.error('--step と --window-size は1以上、--interval は0以上にしてください。')
    print('各推定はAzure APIを1回呼びます。開始ボタンを押すまでAPIは呼びません。')
    create_app(Path(args.input), args.interval, args.step, args.window_size).run(host='127.0.0.1', port=args.port, threaded=True)
