"""
グリッドサーチスクリプト - 最適なパラメータを探索

LLM_MODEL、GAMMA、MAX_HISTORY_SESSIONS、MAX_HISTORY_HUMANの全組み合わせで
関係性推定を実行し、human1.csvとの近さを評価する。

使用方法:
    python parameter_grid_search.py
"""

import os
import csv
import itertools
from collections import defaultdict
from typing import Dict, Tuple, List
import numpy as np
from scipy import stats

# relation_estimator_from_txt.pyから関数をインポート
from relation_estimator_from_txt import (
    EMAScorer,
    parse_conversation_file,
    detect_participants,
    split_into_rounds,
    estimate_relation_once,
    parse_scores_from_response
)
import config

# ========== 設定 ==========
# エピソード設定（会話ファイルと人間データのペア）
EPISODES = [
    {
        "name": "Episode1",
        "conversation_file": "estimation_accuracy/conversation1.txt",
        "human_file": "estimation_accuracy/human1.csv"
    },
    {
        "name": "Episode2",
        "conversation_file": "estimation_accuracy/conversation2.txt",
        "human_file": "estimation_accuracy/human2.csv"
    }
]

OUTPUT_DIR = "estimation_accuracy/grid_search_results"
SUMMARY_FILE = "estimation_accuracy/grid_search_summary.csv"
NUM_TRIALS = 5  # 各パラメータセットでの試行回数
DEBUG = True

# グリッドパラメータ
LLM_MODELS = ["gpt-4.1", "gpt-5-chat"]
GAMMAS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
MAX_HISTORY_SESSIONS_LIST = [2, 3]
MAX_HISTORY_HUMAN_LIST = [6]
# ==========================


def load_human_data(file_path: str) -> Dict[str, float]:
    """
    human1.csvを読み込み、{ラウンド-ペア: 人間平均値}の辞書を返す

    Args:
        file_path: human1.csvのパス

    Returns:
        {"1-A-B": -0.746, "1-A-C": -0.269, ...}
        注：ペア名はアルファベット順にソート（C-A → A-C）
    """
    human_data = {}
    # UTF-8の不完全な文字を処理するため、errors='replace'を使用
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        # ヘッダー行をスキップ
        next(reader)

        for row in reader:
            if len(row) >= 5:
                # row[0]: 識別ラベル (例: "1-1-A-B" または "1-1-C-A")
                # row[4]: 人の平均
                label = row[0]
                parts = label.split('-')
                if len(parts) >= 4:
                    section = parts[1]  # セクション番号（ラウンド番号）
                    pair_parts = parts[2:]  # ペアの要素 (例: ['C', 'A'])
                    # アルファベット順にソートして正規化（C-A → A-C）
                    pair_sorted = '-'.join(sorted(pair_parts))
                    key = f"{section}-{pair_sorted}"
                    human_data[key] = float(row[4])

    return human_data


def calculate_mae(predictions: Dict[str, float], ground_truth: Dict[str, float]) -> float:
    """
    MAE（平均絶対誤差）を計算

    Args:
        predictions: {ラウンド-ペア: 予測値}
        ground_truth: {ラウンド-ペア: 正解値}

    Returns:
        MAE値
    """
    errors = []
    for key in ground_truth:
        if key in predictions:
            errors.append(abs(predictions[key] - ground_truth[key]))

    if not errors:
        return float('inf')

    return np.mean(errors)


def calculate_pearson(predictions: Dict[str, float], ground_truth: Dict[str, float]) -> float:
    """
    ピアソン相関係数を計算

    Args:
        predictions: {ラウンド-ペア: 予測値}
        ground_truth: {ラウンド-ペア: 正解値}

    Returns:
        ピアソン相関係数
    """
    pred_values = []
    true_values = []

    for key in ground_truth:
        if key in predictions:
            pred_values.append(predictions[key])
            true_values.append(ground_truth[key])

    if len(pred_values) < 2:
        return 0.0

    correlation, _ = stats.pearsonr(pred_values, true_values)
    return correlation


def run_single_parameter_set(
    llm_model: str,
    gamma: float,
    max_history_sessions: int,
    max_history_human: int,
    logs: List[Dict],
    participants: List[str],
    round_end_indices: List[int],
    _CFG
) -> Dict[Tuple[str, str], List[float]]:
    """
    単一のパラメータセットで関係性推定を実行

    Returns:
        {(A, B): [trial1_avg, trial2_avg, ...], ...}
    """
    # 各試行用のEMAスコアラーを初期化
    ema_scorers = [EMAScorer(True, gamma, max_history_sessions) for _ in range(NUM_TRIALS)]

    # 結果を格納: {pair: [[trial1_round1, trial1_round2, ...], [trial2_round1, ...], ...]}
    results = defaultdict(lambda: [[] for _ in range(NUM_TRIALS)])

    # 各ラウンドで推定を実行
    for round_idx, end_idx in enumerate(round_end_indices, 1):
        # このラウンドまでのログ
        round_logs = logs[:end_idx + 1]

        # 各参加者の発話数をカウント
        utterance_counts = defaultdict(int)
        for log in round_logs:
            if log['speaker'] in participants:
                utterance_counts[log['speaker']] += 1

        # NUM_TRIALS回推定を実行
        for trial_idx in range(NUM_TRIALS):
            # 関係性推定
            raw_scores = estimate_relation_once(
                round_logs, participants, max_history_human, llm_model, _CFG
            )

            # EMA適用
            ema_scorer = ema_scorers[trial_idx]
            for pair in itertools.combinations(participants, 2):
                if pair in raw_scores:
                    ema_score = ema_scorer.update(pair, raw_scores[pair], utterance_counts)
                    results[pair][trial_idx].append(ema_score)
                else:
                    # スコアが得られなかった場合は前回値を使用
                    prev_score = ema_scorer.scores.get(pair, 0.0)
                    results[pair][trial_idx].append(prev_score)

            # ラウンド終了時に発話数を記録
            ema_scorer.finalize_round(utterance_counts)

    # 各ペア・各試行の平均を計算
    pair_averages = {}
    for pair in results:
        trial_averages = []
        for trial_idx in range(NUM_TRIALS):
            # 各試行の全ラウンドの平均
            trial_avg = np.mean(results[pair][trial_idx])
            trial_averages.append(trial_avg)
        pair_averages[pair] = trial_averages

    return results, pair_averages


def save_individual_results(
    episode_results: List[Dict],  # [{"name": "Episode1", "results": {...}, "round_end_indices": [...], "mae": 0.123, "pearson": 0.85}, ...]
    llm_model: str,
    gamma: float,
    max_history_sessions: int,
    max_history_human: int,
    output_dir: str
):
    """
    個別の結果CSVを保存（複数エピソード対応）
    """
    # ファイル名生成
    model_short = llm_model.replace("gpt-", "gpt").replace(".", "")
    filename = f"{model_short}_g{gamma:.2f}_hs{max_history_sessions}_hh{max_history_human}.csv"
    filepath = os.path.join(output_dir, filename)

    # 平均MAE・Pearsonを計算
    avg_mae = np.mean([ep["mae"] for ep in episode_results])
    avg_pearson = np.mean([ep["pearson"] for ep in episode_results])

    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # メタデータ行
        writer.writerow([f"# Model: {llm_model}, Gamma: {gamma}, Max_History_Sessions: {max_history_sessions}, Max_History_Human: {max_history_human}"])

        # 各エピソードのMAE・Pearson
        for ep in episode_results:
            writer.writerow([f"# {ep['name']} - MAE: {ep['mae']:.4f}, Pearson: {ep['pearson']:.4f}"])

        # 平均
        writer.writerow([f"# Average - MAE: {avg_mae:.4f}, Pearson: {avg_pearson:.4f}"])
        writer.writerow([])  # 空行

        # 各エピソードの結果を出力
        for ep in episode_results:
            writer.writerow([f"=== {ep['name']} ==="])
            writer.writerow(['ペア-ラウンド', '試行1', '試行2', '試行3', '試行4', '試行5', '平均', 'エピソード'])

            results = ep["results"]
            round_end_indices = ep["round_end_indices"]

            # ペアごとにまとめて出力
            for pair in sorted(results.keys()):
                pair_name = f"{pair[0]}-{pair[1]}"

                # 各ラウンドの結果を出力
                for round_idx in range(len(round_end_indices)):
                    row_label = f"{round_idx + 1}-{pair_name}"

                    # 5試行分のスコア
                    trial_scores = [
                        round(results[pair][trial_idx][round_idx], 1)
                        for trial_idx in range(NUM_TRIALS)
                    ]

                    # 平均（小数第1位に四捨五入）
                    avg = round(sum(trial_scores) / len(trial_scores), 1)

                    writer.writerow([row_label] + trial_scores + [avg] + [ep['name']])

            writer.writerow([])  # エピソード間に空行


def format_model_name(llm_model: str) -> str:
    """モデル名を短縮形式に変換"""
    return llm_model.replace("gpt-", "gpt").replace(".", "")


def main():
    """メイン処理"""
    # 設定読み込み
    _CFG = config.get_config()

    print("=" * 80)
    print("🔍 グリッドサーチ開始（複数エピソード対応）")
    print("=" * 80)

    # 出力ディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 各エピソードのデータを読み込み
    episode_data = []
    for episode in EPISODES:
        print(f"\n📁 {episode['name']} 読み込み中...")

        # 人間データ読み込み
        human_data = load_human_data(episode['human_file'])
        print(f"  📊 人間データ: {len(human_data)} データポイント")

        # 会話ログ読み込み
        logs = parse_conversation_file(episode['conversation_file'])
        print(f"  📄 会話ログ: {len(logs)} 発話")

        # 参加者を検出
        participants = detect_participants(logs)
        print(f"  👥 参加者: {participants}")

        # ラウンド分割
        round_end_indices = split_into_rounds(logs, participants)
        print(f"  🔄 ラウンド数: {len(round_end_indices)}")

        episode_data.append({
            "name": episode["name"],
            "human_data": human_data,
            "logs": logs,
            "participants": participants,
            "round_end_indices": round_end_indices
        })

    # グリッドパラメータの全組み合わせを生成
    param_grid = list(itertools.product(
        LLM_MODELS, GAMMAS, MAX_HISTORY_SESSIONS_LIST, MAX_HISTORY_HUMAN_LIST
    ))
    total_sets = len(param_grid)
    print(f"\n🎯 パラメータセット数: {total_sets}")
    print(f"   エピソード数: {len(EPISODES)}")
    print(f"   各セット: {NUM_TRIALS}試行 × エピソードあたり約{episode_data[0]['round_end_indices'].__len__()}ラウンド推定")
    print()

    # サマリーデータを格納
    summary_data = []

    # グリッドサーチ実行
    for idx, (llm_model, gamma, max_history_sessions, max_history_human) in enumerate(param_grid, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{total_sets}] パラメータセット実行中...")
        print(f"  Model: {llm_model}, γ={gamma}, Sessions={max_history_sessions}, Human={max_history_human}")
        print(f"{'='*80}")

        try:
            # 各エピソードで実行
            ep_results = []

            for ep in episode_data:
                print(f"\n  📍 {ep['name']} 実行中...")

                # 推定実行
                results, pair_averages = run_single_parameter_set(
                    llm_model, gamma, max_history_sessions, max_history_human,
                    ep['logs'], ep['participants'], ep['round_end_indices'], _CFG
                )

                # 予測値を{ラウンド-ペア: 平均値}形式に変換
                predictions = {}
                for pair in results:
                    pair_name = f"{pair[0]}-{pair[1]}"
                    for round_idx in range(len(ep['round_end_indices'])):
                        key = f"{round_idx + 1}-{pair_name}"
                        # 5試行の平均を計算
                        trial_scores = [results[pair][trial_idx][round_idx] for trial_idx in range(NUM_TRIALS)]
                        predictions[key] = np.mean(trial_scores)

                # 評価指標を計算
                mae = calculate_mae(predictions, ep['human_data'])
                pearson = calculate_pearson(predictions, ep['human_data'])

                print(f"    ✅ {ep['name']}: MAE={mae:.4f}, Pearson={pearson:.4f}")

                # エピソード結果を保存
                ep_results.append({
                    "name": ep['name'],
                    "results": results,
                    "round_end_indices": ep['round_end_indices'],
                    "mae": mae,
                    "pearson": pearson
                })

            # 個別結果CSVを保存（全エピソード分）
            save_individual_results(
                ep_results, llm_model, gamma, max_history_sessions, max_history_human, OUTPUT_DIR
            )

            # 平均MAE・Pearsonを計算
            avg_mae = np.mean([ep["mae"] for ep in ep_results])
            avg_pearson = np.mean([ep["pearson"] for ep in ep_results])

            print(f"\n  ✅ 全エピソード完了: 平均MAE={avg_mae:.4f}, 平均Pearson={avg_pearson:.4f}")

            # サマリーデータに追加
            summary_entry = {
                'LLM_MODEL': llm_model,
                'GAMMA': gamma,
                'MAX_HISTORY_SESSIONS': max_history_sessions,
                'MAX_HISTORY_HUMAN': max_history_human,
                'MAE_Avg': avg_mae,
                'Pearson_Avg': avg_pearson
            }
            # 各エピソードのMAE・Pearsonも追加
            for ep in ep_results:
                summary_entry[f'MAE_{ep["name"]}'] = ep['mae']
                summary_entry[f'Pearson_{ep["name"]}'] = ep['pearson']

            summary_data.append(summary_entry)

        except Exception as e:
            print(f"\n  ❌ エラー: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 平均MAEでソート
    summary_data.sort(key=lambda x: x['MAE_Avg'])

    # ランクを追加
    for rank, item in enumerate(summary_data, 1):
        item['Rank'] = rank

    # サマリーCSVを保存
    print(f"\n{'='*80}")
    print(f"💾 サマリーCSV保存: {SUMMARY_FILE}")
    print(f"{'='*80}")

    with open(SUMMARY_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        # フィールド名を動的に生成
        fieldnames = ['Rank', 'LLM_MODEL', 'GAMMA', 'MAX_HISTORY_SESSIONS', 'MAX_HISTORY_HUMAN']
        # 各エピソードのMAE・Pearson
        for ep in EPISODES:
            fieldnames.append(f'MAE_{ep["name"]}')
            fieldnames.append(f'Pearson_{ep["name"]}')
        # 平均
        fieldnames.extend(['MAE_Avg', 'Pearson_Avg'])

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary_data:
            writer.writerow(item)

    # Top 5をコンソール出力
    print(f"\n{'='*80}")
    print("🏆 Top 5 最適パラメータ（平均MAE基準）")
    print(f"{'='*80}")

    for rank, item in enumerate(summary_data[:5], 1):
        print(f"\n{rank}位: {item['LLM_MODEL']}, γ={item['GAMMA']}, sessions={item['MAX_HISTORY_SESSIONS']}, human={item['MAX_HISTORY_HUMAN']}")
        print(f"     平均MAE={item['MAE_Avg']:.4f}, 平均Pearson={item['Pearson_Avg']:.4f}")
        for ep in EPISODES:
            ep_name = ep['name']
            mae_key = f'MAE_{ep_name}'
            pearson_key = f'Pearson_{ep_name}'
            print(f"       {ep_name}: MAE={item[mae_key]:.4f}, Pearson={item[pearson_key]:.4f}")

    print(f"\n{'='*80}")
    print("✅ グリッドサーチ完了")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
