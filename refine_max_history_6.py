"""
MAX_HISTORY_HUMAN=6で上位条件を再比較

grid_search_summary.csvの上位10位の中から、MAX_HISTORY_HUMAN=6の条件を抽出し、
それら9個の条件で詳細な比較を行う。

使用方法:
    python refine_max_history_6.py
"""

import os
import csv
import csv
from typing import Dict, List, Tuple
import itertools
import numpy as np
from scipy import stats

# relation_estimator_from_txt.pyから関数をインポート
from relation_estimator_from_txt import (
    EMAScorer,
    parse_conversation_file,
    detect_participants,
    split_into_rounds,
    estimate_relation_once,
    parse_scores_from_response,
)
import config

# ========== 設定 ==========
# エピソード設定（会話ファイルと人間データのペア）
EPISODES = [
    {
        "name": "Episode1",
        "conversation_file": "estimation_accuracy/conversation1.txt",
        "human_file": "estimation_accuracy/human1.csv",
    },
    {
        "name": "Episode2",
        "conversation_file": "estimation_accuracy/conversation2.txt",
        "human_file": "estimation_accuracy/human2.csv",
    },
]

INPUT_SUMMARY_FILE = "estimation_accuracy/grid_search_summary_13人時点.csv"
OUTPUT_DIR = "estimation_accuracy/max_history_6_refined"
OUTPUT_SUMMARY_FILE = "estimation_accuracy/max_history_6_summary.csv"
NUM_TRIALS = 10  # より多くの試行で精度を高める
TOP_N = 10  # 上位何件からMAX_HISTORY_HUMAN=6を抽出するか
DEBUG = True
# ==========================


def load_human_data(file_path: str) -> Dict[str, float]:
    """
    human1.csvを読み込み、{ラウンド-ペア: 人間平均値}の辞書を返す

    Args:
        file_path: human1.csvのパス

    Returns:
        {"1-A-B": -0.746, "1-A-C": -0.269, ...}
    """
    human_data = {}
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダースキップ

        for row in reader:
            if len(row) >= 5:
                label = row[0]
                parts = label.split("-")
                if len(parts) >= 4:
                    section = parts[1]
                    pair_parts = parts[2:]
                    pair_sorted = "-".join(sorted(pair_parts))
                    key = f"{section}-{pair_sorted}"
                    human_data[key] = float(row[4])

    return human_data


def calculate_mae(
    predictions: Dict[str, float], ground_truth: Dict[str, float]
) -> float:
    """MAE（平均絶対誤差）を計算"""
    errors = []
    for key in ground_truth:
        if key in predictions:
            errors.append(abs(predictions[key] - ground_truth[key]))
    return np.mean(errors) if errors else float("inf")


def calculate_pearson(
    predictions: Dict[str, float], ground_truth: Dict[str, float]
) -> float:
    """Pearson相関係数を計算"""
    pred_values = []
    true_values = []
    for key in ground_truth:
        if key in predictions:
            pred_values.append(predictions[key])
            true_values.append(ground_truth[key])

    if len(pred_values) < 2:
        return 0.0

    corr, _ = stats.pearsonr(pred_values, true_values)
    return corr


def run_estimation_with_params(
    conversation_file: str,
    llm_model: str,
    gamma: float,
    max_history_sessions: int,
    max_history_human: int,
) -> Dict[str, float]:
    """
    指定されたパラメータで関係性推定を実行

    Returns:
        {ラウンド-ペア: スコア}
    """
    # configを一時的に作成して上書き（get_config()を使う）
    _CFG = config.get_config()
    original_model = getattr(_CFG.llm, "relation_model", None)
    _CFG.llm.relation_model = llm_model

    # EMAScorerを複数（試行数分）作成
    ema_scorers = [
        EMAScorer(True, gamma, max_history_sessions) for _ in range(NUM_TRIALS)
    ]

    # 会話ファイルを読み込み
    logs = parse_conversation_file(conversation_file)
    participants = detect_participants(logs)

    # ラウンドに分割
    rounds = split_into_rounds(logs, participants)

    # 各ラウンドで推定（複数試行）
    from collections import defaultdict

    results = defaultdict(lambda: [[] for _ in range(NUM_TRIALS)])

    for round_num, end_idx in enumerate(rounds, start=1):
        round_logs = logs[: end_idx + 1]
        if DEBUG:
            print(f"  ラウンド {round_num}: {len(round_logs)}発話")

        # 各参加者の発話数をカウント
        utterance_counts = defaultdict(int)
        for log in round_logs:
            if log["speaker"] in participants:
                utterance_counts[log["speaker"]] += 1

        # NUM_TRIALS回推定（試行）
        for trial_idx in range(NUM_TRIALS):
            raw_scores = estimate_relation_once(
                round_logs, participants, max_history_human, llm_model, _CFG
            )

            ema_scorer = ema_scorers[trial_idx]
            for pair in itertools.combinations(participants, 2):
                if pair in raw_scores:
                    ema_score = ema_scorer.update(
                        pair, raw_scores[pair], utterance_counts
                    )
                    results[pair][trial_idx].append(ema_score)
                else:
                    prev_score = ema_scorer.scores.get(pair, 0.0)
                    results[pair][trial_idx].append(prev_score)

            ema_scorer.finalize_round(utterance_counts)

    # 予測辞書を生成（ラウンドごとに試行平均）
    predictions = {}
    for pair in results:
        pair_name = f"{pair[0]}-{pair[1]}"
        for round_idx in range(len(rounds)):
            trial_scores = [
                results[pair][trial_idx][round_idx] for trial_idx in range(NUM_TRIALS)
            ]
            predictions[f"{round_idx + 1}-{pair_name}"] = float(np.mean(trial_scores))

    # _CFGはローカルなので復元不要（get_config()で取得したインスタンスを使い切る）

    return predictions


def extract_top_conditions_with_max_history_6(
    summary_file: str, top_n: int = 10
) -> List[Dict]:
    """
    grid_search_summary.csvから上位top_n件の中で、MAX_HISTORY_HUMAN=6の条件を抽出

    Returns:
        [{"LLM_MODEL": "gpt-5-chat", "GAMMA": 0.6, "MAX_HISTORY_SESSIONS": 3, "MAX_HISTORY_HUMAN": 6}, ...]
    """
    conditions: List[Dict] = []
    with open(summary_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = [row for _, row in zip(range(top_n), reader)]

    for row in rows:
        try:
            if int(row.get("MAX_HISTORY_HUMAN", 0)) == 6:
                conditions.append(
                    {
                        "LLM_MODEL": row.get("LLM_MODEL"),
                        "GAMMA": float(row.get("GAMMA", 0.0)),
                        "MAX_HISTORY_SESSIONS": int(row.get("MAX_HISTORY_SESSIONS", 0)),
                        "MAX_HISTORY_HUMAN": int(row.get("MAX_HISTORY_HUMAN", 0)),
                    }
                )
        except Exception:
            continue

    return conditions


def main():
    print("=" * 80)
    print("MAX_HISTORY_HUMAN=6 条件の詳細比較")
    print("=" * 80)

    # 出力ディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 上位条件を抽出
    print(f"\n📊 {INPUT_SUMMARY_FILE} から上位{TOP_N}件を読み込み中...")
    conditions = extract_top_conditions_with_max_history_6(INPUT_SUMMARY_FILE, TOP_N)
    print(f"✅ MAX_HISTORY_HUMAN=6 の条件を {len(conditions)} 件抽出しました\n")

    if len(conditions) == 0:
        print("❌ 条件が見つかりませんでした")
        return

    # 各条件を表示
    for i, cond in enumerate(conditions, 1):
        print(
            f"{i}. LLM_MODEL={cond['LLM_MODEL']}, GAMMA={cond['GAMMA']}, "
            f"MAX_HISTORY_SESSIONS={cond['MAX_HISTORY_SESSIONS']}, "
            f"MAX_HISTORY_HUMAN={cond['MAX_HISTORY_HUMAN']}"
        )

    # サマリーファイルの準備
    summary_data = []

    # 各条件でエピソードごとに評価
    for cond_idx, condition in enumerate(conditions, 1):
        llm_model = condition["LLM_MODEL"]
        gamma = condition["GAMMA"]
        max_history_sessions = condition["MAX_HISTORY_SESSIONS"]
        max_history_human = condition["MAX_HISTORY_HUMAN"]

        print(f"\n{'='*80}")
        print(
            f"条件 {cond_idx}/{len(conditions)}: LLM={llm_model}, γ={gamma}, "
            f"MAX_HIST_SESS={max_history_sessions}, MAX_HIST_HUMAN={max_history_human}"
        )
        print(f"{'='*80}")

        episode_results = {}

        for episode in EPISODES:
            episode_name = episode["name"]
            conversation_file = episode["conversation_file"]
            human_file = episode["human_file"]

            print(f"\n📖 {episode_name} を評価中...")

            # 人間データを読み込み
            human_data = load_human_data(human_file)

            # 推定実行（内部で NUM_TRIALS 回の試行を行います）
            if DEBUG:
                print(
                    f"\n  実行: internal NUM_TRIALS={NUM_TRIALS} を用いて推定を行います"
                )

            predictions = run_estimation_with_params(
                conversation_file,
                llm_model,
                gamma,
                max_history_sessions,
                max_history_human,
            )

            # 評価
            avg_mae = calculate_mae(predictions, human_data)
            avg_pearson = calculate_pearson(predictions, human_data)
            std_mae = 0.0
            std_pearson = 0.0

            episode_results[episode_name] = {
                "MAE_avg": avg_mae,
                "MAE_std": std_mae,
                "Pearson_avg": avg_pearson,
                "Pearson_std": std_pearson,
            }

            print(f"\n  📊 {episode_name} 結果 (内部{NUM_TRIALS}回平均):")
            print(f"    MAE: {avg_mae:.4f} ± {std_mae:.4f}")
            print(f"    Pearson: {avg_pearson:.4f} ± {std_pearson:.4f}")

        # 全エピソードの平均を計算
        overall_mae_avg = np.mean(
            [episode_results[ep["name"]]["MAE_avg"] for ep in EPISODES]
        )
        overall_pearson_avg = np.mean(
            [episode_results[ep["name"]]["Pearson_avg"] for ep in EPISODES]
        )

        print(f"\n  ⭐ 全エピソード平均:")
        print(f"    MAE: {overall_mae_avg:.4f}")
        print(f"    Pearson: {overall_pearson_avg:.4f}")

        # サマリーデータに追加
        summary_row = {
            "LLM_MODEL": llm_model,
            "GAMMA": gamma,
            "MAX_HISTORY_SESSIONS": max_history_sessions,
            "MAX_HISTORY_HUMAN": max_history_human,
            "Overall_MAE_Avg": overall_mae_avg,
            "Overall_Pearson_Avg": overall_pearson_avg,
        }

        # 各エピソードの結果を追加
        for episode in EPISODES:
            episode_name = episode["name"]
            summary_row[f"{episode_name}_MAE_Avg"] = episode_results[episode_name][
                "MAE_avg"
            ]
            summary_row[f"{episode_name}_MAE_Std"] = episode_results[episode_name][
                "MAE_std"
            ]
            summary_row[f"{episode_name}_Pearson_Avg"] = episode_results[episode_name][
                "Pearson_avg"
            ]
            summary_row[f"{episode_name}_Pearson_Std"] = episode_results[episode_name][
                "Pearson_std"
            ]

        summary_data.append(summary_row)

    # サマリーをCSVに保存（Overall_MAE_Avgでソート）
    print(f"\n{'='*80}")
    print(f"📄 サマリーを {OUTPUT_SUMMARY_FILE} に保存中...")

    # summary_dataをOverall_MAE_AvgでソートしてCSVに保存
    summary_data_sorted = sorted(summary_data, key=lambda x: x["Overall_MAE_Avg"])
    fieldnames = [
        "Rank",
        "LLM_MODEL",
        "GAMMA",
        "MAX_HISTORY_SESSIONS",
        "MAX_HISTORY_HUMAN",
    ]
    for episode in EPISODES:
        fieldnames.extend(
            [
                f"{episode['name']}_MAE_Avg",
                f"{episode['name']}_MAE_Std",
                f"{episode['name']}_Pearson_Avg",
                f"{episode['name']}_Pearson_Std",
            ]
        )
    fieldnames.extend(["Overall_MAE_Avg", "Overall_Pearson_Avg"])

    with open(OUTPUT_SUMMARY_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(summary_data_sorted, 1):
            out = {k: v for k, v in row.items()}
            out["Rank"] = rank
            writer.writerow(out)

    print("✅ 保存完了")

    # ランキングを表示
    print(f"\n{'='*80}")
    print("🏆 最終ランキング (Overall MAE Avg)")
    print(f"{'='*80}")

    for rank, row in enumerate(summary_data_sorted, 1):
        print(
            f"{rank}. LLM={row['LLM_MODEL']}, γ={row['GAMMA']:.2f}, "
            f"MAX_HIST_SESS={int(row['MAX_HISTORY_SESSIONS'])}, "
            f"MAE={row['Overall_MAE_Avg']:.4f}, Pearson={row['Overall_Pearson_Avg']:.4f}"
        )

    print(f"\n✅ 完了！詳細結果は {OUTPUT_SUMMARY_FILE} を参照してください")


if __name__ == "__main__":
    main()
