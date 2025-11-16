"""
会話シミュレーション実行スクリプト

使用方法:
    python run_simulation.py [--num-episodes N] [--output OUTPUT_DIR]

例:
    python run_simulation.py --num-episodes 10
    python run_simulation.py --num-episodes 20 --output results/sim_20250113
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import config
from simulation_environment import SimulationEnvironment


def calculate_statistics(all_stats: List[Dict]) -> Dict:
    """
    全エピソードの統計を計算

    Args:
        all_stats: 各エピソードの統計リスト

    Returns:
        集計統計
    """
    if not all_stats:
        return {}

    # 早期終了エピソード（2連続完璧で終了）
    early_termination_episodes = [s for s in all_stats if s.get("early_termination", False)]
    perfect_completion_rate = len(early_termination_episodes) / len(all_stats) if all_stats else 0.0

    # 一度でも安定を達成したエピソード
    stability_achieved_episodes = [
        s for s in all_stats if s.get("first_stable_utterance") is not None
    ]
    stability_achieved_rate = (
        len(stability_achieved_episodes) / len(all_stats) if all_stats else 0.0
    )

    # 一度でも完璧を達成したエピソード
    perfection_achieved_episodes = [
        s for s in all_stats if s.get("first_perfect_utterance") is not None
    ]
    perfection_achieved_rate = (
        len(perfection_achieved_episodes) / len(all_stats) if all_stats else 0.0
    )

    # 早期終了エピソードの統計（完璧終了まで）
    if early_termination_episodes:
        avg_human_utterances_to_perfect = sum(
            s["human_utterance_count"] for s in early_termination_episodes
        ) / len(early_termination_episodes)
        avg_robot_utterances_to_perfect = sum(
            s["robot_utterance_count"] for s in early_termination_episodes
        ) / len(early_termination_episodes)
    else:
        avg_human_utterances_to_perfect = None
        avg_robot_utterances_to_perfect = None

    # 全エピソードの平均
    avg_human_utterances = sum(s["human_utterance_count"] for s in all_stats) / len(
        all_stats
    )
    avg_robot_utterances = sum(s["robot_utterance_count"] for s in all_stats) / len(
        all_stats
    )
    avg_duration = sum(s["duration_seconds"] for s in all_stats) / len(all_stats)

    # 不安定三角形数の平均（最終値）
    avg_final_unstable_triads = sum(s["final_unstable_triads"] for s in all_stats) / len(
        all_stats
    )

    # 新規指標の平均
    avg_stability_rate = sum(s.get("stability_rate", 0.0) for s in all_stats) / len(
        all_stats
    )
    avg_perfection_rate = sum(s.get("perfection_rate", 0.0) for s in all_stats) / len(
        all_stats
    )
    avg_isolation_occurrence_rate = sum(
        s.get("isolation_occurrence_rate", 0.0) for s in all_stats
    ) / len(all_stats)

    # 初回安定発話数（達成したエピソードのみ）
    first_stable_utterances = [
        s["first_stable_utterance"]
        for s in all_stats
        if s.get("first_stable_utterance") is not None
    ]
    avg_first_stable_utterance = (
        sum(first_stable_utterances) / len(first_stable_utterances)
        if first_stable_utterances
        else None
    )

    # 初回完璧発話数（達成したエピソードのみ）
    first_perfect_utterances = [
        s["first_perfect_utterance"]
        for s in all_stats
        if s.get("first_perfect_utterance") is not None
    ]
    avg_first_perfect_utterance = (
        sum(first_perfect_utterances) / len(first_perfect_utterances)
        if first_perfect_utterances
        else None
    )

    avg_oscillation_count = sum(s.get("oscillation_count", 0) for s in all_stats) / len(
        all_stats
    )
    avg_consecutive_unstable_max = sum(
        s.get("consecutive_unstable_max", 0) for s in all_stats
    ) / len(all_stats)

    avg_edge_score = sum(s.get("avg_edge_score", 0.0) for s in all_stats) / len(all_stats)
    avg_positive_ratio = sum(s.get("avg_positive_ratio", 0.0) for s in all_stats) / len(
        all_stats
    )

    avg_intervention_success_rate = sum(
        s.get("intervention_success_rate", 0.0) for s in all_stats
    ) / len(all_stats)
    avg_improvement_per_intervention = sum(
        s.get("avg_improvement_per_intervention", 0.0) for s in all_stats
    ) / len(all_stats)
    avg_intervention_frequency = sum(
        s.get("intervention_frequency", 0.0) for s in all_stats
    ) / len(all_stats)

    # 新規指標
    avg_stable_rate_per_intervention = sum(
        s.get("stable_rate_per_intervention", 0.0) for s in all_stats
    ) / len(all_stats)
    avg_perfect_rate_per_intervention = sum(
        s.get("perfect_rate_per_intervention", 0.0) for s in all_stats
    ) / len(all_stats)

    # interventions_per_stable は「初回安定までのロボット介入回数」の平均
    # 初回安定を達成したエピソードのみを集計
    interventions_until_first_stable_list = [
        s.get("robot_interventions_until_first_stable", 0.0)
        for s in all_stats
        if s.get("robot_interventions_until_first_stable") is not None
    ]
    avg_interventions_per_stable = (
        sum(interventions_until_first_stable_list) / len(interventions_until_first_stable_list)
        if interventions_until_first_stable_list
        else 0.0
    )

    # interventions_per_perfect は「初回完璧までのロボット介入回数」の平均
    # 初回完璧を達成したエピソードのみを集計
    interventions_until_first_perfect_list = [
        s.get("robot_interventions_until_first_perfect", 0.0)
        for s in all_stats
        if s.get("robot_interventions_until_first_perfect") is not None
    ]
    avg_interventions_per_perfect = (
        sum(interventions_until_first_perfect_list) / len(interventions_until_first_perfect_list)
        if interventions_until_first_perfect_list
        else 0.0
    )

    # 地雷タイプのカウント
    all_common_count = sum(1 for s in all_stats if s.get("trigger_type") == "all_common")
    two_person_count = sum(1 for s in all_stats if s.get("trigger_type") == "two_person")

    stats = {
        "total_episodes": len(all_stats),
        # 基本指標
        "perfect_completion_rate": perfect_completion_rate,
        "perfect_completion_episodes": len(early_termination_episodes),
        "stability_achieved_rate": stability_achieved_rate,
        "stability_achieved_episodes": len(stability_achieved_episodes),
        "perfection_achieved_rate": perfection_achieved_rate,
        "perfection_achieved_episodes": len(perfection_achieved_episodes),
        # 安定性指標
        "avg_stability_rate": avg_stability_rate,
        "avg_perfection_rate": avg_perfection_rate,
        "avg_isolation_occurrence_rate": avg_isolation_occurrence_rate,
        # 発話数指標
        "avg_human_utterances": avg_human_utterances,
        "avg_robot_utterances": avg_robot_utterances,
        "avg_human_utterances_to_perfect": avg_human_utterances_to_perfect,
        "avg_robot_utterances_to_perfect": avg_robot_utterances_to_perfect,
        "avg_first_stable_utterance": avg_first_stable_utterance,
        "avg_first_perfect_utterance": avg_first_perfect_utterance,
        # 構造指標
        "avg_final_unstable_triads": avg_final_unstable_triads,
        "avg_oscillation_count": avg_oscillation_count,
        "avg_consecutive_unstable_max": avg_consecutive_unstable_max,
        # 関係性スコア指標
        "avg_edge_score": avg_edge_score,
        "avg_positive_ratio": avg_positive_ratio,
        # 介入効果指標
        "avg_intervention_success_rate": avg_intervention_success_rate,
        "avg_improvement_per_intervention": avg_improvement_per_intervention,
        "avg_intervention_frequency": avg_intervention_frequency,
        "avg_stable_rate_per_intervention": avg_stable_rate_per_intervention,
        "avg_perfect_rate_per_intervention": avg_perfect_rate_per_intervention,
        "avg_interventions_per_stable": avg_interventions_per_stable,
        "avg_interventions_per_perfect": avg_interventions_per_perfect,
        # 地雷タイプ統計
        "all_common_trigger_count": all_common_count,
        "two_person_trigger_count": two_person_count,
        # その他
        "avg_duration_seconds": avg_duration,
    }

    return stats


def save_results(all_stats: List[Dict], summary_stats: Dict, output_dir: str):
    """
    結果を保存

    Args:
        all_stats: 各エピソードの統計リスト
        summary_stats: 集計統計
        output_dir: 出力ディレクトリ
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # サマリーを保存
    summary_path = output_path / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    print(f"\n💾 サマリーを保存: {summary_path}")

    # 各エピソードの詳細を保存
    for stats in all_stats:
        episode_id = stats["episode_id"]
        episode_path = output_path / f"episode_{episode_id}.json"

        # ログは別ファイルに保存（サイズが大きくなるため）
        logs = stats.pop("logs", [])
        robot_utterances = stats.pop("robot_utterances", [])

        with open(episode_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 会話ログを保存
        log_path = output_path / f"episode_{episode_id}_conversation.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"エピソード {episode_id}\n")
            f.write(f"話題: {stats['topic']}\n")
            if stats.get("topic_trigger"):
                f.write(f"トリガー: {stats['topic_trigger']}\n")
            f.write("=" * 80 + "\n\n")

            for log in logs:
                speaker = log.get("speaker", "?")
                utterance = log.get("utterance", "")
                f.write(f"[{speaker}] {utterance}\n")

    print(f"💾 各エピソードの詳細を保存: {output_path}")


def print_episode_details(all_stats: List[Dict]):
    """
    エピソードごとの詳細を表示

    Args:
        all_stats: 各エピソードの統計リスト
    """
    print(f"\n{'='*80}")
    print(f"📋 エピソードごとの詳細")
    print(f"{'='*80}")

    for stats in all_stats:
        episode_id = stats["episode_id"]
        print(f"\n--- エピソード {episode_id} ---")
        print(f"話題: {stats['topic']}")
        if stats.get("topic_trigger"):
            trigger_type_label = "全員共通" if stats.get("trigger_type") == "all_common" else "2人地雷"
            print(f"トリガー: {stats['topic_trigger']} ({trigger_type_label})")
        print(f"総人間発話数: {stats['human_utterance_count']}")
        print(f"ロボット介入回数: {stats['robot_utterance_count']}")
        print(
            f"早期終了: {'✅ はい (2連続完璧)' if stats['early_termination'] else '❌ いいえ'}"
        )
        print(f"最終状態: {'✅ 安定' if stats['final_stable'] else '❌ 不安定'}")
        print(f"最終完璧: {'✅ はい' if stats.get('final_perfect', False) else '❌ いいえ'}")
        print(f"最終不安定三角形数: {stats['final_unstable_triads']}")
        print(f"最終疎外ノード: {stats['final_isolated_nodes']}")
        print(f"所要時間: {stats['duration_seconds']:.1f}秒")
        print(f"安定率: {stats['stability_rate']*100:.1f}%")
        print(f"完璧率: {stats['perfection_rate']*100:.1f}%")
        print(f"疎外発生率: {stats['isolation_occurrence_rate']*100:.1f}%")
        if stats.get("first_stable_utterance"):
            print(f"初回安定達成: {stats['first_stable_utterance']}発話")
        if stats.get("first_perfect_utterance"):
            print(f"初回完璧達成: {stats['first_perfect_utterance']}発話")
        if stats.get("robot_interventions_until_first_stable") is not None:
            print(
                f"初回安定までのロボット介入回数: {stats['robot_interventions_until_first_stable']}"
            )
        if stats.get("robot_interventions_until_first_perfect") is not None:
            print(
                f"初回完璧までのロボット介入回数: {stats['robot_interventions_until_first_perfect']}"
            )
        print(f"切り替わり回数: {stats['oscillation_count']}回")
        print(f"連続不安定最大: {stats['consecutive_unstable_max']}回")
        print(f"平均エッジスコア: {stats['avg_edge_score']:+.2f}")
        print(f"平均正エッジ割合: {stats['avg_positive_ratio']*100:.1f}%")
        print(f"介入成功率: {stats['intervention_success_rate']*100:.1f}%")
        print(f"介入あたり平均改善度: {stats['avg_improvement_per_intervention']:+.3f}")
        print(f"介入頻度: {stats['intervention_frequency']:.2f}")
        print(f"1介入あたりの安定評価回数: {stats['stable_rate_per_intervention']:.2f}")
        print(f"1介入あたりの完璧評価回数: {stats.get('perfect_rate_per_intervention', 0.0):.2f}")


def print_summary(stats: Dict):
    """
    サマリー統計を表示

    Args:
        stats: 集計統計
    """
    print(f"\n{'='*80}")
    print(f"📊 シミュレーション結果サマリー（全エピソードの平均）")
    print(f"{'='*80}")
    print(f"総エピソード数: {stats['total_episodes']}")

    print(f"\n【エピソード達成率】")
    print(
        f"完璧終了エピソード数（2連続完璧）: {stats['perfect_completion_episodes']} ({stats['perfect_completion_rate']*100:.1f}%)"
    )
    print(
        f"一度でも安定達成: {stats['stability_achieved_episodes']} ({stats['stability_achieved_rate']*100:.1f}%)"
    )
    print(
        f"一度でも完璧達成: {stats['perfection_achieved_episodes']} ({stats['perfection_achieved_rate']*100:.1f}%)"
    )

    print(f"\n【安定性指標】")
    print(f"平均安定率: {stats['avg_stability_rate']*100:.1f}%")
    print(f"平均完璧率: {stats['avg_perfection_rate']*100:.1f}%")
    print(f"平均疎外発生率: {stats['avg_isolation_occurrence_rate']*100:.1f}%")

    print(f"\n【発話数指標】")
    print(f"平均人間発話数: {stats['avg_human_utterances']:.1f}")
    print(f"平均ロボット介入回数: {stats['avg_robot_utterances']:.1f}")

    if stats["avg_human_utterances_to_perfect"] is not None:
        print(f"\n【早期終了エピソードのみ（2連続完璧）】")
        print(
            f"平均人間発話数（完璧終了まで）: {stats['avg_human_utterances_to_perfect']:.1f}"
        )
        print(
            f"平均ロボット介入回数（完璧終了まで）: {stats['avg_robot_utterances_to_perfect']:.1f}"
        )

    if stats["avg_first_stable_utterance"] is not None:
        print(f"\n【初回達成指標】")
        print(f"平均初回安定達成: {stats['avg_first_stable_utterance']:.1f}発話")
    if stats["avg_first_perfect_utterance"] is not None:
        if stats["avg_first_stable_utterance"] is None:
            print(f"\n【初回達成指標】")
        print(f"平均初回完璧達成: {stats['avg_first_perfect_utterance']:.1f}発話")

    print(f"\n【構造指標】")
    print(f"平均最終不安定三角形数: {stats['avg_final_unstable_triads']:.2f}")
    print(f"平均切り替わり回数: {stats['avg_oscillation_count']:.1f}")
    print(f"平均最大連続不安定: {stats['avg_consecutive_unstable_max']:.1f}")

    print(f"\n【関係性スコア】")
    print(f"平均エッジスコア: {stats['avg_edge_score']:+.2f}")
    print(f"平均正エッジ割合: {stats['avg_positive_ratio']*100:.1f}%")

    print(f"\n【介入効果】")
    print(f"介入成功率: {stats['avg_intervention_success_rate']*100:.1f}%")
    print(f"介入あたり平均改善度: {stats['avg_improvement_per_intervention']:+.3f}")
    print(f"介入頻度: {stats['avg_intervention_frequency']:.2f}")
    print(f"1介入あたりの安定評価回数: {stats['avg_stable_rate_per_intervention']:.2f}")
    print(f"1介入あたりの完璧評価回数: {stats['avg_perfect_rate_per_intervention']:.2f}")
    print(f"初回安定までのロボット介入回数（平均）: {stats['avg_interventions_per_stable']:.2f}")
    print(f"初回完璧までのロボット介入回数（平均）: {stats['avg_interventions_per_perfect']:.2f}")

    print(f"\n【地雷タイプ】")
    print(f"全員共通地雷エピソード数: {stats['all_common_trigger_count']}")
    print(f"2人地雷エピソード数: {stats['two_person_trigger_count']}")

    print(f"\n【その他】")
    print(f"平均所要時間: {stats['avg_duration_seconds']:.1f}秒")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="会話シミュレーション実行")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="実行するエピソード数（デフォルト: config.yamlの設定）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力ディレクトリ（デフォルト: results/MMDD_HHMMSS）",
    )

    args = parser.parse_args()

    # 設定読み込み
    cfg = config.get_config()
    sim_cfg = getattr(cfg, "simulation", None)

    num_episodes = (
        args.num_episodes
        if args.num_episodes is not None
        else getattr(sim_cfg, "num_episodes", 10)
    )

    # 出力ディレクトリ
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        output_dir = f"results/simulation_{timestamp}"

    print(f"{'='*80}")
    print(f"🚀 会話シミュレーション開始")
    print(f"{'='*80}")
    print(f"エピソード数: {num_episodes}")
    print(f"出力ディレクトリ: {output_dir}")

    # シミュレーション環境
    env = SimulationEnvironment()

    # 各エピソードを実行
    # ロボット介入があったエピソードのみをカウント
    all_stats = []
    valid_episode_count = 0  # ロボット介入があったエピソード数
    total_episode_count = 0  # 実行した総エピソード数（スキップ含む）

    while valid_episode_count < num_episodes:
        total_episode_count += 1
        try:
            stats = env.run_episode(total_episode_count)

            # ロボット介入がなかったエピソードはスキップ
            if stats["robot_utterance_count"] == 0 and stats["early_termination"]:
                print(
                    f"\n⏭️  エピソード {total_episode_count} はロボット介入なしで早期終了したため、統計から除外します\n"
                )
                continue

            # 有効なエピソードとしてカウント
            all_stats.append(stats)
            valid_episode_count += 1

        except KeyboardInterrupt:
            print("\n⚠️ 中断されました")
            break
        except Exception as e:
            print(f"\n❌ エピソード {total_episode_count} でエラー: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_stats:
        print("❌ 実行されたエピソードがありません")
        return

    print(f"\n{'='*80}")
    print(
        f"✅ 有効エピソード数: {valid_episode_count} / 総実行エピソード数: {total_episode_count}"
    )
    print(f"{'='*80}")

    # 統計計算
    summary_stats = calculate_statistics(all_stats)

    # エピソードごとの詳細を表示
    print_episode_details(all_stats)

    # サマリー表示
    print_summary(summary_stats)

    # 結果保存
    save_results(all_stats, summary_stats, output_dir)

    print(f"\n✅ シミュレーション完了")


if __name__ == "__main__":
    main()
