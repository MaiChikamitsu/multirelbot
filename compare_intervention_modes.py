"""
介入モード比較スクリプト

3つの介入モード（proposal, few_utterances, random_target）を
同じ初期条件（話題、地雷、シード）で実行し、結果を比較する

使用方法:
    python compare_intervention_modes.py [--num-episodes N] [--output OUTPUT_DIR]

例:
    python compare_intervention_modes.py --num-episodes 10
"""

import argparse
import csv
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import config
from simulation_environment import SimulationEnvironment


def generate_initial_utterances(
    env: SimulationEnvironment,
    topic: str,
    topic_trigger: str,
    seed: int,
    num_utterances: int = 3,
) -> tuple[List[Dict], Dict, bool, bool, bool]:
    """
    最初の人間発話を生成し、関係性を評価する

    Args:
        env: シミュレーション環境
        topic: 話題
        topic_trigger: トリガー
        seed: ランダムシード
        num_utterances: 生成する発話数

    Returns:
        (logs, metrics, is_stable, has_isolated, is_perfect): 発話ログ、メトリクス、安定性、孤立有無、完璧判定
    """
    random.seed(seed)
    logs = []

    # num_utterances発話を生成
    for i in range(num_utterances):
        random.seed(seed + i)
        human_replies = env.human_llm.generate_human_reply(
            logs, topic=topic, topic_trigger=topic_trigger, num_speakers=1
        )
        if not human_replies:
            break
        logs.extend(human_replies)

    # 関係性評価
    env.analyzer.reset_ema()  # EMAをリセット
    metrics, is_stable, has_isolated, is_perfect = env.evaluate_relationships(logs)

    return logs, metrics, is_stable, has_isolated, is_perfect


def run_episode_with_mode(
    env: SimulationEnvironment,
    episode_id: int,
    mode: str,
    topic: str,
    topic_trigger: str,
    trigger_type: str,
    initial_logs: List[Dict],
    initial_metrics: Dict,
    seed: int,
) -> Dict:
    """
    指定されたモードでエピソードを実行

    Args:
        env: シミュレーション環境
        episode_id: エピソードID
        mode: 介入モード
        topic: 話題
        topic_trigger: トリガーとなった地雷
        trigger_type: 地雷タイプ
        initial_logs: 初期発話ログ（全モード共通）
        initial_metrics: 初期メトリクス
        seed: ランダムシード

    Returns:
        stats: エピソード統計
    """
    # ランダムシードを設定
    random.seed(seed)

    # configのモードを一時的に変更（envが保持しているcfgインスタンスを直接変更）
    original_mode = getattr(env._CFG.intervention, "mode", "proposal")
    env._CFG.intervention.mode = mode

    print(f"\n{'='*80}")
    print(f"🤖 エピソード {episode_id} - モード: {mode}")
    print(f"{'='*80}")

    # EMA履歴とロボット発話履歴をリセット
    env.analyzer.reset_ema()
    env.past_robot_utterances = []

    # 開始時刻
    start_time = datetime.now()

    # 初期ログをコピー（変更を他のモードに影響させないため）
    import copy

    logs = copy.deepcopy(initial_logs)
    robot_utterances = []

    # 初期発話を表示
    print(f"📖 話題: {topic}")
    if topic_trigger:
        print(f"  (トリガー: {topic_trigger}, タイプ: {trigger_type})")
    print(f"\n初期発話（全モード共通）:")
    for log in logs:
        print(f"[{log['speaker']}] {log['utterance']}")

    # カウンター（初期発話分）
    human_utterance_count = len(logs)
    intervention_count = 0

    # 統計用変数
    stability_checks = []
    perfection_checks = []
    isolation_checks = []
    edge_score_history = []
    positive_ratio_history = []
    intervention_improvements = []
    intervention_reversals = []
    consecutive_stable_count = 0
    consecutive_perfect_count = 0
    consecutive_unstable_count = 0
    consecutive_unstable_max = 0
    first_stable_utterance = None
    first_perfect_utterance = None
    robot_interventions_until_first_stable = None
    robot_interventions_until_first_perfect = None
    last_stable_state = None
    oscillation_count = 0
    early_termination = False

    # 初期関係性は共通部分で評価済み（initial_metricsを使用）
    print(f"\n📊 初期関係性評価 ({human_utterance_count}発話時点)")
    metrics = initial_metrics
    edges = metrics.get("edges", {})

    # EMAのスコアと履歴を手動で設定（LLMを再呼び出ししない）
    env.analyzer.scores = {}
    env.analyzer.history = {}
    for (a, b), score in edges.items():
        key = tuple(sorted([a, b]))
        env.analyzer.scores[key] = score
        # 初期状態なので発話数は各人1回ずつ
        from collections import deque

        env.analyzer.history[key] = deque([1], maxlen=env.analyzer.max_history_sessions)

    # 初期状態を表示
    if edges:
        print(f"  関係性スコア:")
        for (a, b), score in sorted(edges.items()):
            print(f"    {a}-{b}: {score:+.1f}")
        avg_edge_score = sum(edges.values()) / len(edges)
        edge_score_history.append(avg_edge_score)
        positive_edges = sum(1 for score in edges.values() if score > 0)
        positive_ratio = positive_edges / len(edges)
        positive_ratio_history.append(positive_ratio)

    # 初期状態の判定値を計算（generate_initial_utterancesで取得済みだが、念のため再計算）
    unstable_triads = metrics.get("unstable_triads", 0)
    isolated_nodes = metrics.get("isolated_nodes", [])
    is_stable = unstable_triads == 0
    has_isolated = len(isolated_nodes) > 0
    is_perfect = is_stable and not has_isolated

    stability_checks.append(is_stable)
    perfection_checks.append(is_perfect)
    isolation_checks.append(has_isolated)

    print(f"  不安定三角形数: {unstable_triads}")
    print(f"  疎外ノード: {isolated_nodes}")
    print(f"  安定状態: {'✅ はい' if is_stable else '❌ いいえ'}")
    print(f"  完璧状態: {'✅ はい' if is_perfect else '❌ いいえ'}")

    if is_stable and first_stable_utterance is None:
        first_stable_utterance = human_utterance_count
        robot_interventions_until_first_stable = 0
        print(f"  🎯 初回安定達成: {human_utterance_count}発話（ロボット介入0回）")

    if is_perfect and first_perfect_utterance is None:
        first_perfect_utterance = human_utterance_count
        robot_interventions_until_first_perfect = 0
        print(f"  ⭐ 初回完璧達成: {human_utterance_count}発話（ロボット介入0回）")

    # 初期状態での介入判定
    # few_utterancesとrandom_targetは完璧でも介入、proposalは完璧なら介入しない
    should_check_intervention = (mode in ["few_utterances", "random_target"]) or (
        not is_perfect
    )
    if should_check_intervention:
        import networkx as nx

        graph = nx.Graph()
        for (a, b), score in edges.items():
            graph.add_edge(a, b, score=score)
        should_intervene, plan, robot_utterance = env.should_intervene(
            logs, metrics, scores=edges, graph=graph
        )

        if should_intervene and robot_utterance:
            intervention_count += 1
            print(f"\n🤖 ロボット介入 ({intervention_count}回目)")
            print(f"  介入タイプ: {plan.get('type', '不明')}")
            print(f"  発話: {robot_utterance}")

            # 介入対象エッジの介入前スコアを記録
            target_edge = plan.get("edge")
            pre_intervention_score = None
            if target_edge and edges:
                pre_intervention_score = edges.get(target_edge) or edges.get(
                    (target_edge[1], target_edge[0])
                )
                if pre_intervention_score is not None:
                    print(
                        f"  介入対象エッジ: {target_edge[0]}-{target_edge[1]} (介入前: {pre_intervention_score:+.1f})"
                    )

            robot_entry = {
                "speaker": "ロボット",
                "utterance": robot_utterance,
                "plan": plan,
                "pre_score": pre_intervention_score,
                "target_edge": target_edge,
            }
            logs.append(robot_entry)
            robot_utterances.append(robot_entry)

    print()
    while human_utterance_count < env.max_human_utterances:
        # 人間発話を1つ生成
        random.seed(seed + human_utterance_count)
        human_replies = env.human_llm.generate_human_reply(
            logs, topic=topic, topic_trigger=topic_trigger, num_speakers=1
        )

        if not human_replies:
            print("⚠️ 人間発話生成失敗")
            break

        logs.extend(human_replies)
        human_utterance_count += 1

        # 発話を表示
        for reply in human_replies:
            print(f"[{reply['speaker']}] {reply['utterance']}")

        # stability_check_interval ごとに関係性評価
        if human_utterance_count % env.stability_check_interval == 0:
            print(f"📊 関係性評価 ({human_utterance_count}発話時点)")

            metrics, is_stable, has_isolated, is_perfect = env.evaluate_relationships(
                logs
            )

            # 直前のロボット介入の効果を測定
            if robot_utterances:
                last_robot = robot_utterances[-1]
                if "pre_score" in last_robot and last_robot["pre_score"] is not None:
                    target_edge = last_robot.get("target_edge")
                    pre_score = last_robot["pre_score"]
                    edges = metrics.get("edges", {})

                    post_score = None
                    if target_edge:
                        # proposalモード: 特定エッジのスコア変化を測定
                        post_score = edges.get(target_edge) or edges.get(
                            (target_edge[1], target_edge[0])
                        )
                        if post_score is not None:
                            improvement = post_score - pre_score
                            intervention_improvements.append(improvement)
                            # 反転判定（-から+への変化）
                            is_reversal = pre_score < 0 and post_score > 0
                            intervention_reversals.append(is_reversal)
                            reversal_mark = " 🔄反転" if is_reversal else ""
                            print(
                                f"  📈 介入効果: {target_edge[0]}-{target_edge[1]} = {pre_score:+.1f} → {post_score:+.1f} (変化: {improvement:+.1f}){reversal_mark}"
                            )
                    elif edges:
                        # few_utterances/random_targetモード: 全エッジ平均の変化を測定
                        post_score = sum(edges.values()) / len(edges)
                        improvement = post_score - pre_score
                        intervention_improvements.append(improvement)
                        # 反転判定（平均が-から+への変化）
                        is_reversal = pre_score < 0 and post_score > 0
                        intervention_reversals.append(is_reversal)
                        reversal_mark = " 🔄反転" if is_reversal else ""
                        print(
                            f"  📈 介入効果（全エッジ平均）: {pre_score:+.1f} → {post_score:+.1f} (変化: {improvement:+.1f}){reversal_mark}"
                        )

                    if post_score is not None:
                        del last_robot["pre_score"]
                        del last_robot["target_edge"]

            print(f"  不安定三角形数: {metrics.get('unstable_triads', 0)}")
            print(f"  疎外ノード: {metrics.get('isolated_nodes', [])}")
            print(f"  安定状態: {'✅ はい' if is_stable else '❌ いいえ'}")
            print(f"  完璧状態: {'✅ はい' if is_perfect else '❌ いいえ'}")

            # エッジスコアを表示と記録
            edges = metrics.get("edges", {})
            if edges:
                print(f"  関係性スコア:")
                for (a, b), score in sorted(edges.items()):
                    print(f"    {a}-{b}: {score:+.1f}")

                avg_edge_score = sum(edges.values()) / len(edges)
                edge_score_history.append(avg_edge_score)

                positive_edges = sum(1 for score in edges.values() if score > 0)
                positive_ratio = positive_edges / len(edges)
                positive_ratio_history.append(positive_ratio)

            # 安定性、完璧性、疎外ノード有無を記録
            stability_checks.append(is_stable)
            perfection_checks.append(is_perfect)
            isolation_checks.append(has_isolated)

            # 初回安定時の発話数とロボット介入回数を記録
            if is_stable and first_stable_utterance is None:
                first_stable_utterance = human_utterance_count
                robot_interventions_until_first_stable = intervention_count
                print(
                    f"  🎯 初回安定達成: {human_utterance_count}発話（ロボット介入{intervention_count}回）"
                )

            # 初回完璧時の発話数とロボット介入回数を記録
            if is_perfect and first_perfect_utterance is None:
                first_perfect_utterance = human_utterance_count
                robot_interventions_until_first_perfect = intervention_count
                print(
                    f"  ⭐ 初回完璧達成: {human_utterance_count}発話（ロボット介入{intervention_count}回）"
                )

            # 安定⇔不安定の切り替わりを検出
            if last_stable_state is not None and last_stable_state != is_stable:
                oscillation_count += 1
            last_stable_state = is_stable

            # 早期終了判定: 完璧状態が2連続
            if is_perfect:
                consecutive_perfect_count += 1
                consecutive_unstable_count = 0
                print(
                    f"  連続完璧回数: {consecutive_perfect_count}/{env.consecutive_stable_threshold}\n"
                )

                if consecutive_perfect_count >= env.consecutive_stable_threshold:
                    print(
                        f"\n🎉 {env.consecutive_stable_threshold}回連続で完璧状態 → エピソード終了"
                    )
                    early_termination = True
                    break
            else:
                # 完璧ではない（不安定 or 孤立ノードあり）
                consecutive_perfect_count = 0

                if is_stable:
                    # 安定だが孤立ノードあり
                    consecutive_unstable_count = 0
                    print(f"  安定状態維持（孤立ノードあり）")
                else:
                    # 不安定
                    consecutive_unstable_count += 1
                    consecutive_unstable_max = max(
                        consecutive_unstable_max, consecutive_unstable_count
                    )

            # 介入判定（few_utterances/random_targetは完璧でも介入、proposalは完璧なら介入しない）
            should_check_intervention = (
                mode in ["few_utterances", "random_target"]
            ) or (not is_perfect)
            if should_check_intervention:
                import networkx as nx

                graph = nx.Graph()
                for (a, b), score in edges.items():
                    graph.add_edge(a, b, score=score)
                should_intervene, plan, robot_utterance = env.should_intervene(
                    logs, metrics, scores=edges, graph=graph
                )

                if should_intervene and robot_utterance:
                    intervention_count += 1
                    print(f"\n🤖 ロボット介入 ({intervention_count}回目)")
                    print(f"  介入タイプ: {plan.get('type', '不明')}")
                    print(f"  発話: {robot_utterance}")

                    # 介入対象エッジの介入前スコアを記録
                    target_edge = plan.get("edge")
                    pre_intervention_score = None

                    if target_edge and edges:
                        # proposalモード: 特定エッジのスコアを記録
                        pre_intervention_score = edges.get(target_edge) or edges.get(
                            (target_edge[1], target_edge[0])
                        )
                        if pre_intervention_score is not None:
                            print(
                                f"  介入対象エッジ: {target_edge[0]}-{target_edge[1]} (介入前: {pre_intervention_score:+.1f})"
                            )
                    elif edges:
                        # few_utterances/random_targetモード: 全エッジの平均スコアを記録
                        pre_intervention_score = sum(edges.values()) / len(edges)
                        print(
                            f"  全エッジ平均スコア (介入前): {pre_intervention_score:+.1f}"
                        )

                    robot_entry = {
                        "speaker": "ロボット",
                        "utterance": robot_utterance,
                        "plan": plan,
                        "pre_score": pre_intervention_score,
                        "target_edge": target_edge,
                    }
                    logs.append(robot_entry)
                    robot_utterances.append(robot_entry)

                print()  # 空行を追加

    # 終了時刻
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 最終評価
    if not early_termination:
        final_metrics, final_stable, final_has_isolated, final_perfect = (
            env.evaluate_relationships(logs)
        )
    else:
        final_metrics = metrics
        final_stable = is_stable
        final_has_isolated = has_isolated
        final_perfect = is_perfect

    print(f"\n📈 エピソード終了 (モード: {mode})")
    print(f"  総人間発話数: {human_utterance_count}")
    print(f"  ロボット介入回数: {intervention_count}")
    print(f"  早期終了: {'✅ はい (2連続完璧)' if early_termination else '❌ いいえ'}")
    print(f"  最終状態: {'✅ 安定' if final_stable else '❌ 不安定'}")
    print(f"  最終完璧: {'✅ はい' if final_perfect else '❌ いいえ'}")

    # 新規指標の計算
    num_checks = len(stability_checks)
    stable_count_in_checks = sum(stability_checks)
    perfect_count_in_checks = sum(perfection_checks)

    if early_termination:
        remaining_utterances = env.max_human_utterances - human_utterance_count
        remaining_checks = remaining_utterances // env.stability_check_interval

        # few_utterancesとrandom_targetモードでは、早期終了後も介入があったとカウント
        if mode in ("few_utterances", "random_target"):
            intervention_count += remaining_checks

        total_checks = num_checks + remaining_checks
        total_stable_checks = stable_count_in_checks + remaining_checks
        total_perfect_checks = perfect_count_in_checks + remaining_checks
        stability_rate = total_stable_checks / total_checks if total_checks > 0 else 0.0
        perfection_rate = (
            total_perfect_checks / total_checks if total_checks > 0 else 0.0
        )
    else:
        total_checks = num_checks
        total_stable_checks = stable_count_in_checks
        total_perfect_checks = perfect_count_in_checks
        stability_rate = stable_count_in_checks / num_checks if num_checks > 0 else 0.0
        perfection_rate = (
            perfect_count_in_checks / num_checks if num_checks > 0 else 0.0
        )

    isolation_occurrence_rate = (
        sum(isolation_checks) / num_checks if num_checks > 0 else 0.0
    )
    avg_edge_score = (
        sum(edge_score_history) / len(edge_score_history) if edge_score_history else 0.0
    )
    avg_positive_ratio = (
        sum(positive_ratio_history) / len(positive_ratio_history)
        if positive_ratio_history
        else 0.0
    )

    intervention_success_rate = 0.0
    intervention_reversal_rate = 0.0
    avg_improvement_per_intervention = 0.0
    if intervention_improvements:
        successful_interventions = sum(
            1 for imp in intervention_improvements if imp > 0
        )
        intervention_success_rate = successful_interventions / len(
            intervention_improvements
        )
        avg_improvement_per_intervention = sum(intervention_improvements) / len(
            intervention_improvements
        )
    if intervention_reversals:
        intervention_reversal_rate = sum(intervention_reversals) / len(
            intervention_reversals
        )

    intervention_frequency = (
        intervention_count / human_utterance_count if human_utterance_count > 0 else 0.0
    )

    stable_rate_per_intervention = (
        total_stable_checks / intervention_count if intervention_count > 0 else 0.0
    )
    perfect_rate_per_intervention = (
        total_perfect_checks / intervention_count if intervention_count > 0 else 0.0
    )
    interventions_per_stable = (
        robot_interventions_until_first_stable
        if robot_interventions_until_first_stable is not None
        else 0.0
    )
    interventions_per_perfect = (
        robot_interventions_until_first_perfect
        if robot_interventions_until_first_perfect is not None
        else 0.0
    )

    # configを元に戻す
    env._CFG.intervention.mode = original_mode

    # 初回から完璧だったかを判定（介入機会がなかった）
    was_perfect_from_start = (
        first_perfect_utterance == env.stability_check_interval and early_termination
    )

    # 統計を返す
    stats = {
        "episode_id": episode_id,
        "mode": mode,
        "topic": topic,
        "topic_trigger": topic_trigger if topic_trigger else "",
        "trigger_type": trigger_type if trigger_type else "",
        "human_utterance_count": human_utterance_count,
        "robot_utterance_count": intervention_count,
        "early_termination": early_termination,
        "was_perfect_from_start": was_perfect_from_start,
        "final_stable": final_stable,
        "final_perfect": final_perfect,
        "final_unstable_triads": final_metrics.get("unstable_triads", 0),
        "final_isolated_nodes": len(final_metrics.get("isolated_nodes", [])),
        "duration_seconds": duration,
        "stability_rate": stability_rate,
        "perfection_rate": perfection_rate,
        "isolation_occurrence_rate": isolation_occurrence_rate,
        "first_stable_utterance": (
            first_stable_utterance if first_stable_utterance else 0
        ),
        "first_perfect_utterance": (
            first_perfect_utterance if first_perfect_utterance else 0
        ),
        "robot_interventions_until_first_stable": (
            robot_interventions_until_first_stable
            if robot_interventions_until_first_stable is not None
            else 0
        ),
        "robot_interventions_until_first_perfect": (
            robot_interventions_until_first_perfect
            if robot_interventions_until_first_perfect is not None
            else 0
        ),
        "oscillation_count": oscillation_count,
        "consecutive_unstable_max": consecutive_unstable_max,
        "avg_edge_score": avg_edge_score,
        "avg_positive_ratio": avg_positive_ratio,
        "intervention_success_rate": intervention_success_rate,
        "intervention_reversal_rate": intervention_reversal_rate,
        "avg_improvement_per_intervention": avg_improvement_per_intervention,
        "intervention_frequency": intervention_frequency,
        "stable_rate_per_intervention": stable_rate_per_intervention,
        "perfect_rate_per_intervention": perfect_rate_per_intervention,
        "interventions_per_stable": interventions_per_stable,
        "interventions_per_perfect": interventions_per_perfect,
    }

    return stats


def save_comparison_results(all_results: List[Dict], output_dir: str):
    """
    比較結果をCSVに保存

    Args:
        all_results: 全ての結果リスト
        output_dir: 出力ディレクトリ
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "mode_comparison.csv"

    if not all_results:
        print("⚠️ 保存する結果がありません")
        return

    # CSVフィールド名
    fieldnames = [
        "episode_id",
        "mode",
        "topic",
        "topic_trigger",
        "trigger_type",
        "human_utterance_count",
        "robot_utterance_count",
        "early_termination",
        "was_perfect_from_start",
        "final_stable",
        "final_perfect",
        "final_unstable_triads",
        "final_isolated_nodes",
        "duration_seconds",
        "stability_rate",
        "perfection_rate",
        "isolation_occurrence_rate",
        "first_stable_utterance",
        "first_perfect_utterance",
        "robot_interventions_until_first_stable",
        "robot_interventions_until_first_perfect",
        "oscillation_count",
        "consecutive_unstable_max",
        "avg_edge_score",
        "avg_positive_ratio",
        "intervention_success_rate",
        "intervention_reversal_rate",
        "avg_improvement_per_intervention",
        "intervention_frequency",
        "stable_rate_per_intervention",
        "perfect_rate_per_intervention",
        "interventions_per_stable",
        "interventions_per_perfect",
    ]

    # CSVに書き込み
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # episode_id -> mode順でソート
        sorted_results = sorted(all_results, key=lambda x: (x["episode_id"], x["mode"]))
        writer.writerows(sorted_results)

    print(f"\n💾 比較結果を保存: {csv_path}")


def save_mode_summary(all_results: List[Dict], output_dir: str):
    """
    モードごとのサマリーをCSVに保存

    Args:
        all_results: 全ての結果リスト
        output_dir: 出力ディレクトリ
    """
    output_path = Path(output_dir)
    summary_csv_path = output_path / "mode_summary.csv"

    modes = ["proposal", "few_utterances", "random_target"]
    mode_summaries = []

    for mode in modes:
        mode_results = [r for r in all_results if r["mode"] == mode]
        if not mode_results:
            continue

        # 各指標の平均を計算
        summary = {
            "mode": mode,
            "num_episodes": len(mode_results),
            "avg_stability_rate": sum(r["stability_rate"] for r in mode_results)
            / len(mode_results),
            "avg_perfection_rate": sum(r["perfection_rate"] for r in mode_results)
            / len(mode_results),
            "avg_isolation_occurrence_rate": sum(
                r["isolation_occurrence_rate"] for r in mode_results
            )
            / len(mode_results),
            "avg_robot_utterance_count": sum(
                r["robot_utterance_count"] for r in mode_results
            )
            / len(mode_results),
            "avg_first_stable_utterance": (
                sum(
                    r["first_stable_utterance"]
                    for r in mode_results
                    if r["first_stable_utterance"] > 0
                )
                / len([r for r in mode_results if r["first_stable_utterance"] > 0])
                if any(r["first_stable_utterance"] > 0 for r in mode_results)
                else 0
            ),
            "avg_first_perfect_utterance": (
                sum(
                    r["first_perfect_utterance"]
                    for r in mode_results
                    if r["first_perfect_utterance"] > 0
                )
                / len([r for r in mode_results if r["first_perfect_utterance"] > 0])
                if any(r["first_perfect_utterance"] > 0 for r in mode_results)
                else 0
            ),
            "avg_oscillation_count": sum(r["oscillation_count"] for r in mode_results)
            / len(mode_results),
            "avg_consecutive_unstable_max": sum(
                r["consecutive_unstable_max"] for r in mode_results
            )
            / len(mode_results),
            "avg_edge_score": sum(r["avg_edge_score"] for r in mode_results)
            / len(mode_results),
            "avg_positive_ratio": sum(r["avg_positive_ratio"] for r in mode_results)
            / len(mode_results),
            "avg_intervention_success_rate": sum(
                r["intervention_success_rate"] for r in mode_results
            )
            / len(mode_results),
            "avg_intervention_reversal_rate": sum(
                r["intervention_reversal_rate"] for r in mode_results
            )
            / len(mode_results),
            "avg_improvement_per_intervention": sum(
                r["avg_improvement_per_intervention"] for r in mode_results
            )
            / len(mode_results),
            "avg_intervention_frequency": sum(
                r["intervention_frequency"] for r in mode_results
            )
            / len(mode_results),
            "early_termination_count": sum(
                1 for r in mode_results if r["early_termination"]
            ),
            "early_termination_rate": sum(
                1 for r in mode_results if r["early_termination"]
            )
            / len(mode_results),
        }
        mode_summaries.append(summary)

    # CSVに書き込み
    if mode_summaries:
        fieldnames = list(mode_summaries[0].keys())
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mode_summaries)

        print(f"💾 モード別サマリーを保存: {summary_csv_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="介入モード比較実験")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="実行するエピソード数（デフォルト: 10）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力ディレクトリ（デフォルト: results/mode_comparison_MMDD_HHMMSS）",
    )

    args = parser.parse_args()

    num_episodes = args.num_episodes

    # 出力ディレクトリ
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        output_dir = f"results/mode_comparison_{timestamp}"

    print(f"{'='*80}")
    print(f"🚀 介入モード比較実験開始")
    print(f"{'='*80}")
    print(f"エピソード数: {num_episodes}")
    print(f"比較モード: proposal, few_utterances, random_target")
    print(f"出力ディレクトリ: {output_dir}")

    # シミュレーション環境
    env = SimulationEnvironment()

    # 結果を格納
    all_results = []

    # 有効エピソード数と試行回数
    valid_episode_count = 0
    episode_attempt = 0

    # 有効なエピソードがnum_episodesになるまで実行
    while valid_episode_count < num_episodes:
        episode_attempt += 1

        print(f"\n{'='*80}")
        print(f"📋 エピソード試行 {episode_attempt} の準備")
        print(f"{'='*80}")

        # エピソード用のシードを設定（話題生成に使用）
        seed = episode_attempt * 1000
        random.seed(seed)

        # 話題を生成（3モード共通）
        # スキップされたエピソードを除き、有効エピソードで交互になるようにする
        prefer_type = "all_common" if valid_episode_count % 2 == 0 else "two_person"
        topic, topic_trigger, trigger_type = env.generate_topic(prefer_type)

        print(f"📖 共通話題: {topic}")
        if topic_trigger:
            print(f"  (トリガー: {topic_trigger}, タイプ: {trigger_type})")

        # 最初の3発話を生成（全モード共通）
        print(f"\n🎬 初期発話を生成中...")
        initial_logs, initial_metrics, is_stable, has_isolated, is_perfect = (
            generate_initial_utterances(
                env,
                topic,
                topic_trigger,
                seed,
                num_utterances=env.stability_check_interval,
            )
        )

        # 初期発話を表示
        print(f"\n初期発話（全モード共通）:")
        for log in initial_logs:
            print(f"[{log['speaker']}] {log['utterance']}")

        print(f"\n初期関係性:")
        print(f"  不安定三角形数: {initial_metrics.get('unstable_triads', 0)}")
        print(f"  疎外ノード: {initial_metrics.get('isolated_nodes', [])}")
        print(f"  安定状態: {'✅ はい' if is_stable else '❌ いいえ'}")
        print(f"  完璧状態: {'✅ はい' if is_perfect else '❌ いいえ'}")

        # 初期状態が完璧な場合、スキップ
        if is_perfect:
            print(
                f"\n⏭️  初期状態が完璧（介入機会なし）のため、このエピソードをスキップします"
            )
            continue

        # 3つのモードで実行
        modes = ["proposal", "few_utterances", "random_target"]
        episode_results = []
        episode_failed = False

        for mode in modes:
            try:
                # 各モードで異なるシードを使用（ロボット発話のランダム性を保つため）
                mode_seed = (
                    seed
                    + {"proposal": 1, "few_utterances": 2, "random_target": 3}[mode]
                )

                stats = run_episode_with_mode(
                    env,
                    episode_attempt,
                    mode,
                    topic,
                    topic_trigger,
                    trigger_type,
                    initial_logs,
                    initial_metrics,
                    mode_seed,
                )
                episode_results.append(stats)

            except KeyboardInterrupt:
                print("\n⚠️ 中断されました")
                episode_failed = True
                break
            except Exception as e:
                print(
                    f"\n❌ エピソード {episode_attempt} (モード: {mode}) でエラー: {e}"
                )
                import traceback

                traceback.print_exc()
                episode_failed = True
                break

        # 中断またはエラーがあった場合は終了
        if episode_failed:
            break

        # 有効なエピソードとして追加（episode_idを振り直す）
        valid_episode_count += 1
        for stats in episode_results:
            stats["episode_id"] = valid_episode_count
            all_results.append(stats)

        print(f"\n✅ 有効エピソード {valid_episode_count} として記録")

    if not all_results:
        print("❌ 実行された結果がありません")
        return

    print(f"\n{'='*80}")
    print(f"✅ 全エピソード完了")
    print(f"{'='*80}")
    print(f"有効エピソード数: {valid_episode_count} / 総試行数: {episode_attempt}")
    print(f"総実行数: {len(all_results)} ({valid_episode_count}エピソード × 3モード)")

    # 結果を保存
    save_comparison_results(all_results, output_dir)
    save_mode_summary(all_results, output_dir)

    # 簡易統計を表示
    print(f"\n{'='*80}")
    print(f"📊 モード別サマリー")
    print(f"{'='*80}")

    for mode in ["proposal", "few_utterances", "random_target"]:
        mode_results = [r for r in all_results if r["mode"] == mode]
        if not mode_results:
            continue

        avg_stability = sum(r["stability_rate"] for r in mode_results) / len(
            mode_results
        )
        avg_perfection = sum(r["perfection_rate"] for r in mode_results) / len(
            mode_results
        )
        avg_interventions = sum(r["robot_utterance_count"] for r in mode_results) / len(
            mode_results
        )
        early_term_count = sum(1 for r in mode_results if r["early_termination"])

        print(f"\n【{mode}】")
        print(f"  平均安定率: {avg_stability*100:.1f}%")
        print(f"  平均完璧率: {avg_perfection*100:.1f}%")
        print(f"  平均介入回数: {avg_interventions:.1f}")
        print(f"  早期終了エピソード数: {early_term_count}/{len(mode_results)}")

    print(f"\n✅ 比較実験完了")


if __name__ == "__main__":
    main()
