from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Mapping, Sequence

import numpy as np

import config
import relation_estimator_from_txt as relation_estimator
from baselines.common import Pair, clamp_score, iter_pairs, label_to_pair, pair_to_label
from baselines.interaction_frequency import (
    METHOD_NAME as INTERACTION_NAME,
    METHOD_VERSION as INTERACTION_VERSION,
    estimate_interaction_frequency_with_metadata,
)
from baselines.sentiment import METHOD_NAME as SENTIMENT_NAME, METHOD_VERSION as SENTIMENT_VERSION, estimate_sentiment
from baselines.sma import METHOD_NAME as SMA_NAME, METHOD_VERSION as SMA_VERSION, SimpleMovingAverageScorer
from baselines.stance import METHOD_NAME as STANCE_NAME, METHOD_VERSION as STANCE_VERSION, estimate_stance
from evaluation_utils import (
    DEFAULT_EPISODES,
    EpisodeConfig,
    build_detail_rows,
    build_summary_row,
    episode_suffix,
    evaluate_predictions,
    load_human_data,
    parse_prediction_key,
    prediction_key,
    round_scores_to_predictions,
    write_detail_csv,
    write_summary_csv,
)
from relation_estimator_from_txt import (
    EMAScorer,
    detect_participants,
    estimate_relation_once,
    parse_conversation_file,
    split_into_rounds,
)

LLM_ONLY_NAME = "LLM-only"
MAVERD_NAME = "MAVeRD"


def serialize_scores(scores: Mapping[Pair, float]) -> Dict[str, float]:
    return {pair_to_label(pair): clamp_score(score) for pair, score in scores.items()}


def deserialize_scores(scores: Mapping[str, float]) -> Dict[Pair, float]:
    return {label_to_pair(label): clamp_score(score) for label, score in scores.items()}


def utterance_counts(logs: Sequence[dict], participants: Sequence[str]) -> Dict[str, int]:
    counts = defaultdict(int)
    for log in logs:
        speaker = log.get("speaker", "")
        if speaker in participants:
            counts[speaker] += 1
    return dict(counts)


def load_episode_inputs(episode_configs: Sequence[EpisodeConfig]) -> Dict[str, dict]:
    episodes = {}
    for episode in episode_configs:
        logs = parse_conversation_file(episode.conversation_file)
        participants = detect_participants(logs)
        rounds = split_into_rounds(logs, participants)
        pairs = iter_pairs(participants)
        human_data = load_human_data(episode.human_file)
        expected_human_keys = {
            prediction_key(round_number, pair)
            for round_number in range(1, len(rounds) + 1)
            for pair in pairs
        }
        actual_human_keys = set(human_data)
        evaluation_skip_reason = None
        if actual_human_keys != expected_human_keys:
            evaluation_skip_reason = (
                f"{episode.name}: conversation and human labels do not match. "
                f"The conversation produces {len(rounds)} rounds "
                f"({len(expected_human_keys)} pair labels), but "
                f"{episode.human_file} contains {len(actual_human_keys)} labels. "
                "MAE and Pearson were skipped."
            )
        episodes[episode.name] = {
            "config": episode,
            "logs": logs,
            "participants": participants,
            "pairs": pairs,
            "round_end_indices": rounds,
            "human_data": human_data,
            "evaluation_skip_reason": evaluation_skip_reason,
        }
    return episodes


def raw_cache_metadata(args: argparse.Namespace, episode_configs: Sequence[EpisodeConfig]) -> Dict[str, object]:
    return {
        "llm_model": args.llm_model,
        "max_history_human": args.max_history_human,
        "num_trials": args.num_trials,
        "episodes": [
            {
                "name": episode.name,
                "conversation_file": episode.conversation_file,
                "human_file": episode.human_file,
            }
            for episode in episode_configs
        ],
    }


def cache_matches(cache: Mapping[str, object], expected_metadata: Mapping[str, object]) -> bool:
    actual = cache.get("metadata", {})
    if not isinstance(actual, Mapping):
        return False
    return all(actual.get(key) == value for key, value in expected_metadata.items())


def generate_raw_score_cache(
    episodes: Mapping[str, dict],
    args: argparse.Namespace,
    cfg,
) -> Dict[str, object]:
    metadata = raw_cache_metadata(args, DEFAULT_EPISODES)
    metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
    cache = {
        "metadata": metadata,
        "episodes": {},
    }
    total_calls = sum(
        len(episode["round_end_indices"]) * args.num_trials
        for episode in episodes.values()
    )
    completed_calls = 0
    print(
        f"[2/4] Generating shared raw LLM scores: {total_calls} API calls",
        flush=True,
    )

    for episode_name, episode in episodes.items():
        episode_cache = {
            "participants": episode["participants"],
            "rounds": [],
        }
        total_rounds = len(episode["round_end_indices"])
        for round_number, end_index in enumerate(episode["round_end_indices"], start=1):
            round_logs = episode["logs"][: end_index + 1]
            round_entry = {
                "round": round_number,
                "end_index": end_index,
                "utterance_counts": utterance_counts(round_logs, episode["participants"]),
                "trials": [],
            }
            print(
                f"  {episode_name} round {round_number}/{total_rounds}: ",
                end="",
                flush=True,
            )
            for _ in range(args.num_trials):
                raw_scores = estimate_relation_once(
                    round_logs,
                    episode["participants"],
                    args.max_history_human,
                    args.llm_model,
                    cfg,
                )
                round_entry["trials"].append(serialize_scores(raw_scores))
                completed_calls += 1
                print(".", end="", flush=True)
            print(
                f" done ({completed_calls}/{total_calls} calls)",
                flush=True,
            )
            episode_cache["rounds"].append(round_entry)
        cache["episodes"][episode_name] = episode_cache

    os.makedirs(os.path.dirname(args.raw_cache), exist_ok=True)
    with open(args.raw_cache, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return cache


def load_or_generate_raw_score_cache(
    episodes: Mapping[str, dict],
    args: argparse.Namespace,
    cfg,
) -> Dict[str, object]:
    expected = raw_cache_metadata(args, DEFAULT_EPISODES)
    if os.path.exists(args.raw_cache) and not args.refresh_raw_cache:
        with open(args.raw_cache, encoding="utf-8") as f:
            cache = json.load(f)
        if cache_matches(cache, expected) or args.allow_cache_mismatch:
            print(f"[2/4] Reusing raw LLM score cache: {args.raw_cache}", flush=True)
            return cache
        raise ValueError(
            "Raw score cache metadata does not match this run. "
            "Use --refresh-raw-cache to regenerate or --allow-cache-mismatch to reuse it."
        )
    return generate_raw_score_cache(episodes, args, cfg)


def build_deterministic_predictions(
    episodes: Mapping[str, dict],
) -> tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, dict]]]]:
    predictions = {
        INTERACTION_NAME: {},
        SENTIMENT_NAME: {},
        STANCE_NAME: {},
    }
    extras = {
        INTERACTION_NAME: {},
        SENTIMENT_NAME: {},
        STANCE_NAME: {},
    }

    print("[1/4] Running deterministic baselines", flush=True)
    for episode_name, episode in episodes.items():
        print(
            f"  {episode_name}: {len(episode['round_end_indices'])} rounds",
            flush=True,
        )
        method_round_scores = {
            INTERACTION_NAME: [],
            SENTIMENT_NAME: [],
            STANCE_NAME: [],
        }
        method_extras = {
            INTERACTION_NAME: {},
            SENTIMENT_NAME: {},
            STANCE_NAME: {},
        }

        for round_number, end_index in enumerate(episode["round_end_indices"], start=1):
            round_logs = episode["logs"][: end_index + 1]
            interaction_scores, interaction_meta = estimate_interaction_frequency_with_metadata(
                round_logs,
                episode["participants"],
            )
            sentiment_scores = estimate_sentiment(round_logs, episode["participants"])
            stance_scores = estimate_stance(round_logs, episode["participants"])

            method_round_scores[INTERACTION_NAME].append(interaction_scores)
            method_round_scores[SENTIMENT_NAME].append(sentiment_scores)
            method_round_scores[STANCE_NAME].append(stance_scores)

            for pair, meta in interaction_meta.items():
                method_extras[INTERACTION_NAME][prediction_key(round_number, pair)] = meta

        for method, round_scores in method_round_scores.items():
            predictions[method][episode_name] = round_scores_to_predictions(round_scores)
            extras[method][episode_name] = method_extras[method]

    return predictions, extras


def build_llm_only_predictions_from_cache(
    episodes: Mapping[str, dict],
    raw_cache: Mapping[str, object],
    num_trials: int,
) -> Dict[str, Dict[str, float]]:
    predictions = {}
    for episode_name, episode in episodes.items():
        previous_by_trial = [dict() for _ in range(num_trials)]
        round_scores = []
        cache_rounds = raw_cache["episodes"][episode_name]["rounds"]
        for round_entry in cache_rounds:
            current_scores = {}
            for pair in episode["pairs"]:
                trial_values = []
                for trial_index in range(num_trials):
                    raw_scores = deserialize_scores(round_entry["trials"][trial_index])
                    if pair in raw_scores:
                        value = raw_scores[pair]
                        previous_by_trial[trial_index][pair] = value
                    else:
                        value = previous_by_trial[trial_index].get(pair, 0.0)
                    trial_values.append(value)
                current_scores[pair] = clamp_score(float(np.mean(trial_values)))
            round_scores.append(current_scores)
        predictions[episode_name] = round_scores_to_predictions(round_scores)
    return predictions


def build_sma_predictions_from_cache(
    episodes: Mapping[str, dict],
    raw_cache: Mapping[str, object],
    num_trials: int,
    window_size: int,
) -> Dict[str, Dict[str, float]]:
    predictions = {}
    for episode_name, episode in episodes.items():
        scorers = [SimpleMovingAverageScorer(window_size) for _ in range(num_trials)]
        round_scores = []
        cache_rounds = raw_cache["episodes"][episode_name]["rounds"]
        for round_entry in cache_rounds:
            current_scores = {}
            for pair in episode["pairs"]:
                trial_values = []
                for trial_index, scorer in enumerate(scorers):
                    raw_scores = deserialize_scores(round_entry["trials"][trial_index])
                    trial_values.append(scorer.update(pair, raw_scores.get(pair)))
                current_scores[pair] = clamp_score(float(np.mean(trial_values)))
            round_scores.append(current_scores)
        predictions[episode_name] = round_scores_to_predictions(round_scores)
    return predictions


def build_maverd_predictions_from_cache(
    episodes: Mapping[str, dict],
    raw_cache: Mapping[str, object],
    num_trials: int,
    gamma: float,
    max_history_sessions: int,
) -> Dict[str, Dict[str, float]]:
    predictions = {}
    for episode_name, episode in episodes.items():
        scorers = [EMAScorer(True, gamma, max_history_sessions) for _ in range(num_trials)]
        round_scores = []
        cache_rounds = raw_cache["episodes"][episode_name]["rounds"]
        for round_entry in cache_rounds:
            counts = round_entry["utterance_counts"]
            current_scores = {}
            for pair in episode["pairs"]:
                trial_values = []
                for trial_index, scorer in enumerate(scorers):
                    raw_scores = deserialize_scores(round_entry["trials"][trial_index])
                    if pair in raw_scores:
                        value = scorer.update(pair, raw_scores[pair], counts)
                    else:
                        value = scorer.scores.get(pair, 0.0)
                    trial_values.append(value)
                current_scores[pair] = clamp_score(float(np.mean(trial_values)))
            for scorer in scorers:
                scorer.finalize_round(counts)
            round_scores.append(current_scores)
        predictions[episode_name] = round_scores_to_predictions(round_scores)
    return predictions


def build_prediction_rows(
    predictions_by_method: Mapping[str, Mapping[str, Mapping[str, float]]],
    extra_by_method: Mapping[str, Mapping[str, Mapping[str, dict]]],
) -> List[Dict[str, object]]:
    rows = []
    for method, episode_predictions in predictions_by_method.items():
        for episode_name, predictions in episode_predictions.items():
            extras = extra_by_method.get(method, {}).get(episode_name, {})
            for key in sorted(predictions, key=parse_prediction_key):
                round_number, pair = parse_prediction_key(key)
                extra = dict(extras.get(key, {}))
                rows.append(
                    {
                        "Method": method,
                        "Episode": episode_name,
                        "Round": round_number,
                        "Pair": pair_to_label(pair),
                        "Prediction": predictions[key],
                        "Interaction_Intensity": extra.pop("interaction_intensity", ""),
                        "Interaction_Count": extra.pop("interaction_count", ""),
                        "Directional_Counts": json.dumps(
                            extra.pop("directional_counts", {}), ensure_ascii=False
                        ),
                        "Extra": json.dumps(extra, ensure_ascii=False, sort_keys=True),
                    }
                )
    return rows


def write_prediction_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "Method",
        "Episode",
        "Round",
        "Pair",
        "Prediction",
        "Interaction_Intensity",
        "Interaction_Count",
        "Directional_Counts",
        "Extra",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    path: str,
    args: argparse.Namespace,
    raw_cache_path: str,
    evaluation_status: str,
    evaluation_reasons: Sequence[str],
    predictions_path: str,
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "setting": "controlled_comparison",
        "raw_score_cache": raw_cache_path,
        "predictions_file": predictions_path,
        "evaluation": {
            "status": evaluation_status,
            "reasons": list(evaluation_reasons),
        },
        "llm_model": args.llm_model,
        "context_length_max_history_human": args.max_history_human,
        "num_trials": args.num_trials,
        "sma_window_k": args.sma_window,
        "maverd_gamma": args.gamma,
        "maverd_max_history_sessions": args.max_history_sessions,
        "random_seed": None,
        "methods": [
            {
                "method": INTERACTION_NAME,
                "model_or_version": INTERACTION_VERSION,
                "score_policy": "signed score is neutral because frequency has no valence; intensity saved as auxiliary metadata",
            },
            {
                "method": SENTIMENT_NAME,
                "model_or_version": SENTIMENT_VERSION,
                "input": "target-specific utterance clauses",
                "output": "pair score in [-1, 1]",
                "aggregation": "mean of directional target sentiment evidence for both directions",
            },
            {
                "method": STANCE_NAME,
                "model_or_version": STANCE_VERSION,
                "input": "target-specific utterance clauses",
                "output": "pair score in [-1, 1]",
                "aggregation": "mean of directional target stance evidence for both directions",
            },
            {
                "method": LLM_ONLY_NAME,
                "model_or_version": args.llm_model,
                "source": "existing estimate_relation_once raw_scores without smoothing",
            },
            {
                "method": SMA_NAME,
                "model_or_version": SMA_VERSION,
                "window_k": args.sma_window,
                "source": "same raw_scores as LLM-only and MAVeRD",
            },
            {
                "method": MAVERD_NAME,
                "model_or_version": "existing EMAScorer DIWS/EMA",
                "gamma": args.gamma,
                "max_history_sessions": args.max_history_sessions,
                "source": "same raw_scores as LLM-only and SMA",
            },
        ],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled baseline comparison for MAVeRD.")
    parser.add_argument("--output-dir", default="estimation_accuracy/baseline_comparison")
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Azure deployment name. Omit to use RELATION_MODEL or AZURE_MODEL.",
    )
    parser.add_argument("--max-history-human", type=int, default=6)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--sma-window", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--max-history-sessions", type=int, default=3)
    parser.add_argument("--raw-cache", default=None)
    parser.add_argument("--refresh-raw-cache", action="store_true")
    parser.add_argument("--allow-cache-mismatch", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.raw_cache = args.raw_cache or os.path.join(args.output_dir, "raw_scores_cache.json")
    os.makedirs(args.output_dir, exist_ok=True)

    relation_estimator.DEBUG = args.debug
    cfg = config.get_config()
    args.llm_model = (
        args.llm_model
        or getattr(cfg.llm, "relation_model", None)
        or getattr(cfg.llm, "azure_model", None)
    )
    if not args.llm_model:
        raise ValueError(
            "No relation model deployment is configured. Set RELATION_MODEL or AZURE_MODEL, "
            "or pass --llm-model."
        )
    print("Baseline comparison started", flush=True)
    print(f"  model: {args.llm_model}", flush=True)
    print(f"  context: {args.max_history_human} human utterances", flush=True)
    print(f"  trials: {args.num_trials}", flush=True)
    print(f"  output: {args.output_dir}", flush=True)
    episodes = load_episode_inputs(DEFAULT_EPISODES)
    episode_names = list(episodes.keys())
    human_by_episode = {name: episode["human_data"] for name, episode in episodes.items()}
    for episode_name, episode in episodes.items():
        status = "SKIPPED" if episode["evaluation_skip_reason"] else "READY"
        print(
            f"  {episode_name}: {len(episode['round_end_indices'])} rounds, "
            f"human evaluation {status}",
            flush=True,
        )

    deterministic_predictions, extra_by_method = build_deterministic_predictions(episodes)
    raw_cache = load_or_generate_raw_score_cache(episodes, args, cfg)

    print("[3/4] Building LLM-only, SMA, and MAVeRD predictions", flush=True)
    predictions_by_method = dict(deterministic_predictions)
    predictions_by_method[LLM_ONLY_NAME] = build_llm_only_predictions_from_cache(
        episodes, raw_cache, args.num_trials
    )
    predictions_by_method[SMA_NAME] = build_sma_predictions_from_cache(
        episodes, raw_cache, args.num_trials, args.sma_window
    )
    predictions_by_method[MAVERD_NAME] = build_maverd_predictions_from_cache(
        episodes, raw_cache, args.num_trials, args.gamma, args.max_history_sessions
    )

    print("[4/4] Writing result files", flush=True)
    prediction_rows = build_prediction_rows(predictions_by_method, extra_by_method)
    predictions_path = os.path.join(args.output_dir, "baseline_predictions.csv")
    write_prediction_csv(predictions_path, prediction_rows)

    evaluation_reasons = [
        episode["evaluation_skip_reason"]
        for episode in episodes.values()
        if episode["evaluation_skip_reason"]
    ]
    evaluation_status = "SKIPPED" if evaluation_reasons else "EVALUATED"
    summary_rows = []
    detail_rows = []
    if evaluation_status == "EVALUATED":
        for method, method_predictions in predictions_by_method.items():
            metrics = evaluate_predictions(method_predictions, human_by_episode)
            summary_rows.append(build_summary_row(method, metrics, episode_names))
            for episode_name in episode_names:
                detail_rows.extend(
                    build_detail_rows(
                        method,
                        episode_name,
                        method_predictions[episode_name],
                        human_by_episode[episode_name],
                        extra_by_method.get(method, {}).get(episode_name, {}),
                    )
                )
        summary_rows.sort(key=lambda row: row["MAE_Avg"])
    else:
        reason = " ".join(evaluation_reasons)
        for method in predictions_by_method:
            row: Dict[str, object] = {
                "Method": method,
                "Evaluation_Status": "SKIPPED",
                "Evaluation_Reason": reason,
            }
            for episode_name in episode_names:
                suffix = episode_suffix(episode_name)
                row[f"MAE_{suffix}"] = ""
                row[f"Pearson_{suffix}"] = ""
            row["MAE_Avg"] = ""
            row["Pearson_Avg"] = ""
            summary_rows.append(row)

    summary_path = os.path.join(args.output_dir, "baseline_summary.csv")
    detail_path = os.path.join(args.output_dir, "baseline_details.csv")
    metadata_path = os.path.join(args.output_dir, "baseline_metadata.json")
    write_summary_csv(summary_path, summary_rows, episode_names)
    write_detail_csv(detail_path, detail_rows)
    write_metadata(
        metadata_path,
        args,
        args.raw_cache,
        evaluation_status,
        evaluation_reasons,
        predictions_path,
    )

    print(f"Saved predictions: {predictions_path}")
    if evaluation_status == "SKIPPED":
        print("Evaluation skipped: " + " ".join(evaluation_reasons))
    print(f"Saved summary: {summary_path}")
    print(f"Saved details: {detail_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Raw score cache: {args.raw_cache}")


if __name__ == "__main__":
    main()
