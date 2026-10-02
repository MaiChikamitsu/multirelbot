from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from baselines.common import Pair, clamp_score, label_to_pair, pair_to_label


@dataclass(frozen=True)
class EpisodeConfig:
    name: str
    conversation_file: str
    human_file: str


DEFAULT_EPISODES = [
    EpisodeConfig(
        name="Episode1",
        conversation_file="estimation_accuracy/conversation1.txt",
        human_file="estimation_accuracy/human1.csv",
    )
    # EpisodeConfig(
    #     name="Episode2",
    #     conversation_file="estimation_accuracy/conversation2.txt",
    #     human_file="estimation_accuracy/human2.csv",
    # ),
]


def load_human_data(file_path: str) -> Dict[str, float]:
    human_data = {}
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            parts = row[0].split("-")
            if len(parts) < 4:
                continue
            section = parts[1]
            pair_label = pair_to_label(tuple(sorted(parts[2:])))
            human_data[f"{section}-{pair_label}"] = float(row[4])
    return human_data


def prediction_key(round_number: int, pair: Pair) -> str:
    return f"{round_number}-{pair_to_label(pair)}"


def parse_prediction_key(key: str) -> Tuple[int, Pair]:
    round_text, a, b = key.split("-", 2)
    return int(round_text), label_to_pair(f"{a}-{b}")


def calculate_mae(predictions: Mapping[str, float], ground_truth: Mapping[str, float]) -> float:
    errors = [
        abs(float(predictions[key]) - float(ground_truth[key]))
        for key in ground_truth
        if key in predictions
    ]
    if not errors:
        return float("inf")
    return float(np.mean(errors))


def calculate_pearson(predictions: Mapping[str, float], ground_truth: Mapping[str, float]) -> float:
    pred_values = []
    true_values = []
    for key in ground_truth:
        if key in predictions:
            pred_values.append(float(predictions[key]))
            true_values.append(float(ground_truth[key]))

    if len(pred_values) < 2:
        return 0.0
    if np.std(pred_values) == 0 or np.std(true_values) == 0:
        return 0.0
    correlation, _ = stats.pearsonr(pred_values, true_values)
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def round_scores_to_predictions(
    round_scores: Sequence[Mapping[Pair, float]],
) -> Dict[str, float]:
    predictions = {}
    for round_index, scores in enumerate(round_scores, start=1):
        for pair, score in scores.items():
            predictions[prediction_key(round_index, pair)] = clamp_score(score)
    return predictions


def evaluate_predictions(
    predictions_by_episode: Mapping[str, Mapping[str, float]],
    human_by_episode: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for episode_name, human_data in human_by_episode.items():
        predictions = predictions_by_episode.get(episode_name, {})
        metrics[episode_name] = {
            "mae": calculate_mae(predictions, human_data),
            "pearson": calculate_pearson(predictions, human_data),
        }
    return metrics


def episode_suffix(episode_name: str) -> str:
    if episode_name.lower().startswith("episode"):
        return f"Ep{episode_name[len('Episode'):]}"
    return episode_name


def build_summary_row(
    method: str,
    metrics_by_episode: Mapping[str, Mapping[str, float]],
    episode_names: Sequence[str],
) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {
        "Method": method,
        "Evaluation_Status": "EVALUATED",
        "Evaluation_Reason": "",
    }
    maes = []
    pearsons = []
    for episode_name in episode_names:
        suffix = episode_suffix(episode_name)
        metrics = metrics_by_episode[episode_name]
        row[f"MAE_{suffix}"] = metrics["mae"]
        row[f"Pearson_{suffix}"] = metrics["pearson"]
        maes.append(metrics["mae"])
        pearsons.append(metrics["pearson"])
    row["MAE_Avg"] = float(np.mean(maes)) if maes else float("inf")
    row["Pearson_Avg"] = float(np.mean(pearsons)) if pearsons else 0.0
    return row


def build_detail_rows(
    method: str,
    episode_name: str,
    predictions: Mapping[str, float],
    human_data: Mapping[str, float],
    extra_by_key: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> List[Dict[str, object]]:
    rows = []
    extra_by_key = extra_by_key or {}
    for key, human_value in human_data.items():
        if key not in predictions:
            continue
        round_number, pair = parse_prediction_key(key)
        prediction = clamp_score(predictions[key])
        extra = dict(extra_by_key.get(key, {}))
        rows.append(
            {
                "Method": method,
                "Episode": episode_name,
                "Round": round_number,
                "Pair": pair_to_label(pair),
                "Prediction": prediction,
                "Human_Ground_Truth": human_value,
                "Error": abs(prediction - human_value),
                "Interaction_Intensity": extra.pop("interaction_intensity", ""),
                "Interaction_Count": extra.pop("interaction_count", ""),
                "Directional_Counts": json.dumps(extra.pop("directional_counts", {}), ensure_ascii=False),
                "Extra": json.dumps(extra, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def write_summary_csv(path: str, rows: Sequence[Mapping[str, object]], episode_names: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["Method", "Evaluation_Status", "Evaluation_Reason"]
    for episode_name in episode_names:
        suffix = episode_suffix(episode_name)
        fieldnames.extend([f"MAE_{suffix}", f"Pearson_{suffix}"])
    fieldnames.extend(["MAE_Avg", "Pearson_Avg"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_detail_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "Method",
        "Episode",
        "Round",
        "Pair",
        "Prediction",
        "Human_Ground_Truth",
        "Error",
        "Interaction_Intensity",
        "Interaction_Count",
        "Directional_Counts",
        "Extra",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
