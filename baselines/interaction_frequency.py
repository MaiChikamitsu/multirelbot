from __future__ import annotations

from collections import defaultdict
from typing import Dict, Sequence, Tuple

from .common import (
    LogEntry,
    Pair,
    iter_pairs,
    normalize_pair,
    pair_to_label,
    resolve_targets,
)

METHOD_NAME = "Interaction Frequency"
METHOD_VERSION = "structure_frequency_neutral_v1"


def count_directed_interactions(
    logs: Sequence[LogEntry],
    participants: Sequence[str],
) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for index, log in enumerate(logs):
        speaker = log.get("speaker", "")
        if speaker not in participants:
            continue
        targets, _ = resolve_targets(logs, index, participants)
        for target in targets:
            if target != speaker:
                counts[(speaker, target)] += 1
    return dict(counts)


def estimate_interaction_intensity(
    logs: Sequence[LogEntry],
    participants: Sequence[str],
) -> Dict[Pair, float]:
    directed = count_directed_interactions(logs, participants)
    pair_counts = {
        pair: directed.get((pair[0], pair[1]), 0) + directed.get((pair[1], pair[0]), 0)
        for pair in iter_pairs(participants)
    }
    max_count = max(pair_counts.values(), default=0)
    if max_count == 0:
        return {pair: 0.0 for pair in pair_counts}
    return {pair: count / max_count for pair, count in pair_counts.items()}


def estimate_interaction_frequency(
    logs: Sequence[LogEntry],
    participants: Sequence[str],
) -> Dict[Pair, float]:
    # Frequency alone does not identify positive/negative valence. The signed
    # relationship prediction is therefore neutral, while intensity is saved as
    # auxiliary evidence by estimate_interaction_frequency_with_metadata().
    return {pair: 0.0 for pair in iter_pairs(participants)}


def estimate_interaction_frequency_with_metadata(
    logs: Sequence[LogEntry],
    participants: Sequence[str],
) -> Tuple[Dict[Pair, float], Dict[Pair, Dict[str, object]]]:
    directed = count_directed_interactions(logs, participants)
    intensities = estimate_interaction_intensity(logs, participants)
    scores = estimate_interaction_frequency(logs, participants)
    metadata = {}
    for pair in iter_pairs(participants):
        ab = directed.get((pair[0], pair[1]), 0)
        ba = directed.get((pair[1], pair[0]), 0)
        metadata[pair] = {
            "interaction_count": ab + ba,
            "interaction_intensity": intensities[pair],
            "directional_counts": {
                pair_to_label((pair[0], pair[1])): ab,
                pair_to_label((pair[1], pair[0])): ba,
            },
            "valence_policy": "neutral_signed_score_intensity_auxiliary_only",
        }
    return scores, metadata

