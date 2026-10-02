from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


Pair = Tuple[str, str]


def detect_ingroup_outgroup(
    logs: Iterable[Dict[str, Any]],
    relation_scores: Dict[Pair, float],
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """Infer simple in-group / out-group structure from pair relation scores."""
    participants = sorted(
        {
            log.get("speaker")
            for log in logs
            if log.get("speaker") and log.get("speaker") != "ロボット"
        }
    )
    ingroup_pairs: List[Pair] = []
    outgroup_pairs: List[Pair] = []

    for pair, score in relation_scores.items():
        normalized_pair = tuple(sorted(pair))
        if score >= threshold:
            ingroup_pairs.append(normalized_pair)
        else:
            outgroup_pairs.append(normalized_pair)

    return {
        "participants": participants,
        "ingroup_pairs": sorted(set(ingroup_pairs)),
        "outgroup_pairs": sorted(set(outgroup_pairs)),
        "threshold": threshold,
    }
