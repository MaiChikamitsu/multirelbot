from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .common import (
    LogEntry,
    Pair,
    clamp_score,
    iter_pairs,
    mean_or_zero,
    resolve_targets,
    score_text_with_lexicon,
    target_specific_texts,
)

METHOD_NAME = "Stance-based"
METHOD_VERSION = "target_lexicon_rule_based_v1"

SUPPORT_PHRASES = [
    "その通り",
    "言う通り",
    "いいと思う",
    "良いと思う",
    "悪くないと思う",
    "それならいい",
    "確かに",
]

OPPOSE_PHRASES = [
    "納得いかない",
    "そうじゃない",
    "それは違う",
    "そうではない",
    "何勝手に",
]

SUPPORT_TERMS = [
    "賛成",
    "同意",
    "支持",
    "正しい",
    "合って",
    "わかる",
    "分かる",
    "納得",
    "共感",
    "あり",
]

OPPOSE_TERMS = [
    "反対",
    "違う",
    "ちがう",
    "否定",
    "間違",
    "だめ",
    "ダメ",
    "いや",
    "は？",
    "決めつけ",
    "押しつけ",
    "ケチつけ",
]


def score_stance_text(text: str) -> float:
    return score_text_with_lexicon(
        text,
        SUPPORT_TERMS,
        OPPOSE_TERMS,
        positive_phrases=SUPPORT_PHRASES,
        negative_phrases=OPPOSE_PHRASES,
    )


def estimate_directional_stance(
    logs: Sequence[dict],
    participants: Sequence[str],
) -> Dict[Tuple[str, str], List[float]]:
    directional: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for index, log in enumerate(logs):
        speaker = log.get("speaker", "")
        if speaker not in participants:
            continue

        utterance = log.get("utterance", "")
        targets, explicit = resolve_targets(logs, index, participants)
        explicit_targets = targets if explicit else []

        for target in targets:
            if target == speaker:
                continue
            texts = target_specific_texts(utterance, target, explicit_targets)
            score = mean_or_zero(score_stance_text(text) for text in texts)
            if score != 0.0:
                directional[(speaker, target)].append(score)

    return dict(directional)


def estimate_stance(
    logs: Sequence[LogEntry],
    participants: Sequence[str],
) -> Dict[Pair, float]:
    directional = estimate_directional_stance(logs, participants)
    pair_scores = {}
    for pair in iter_pairs(participants):
        values = directional.get((pair[0], pair[1]), []) + directional.get((pair[1], pair[0]), [])
        pair_scores[pair] = clamp_score(mean_or_zero(values))
    return pair_scores

