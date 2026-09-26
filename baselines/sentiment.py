from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .common import (
    LogEntry,
    Pair,
    clamp_score,
    iter_pairs,
    mean_or_zero,
    normalize_pair,
    resolve_targets,
    score_text_with_lexicon,
    target_specific_texts,
)

METHOD_NAME = "Sentiment-based"
METHOD_VERSION = "target_lexicon_rule_based_v1"

POSITIVE_PHRASES = [
    "悪くない",
    "大歓迎",
    "言ってくれるなら",
    "わかってくれる",
    "分かってくれる",
]

NEGATIVE_PHRASES = [
    "好きじゃない",
    "好きではない",
    "よくない",
    "良くない",
    "面白くない",
    "楽しくない",
    "納得いかない",
]

POSITIVE_TERMS = [
    "好き",
    "最高",
    "いい",
    "良い",
    "ありがとう",
    "ありがと",
    "嬉",
    "楽しい",
    "楽しみ",
    "共感",
    "わかる",
    "分かる",
    "認め",
    "すご",
    "助か",
    "面白",
    "安心",
    "盛り上が",
]

NEGATIVE_TERMS = [
    "嫌い",
    "嫌",
    "ムカつ",
    "イライラ",
    "うるさい",
    "面倒",
    "しんどい",
    "文句",
    "ケチ",
    "偏見",
    "勝手",
    "否定",
    "突っかか",
    "最悪",
    "不満",
    "不快",
    "攻撃",
]


def score_sentiment_text(text: str) -> float:
    return score_text_with_lexicon(
        text,
        POSITIVE_TERMS,
        NEGATIVE_TERMS,
        positive_phrases=POSITIVE_PHRASES,
        negative_phrases=NEGATIVE_PHRASES,
    )


def estimate_directional_sentiment(
    logs: Sequence[LogEntry],
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
            score = mean_or_zero(score_sentiment_text(text) for text in texts)
            if score != 0.0:
                directional[(speaker, target)].append(score)

    return dict(directional)


def estimate_sentiment(
    logs: Sequence[LogEntry],
    participants: Sequence[str],
) -> Dict[Pair, float]:
    directional = estimate_directional_sentiment(logs, participants)
    pair_scores = {}
    for pair in iter_pairs(participants):
        values = directional.get((pair[0], pair[1]), []) + directional.get((pair[1], pair[0]), [])
        pair_scores[pair] = clamp_score(mean_or_zero(values))
    return pair_scores

