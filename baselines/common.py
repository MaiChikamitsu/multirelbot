from __future__ import annotations

import re
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Pair = Tuple[str, str]
LogEntry = Dict[str, str]

ROBOT_SPEAKER = "ロボット"


def clamp_score(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def normalize_pair(a: str, b: str) -> Pair:
    return tuple(sorted((a, b)))


def iter_pairs(participants: Sequence[str]) -> List[Pair]:
    return [tuple(pair) for pair in combinations(sorted(participants), 2)]


def pair_to_label(pair: Pair) -> str:
    return f"{pair[0]}-{pair[1]}"


def label_to_pair(label: str) -> Pair:
    a, b = label.split("-", 1)
    return normalize_pair(a, b)


def is_human_speaker(speaker: str, participants: Sequence[str]) -> bool:
    return speaker in participants and speaker != ROBOT_SPEAKER


def split_clauses(text: str) -> List[str]:
    clauses = re.split(r"[。．.!！?？\n]+|(?:でも|だけど|けど|ただし|一方で)", text)
    return [clause.strip(" 、,") for clause in clauses if clause.strip(" 、,")]


def _participant_pattern(participant: str) -> re.Pattern[str]:
    escaped = re.escape(participant)
    return re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?:さん|君|ちゃん|くん)?(?![A-Za-z0-9_])")


def mentioned_participants(
    utterance: str,
    participants: Sequence[str],
    speaker: Optional[str] = None,
) -> List[str]:
    targets = []
    for participant in sorted(participants):
        if participant == speaker:
            continue
        if _participant_pattern(participant).search(utterance):
            targets.append(participant)
    return targets


def previous_human_speaker(
    logs: Sequence[LogEntry],
    current_index: int,
    participants: Sequence[str],
) -> Optional[str]:
    for previous in reversed(logs[:current_index]):
        speaker = previous.get("speaker", "")
        if is_human_speaker(speaker, participants):
            return speaker
    return None


def resolve_targets(
    logs: Sequence[LogEntry],
    current_index: int,
    participants: Sequence[str],
) -> Tuple[List[str], bool]:
    log = logs[current_index]
    speaker = log.get("speaker", "")
    utterance = log.get("utterance", "")
    explicit_targets = mentioned_participants(utterance, participants, speaker=speaker)
    if explicit_targets:
        return explicit_targets, True

    previous = previous_human_speaker(logs, current_index, participants)
    if previous and previous != speaker:
        return [previous], False
    return [], False


def target_specific_texts(
    utterance: str,
    target: str,
    explicit_targets: Sequence[str],
) -> List[str]:
    if len(explicit_targets) <= 1:
        return [utterance]

    clauses = split_clauses(utterance)
    target_clauses = [
        clause
        for clause in clauses
        if mentioned_participants(clause, explicit_targets, speaker=None) == [target]
        or target in mentioned_participants(clause, explicit_targets, speaker=None)
    ]
    return target_clauses


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def score_text_with_lexicon(
    text: str,
    positive_terms: Sequence[str],
    negative_terms: Sequence[str],
    positive_phrases: Sequence[str] = (),
    negative_phrases: Sequence[str] = (),
) -> float:
    positive = 0
    negative = 0
    masked_text = text

    for phrase in positive_phrases:
        if phrase in masked_text:
            positive += 1
            masked_text = masked_text.replace(phrase, " ")

    for phrase in negative_phrases:
        if phrase in masked_text:
            negative += 1
            masked_text = masked_text.replace(phrase, " ")

    for term in positive_terms:
        positive += masked_text.count(term)

    for term in negative_terms:
        negative += masked_text.count(term)

    total = positive + negative
    if total == 0:
        return 0.0
    return clamp_score((positive - negative) / total)

