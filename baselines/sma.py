from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence

from .common import Pair, clamp_score

METHOD_NAME = "LLM + SMA"
METHOD_VERSION = "simple_moving_average_v1"


class SimpleMovingAverageScorer:
    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.history: Dict[Pair, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, pair: Pair, raw_score: Optional[float]) -> float:
        if raw_score is not None:
            self.history[pair].append(clamp_score(raw_score))
        if not self.history[pair]:
            return 0.0
        return clamp_score(sum(self.history[pair]) / len(self.history[pair]))


def smooth_raw_score_rounds(
    raw_score_rounds: Sequence[Dict[Pair, float]],
    pairs: Iterable[Pair],
    window_size: int,
) -> List[Dict[Pair, float]]:
    scorer = SimpleMovingAverageScorer(window_size)
    smoothed_rounds = []
    pair_list = list(pairs)
    for raw_scores in raw_score_rounds:
        round_scores = {}
        for pair in pair_list:
            round_scores[pair] = scorer.update(pair, raw_scores.get(pair))
        smoothed_rounds.append(round_scores)
    return smoothed_rounds

