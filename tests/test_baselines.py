from __future__ import annotations

import math

from baselines.common import iter_pairs, normalize_pair
from baselines.interaction_frequency import (
    estimate_interaction_frequency,
    estimate_interaction_frequency_with_metadata,
)
from baselines.sentiment import estimate_sentiment
from baselines.sma import SimpleMovingAverageScorer, smooth_raw_score_rounds
from baselines.stance import estimate_stance
from evaluation_utils import prediction_key, round_scores_to_predictions
from evaluate_baselines import (
    build_llm_only_predictions_from_cache,
    deserialize_scores,
    serialize_scores,
)
from relation_estimator_from_txt import EMAScorer, split_into_rounds


PARTICIPANTS = ["A", "B", "C"]


def test_pair_generation_and_normalization():
    assert iter_pairs(PARTICIPANTS) == [("A", "B"), ("A", "C"), ("B", "C")]
    assert normalize_pair("C", "A") == ("A", "C")


def test_round_split_matches_existing_maverd_timing():
    logs = [
        {"speaker": "A", "utterance": "a1"},
        {"speaker": "B", "utterance": "b1"},
        {"speaker": "C", "utterance": "c1"},
        {"speaker": "ロボット", "utterance": "r1"},
        {"speaker": "A", "utterance": "a2"},
        {"speaker": "B", "utterance": "b2"},
        {"speaker": "C", "utterance": "c2"},
    ]
    assert split_into_rounds(logs, PARTICIPANTS) == [2, 6]


def test_all_rule_based_scores_stay_in_range():
    logs = [
        {"speaker": "A", "utterance": "Bの案は最高。でもCの案は嫌い。"},
        {"speaker": "B", "utterance": "Aの言う通りだと思う。"},
        {"speaker": "C", "utterance": "Bには反対。"},
    ]
    for estimator in [estimate_interaction_frequency, estimate_sentiment, estimate_stance]:
        scores = estimator(logs, PARTICIPANTS)
        assert set(scores) == set(iter_pairs(PARTICIPANTS))
        assert all(-1.0 <= value <= 1.0 for value in scores.values())


def test_interaction_frequency_keeps_signed_score_neutral_and_saves_intensity():
    logs = [
        {"speaker": "A", "utterance": "Bはどう思う？"},
        {"speaker": "B", "utterance": "Aに答えるよ。"},
        {"speaker": "C", "utterance": "なるほど。"},
    ]
    scores, metadata = estimate_interaction_frequency_with_metadata(logs, PARTICIPANTS)
    assert scores[("A", "B")] == 0.0
    assert metadata[("A", "B")]["interaction_count"] == 2
    assert metadata[("A", "B")]["interaction_intensity"] == 1.0


def test_sentiment_distinguishes_targets_in_same_utterance():
    logs = [{"speaker": "A", "utterance": "Bの案は最高。でもCの案は嫌い。"}]
    scores = estimate_sentiment(logs, PARTICIPANTS)
    assert scores[("A", "B")] > 0
    assert scores[("A", "C")] < 0


def test_sentiment_and_stance_are_separate_target_specific_signals():
    logs = [{"speaker": "A", "utterance": "Bのことは好きだけど、それには反対。"}]
    assert estimate_sentiment(logs, PARTICIPANTS)[("A", "B")] > 0
    assert estimate_stance(logs, PARTICIPANTS)[("A", "B")] < 0

    logs = [{"speaker": "A", "utterance": "Bは好きじゃないけど、その意見自体は正しい。"}]
    assert estimate_sentiment(logs, PARTICIPANTS)[("A", "B")] < 0
    assert estimate_stance(logs, PARTICIPANTS)[("A", "B")] > 0


def test_sma_window_and_missing_score_handling():
    scorer = SimpleMovingAverageScorer(window_size=2)
    assert scorer.update(("A", "B"), 0.2) == 0.2
    assert scorer.update(("A", "B"), 0.6) == 0.4
    assert scorer.update(("A", "B"), None) == 0.4
    assert math.isclose(scorer.update(("A", "B"), -0.2), 0.2)

    rounds = smooth_raw_score_rounds(
        [
            {("A", "B"): 0.0},
            {("A", "B"): 1.0},
            {},
            {("A", "B"): -1.0},
        ],
        [("A", "B")],
        window_size=2,
    )
    assert [round_scores[("A", "B")] for round_scores in rounds] == [0.0, 0.5, 0.5, 0.0]


def test_llm_only_predictions_equal_raw_scores_when_present():
    episodes = {
        "Episode1": {
            "pairs": [("A", "B")],
        }
    }
    raw_cache = {
        "episodes": {
            "Episode1": {
                "rounds": [
                    {"trials": [serialize_scores({("A", "B"): 0.3})]},
                    {"trials": [serialize_scores({("A", "B"): -0.7})]},
                ]
            }
        }
    }
    predictions = build_llm_only_predictions_from_cache(episodes, raw_cache, num_trials=1)
    assert predictions["Episode1"] == {"1-A-B": 0.3, "2-A-B": -0.7}


def test_score_serialization_round_trip():
    scores = {("C", "A"): 0.5, ("A", "B"): -0.25}
    round_trip = deserialize_scores(serialize_scores(scores))
    assert round_trip == {("A", "C"): 0.5, ("A", "B"): -0.25}


def test_existing_ema_formula_regression():
    scorer = EMAScorer(use_ema=True, gamma=0.5, max_history_sessions=3)
    pair = ("A", "B")
    first = scorer.update(pair, 0.5, {"A": 1, "B": 1})
    scorer.finalize_round({"A": 1, "B": 1})
    second = scorer.update(pair, -0.5, {"A": 2, "B": 2})

    assert first == 0.5
    assert math.isclose(second, -1.0 / 6.0, rel_tol=1e-9)


def test_round_scores_to_predictions_uses_existing_pair_key_shape():
    predictions = round_scores_to_predictions([{("A", "B"): 0.1}])
    assert predictions[prediction_key(1, ("A", "B"))] == 0.1
