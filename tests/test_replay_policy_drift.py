from __future__ import annotations

import numpy as np

from training.replay_policy_drift import (
    PRIMARY_METRICS,
    _action_payload,
    _block_bootstrap_intervals,
    _distribution_metrics,
)


def test_distribution_metrics_exact_ragged_probabilities():
    behavior = {
        "flat_probs": np.array([0.5, 0.5, 0.8, 0.1, 0.1], dtype=np.float32),
        "selected_log_probs": np.log(
            np.array([0.5, 0.1], dtype=np.float32)
        ),
        "entropy": np.array([0.6931472, 0.6390319], dtype=np.float32),
        "argmax": np.array([0, 0], dtype=np.int32),
    }
    candidate = {
        "flat_probs": np.array([0.75, 0.25, 0.8, 0.1, 0.1], dtype=np.float32),
        "selected_log_probs": np.log(
            np.array([0.75, 0.1], dtype=np.float32)
        ),
        "entropy": np.array([0.5623351, 0.6390319], dtype=np.float32),
        "argmax": np.array([0, 0], dtype=np.int32),
    }
    rollout = {
        "legal_lens": np.array([2, 3], dtype=np.int32),
        "action_idxs": np.array([0, 1], dtype=np.int64),
        "log_probs": behavior["selected_log_probs"].copy(),
        "advantages": np.array([1.0, -1.0], dtype=np.float32),
    }

    metrics = _distribution_metrics(behavior, candidate, rollout, 0.2)

    np.testing.assert_allclose(metrics["ratio"], [1.5, 1.0], rtol=1e-6)
    np.testing.assert_allclose(metrics["total_variation"], [0.25, 0.0])
    np.testing.assert_allclose(
        metrics["chosen_probability_absolute_change"], [0.25, 0.0]
    )
    np.testing.assert_array_equal(metrics["outside_clip_range"], [1.0, 0.0])
    np.testing.assert_array_equal(
        metrics["ppo_objective_saturated"], [1.0, 0.0]
    )
    assert metrics["kl_behavior_to_candidate"][0] > 0
    assert metrics["kl_behavior_to_candidate"][1] == 0


def test_constant_block_bootstrap_intervals_are_exact():
    game_ids = np.repeat(np.arange(8), 2)
    metrics = {
        metric: np.full(len(game_ids), index + 0.25, dtype=np.float64)
        for index, metric in enumerate(PRIMARY_METRICS)
    }

    result = _block_bootstrap_intervals(
        [metrics],
        game_ids,
        games_per_inference_block=2,
        samples=200,
        alpha=0.05,
        seed=17,
    )

    for index, metric in enumerate(PRIMARY_METRICS):
        expected = index + 0.25
        assert result[(0, metric)]["mean"] == expected
        assert result[(0, metric)]["ordinary_ci"] == [expected, expected]
        assert result[(0, metric)]["simultaneous_ci"] == [expected, expected]


def test_action_payload_decodes_cards_and_flags():
    vector = np.zeros(58, dtype=np.uint8)
    vector[[0, 54, 56, 57]] = 1

    assert _action_payload(vector) == {
        "cards": ["2g", "Phoenix"],
        "call_tichu": True,
        "phoenix_high": True,
    }
