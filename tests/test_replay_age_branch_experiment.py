from __future__ import annotations

import numpy as np
import pytest

from training.replay_age_branch_experiment import (
    _crossed_bootstrap,
    _metric_block_sums,
    _parse_ages,
    _rescore_max_tolerance,
)
from training.replay_policy_drift import PRIMARY_METRICS


def test_parse_ages_covers_requested_integer_curve():
    assert _parse_ages("2,3,4,5,6,7,8,9,10", 10) == list(range(2, 11))
    with pytest.raises(ValueError):
        _parse_ages("0,2", 10)
    with pytest.raises(ValueError):
        _parse_ages("2,11", 10)


def test_rescore_maximum_tolerance_accounts_for_cpu_rebatching():
    import torch

    assert _rescore_max_tolerance(torch.device("cpu")) == 0.05
    assert _rescore_max_tolerance(torch.device("cuda")) == 0.02


def test_metric_block_sums_clusters_discretionary_states_by_game():
    game_ids = np.repeat(np.arange(4), 2)
    discretionary = np.array([True, False, True, True, False, True, True, True])
    metrics = {
        metric: np.full(len(game_ids), index + 1.0)
        for index, metric in enumerate(PRIMARY_METRICS)
    }

    sums, counts = _metric_block_sums(
        metrics,
        game_ids,
        discretionary,
        games_per_block=2,
        block_count=2,
    )

    np.testing.assert_array_equal(counts, [3.0, 3.0])
    np.testing.assert_array_equal(sums[:, 0], [3.0, 3.0])
    np.testing.assert_array_equal(sums[:, -1], [27.0, 27.0])


def test_crossed_bootstrap_constant_branches_and_blocks_is_exact():
    branches, ages, blocks, metrics = 3, 2, 4, len(PRIMARY_METRICS)
    counts = np.full(blocks, 5.0)
    values = np.arange(1, ages * metrics + 1, dtype=np.float64).reshape(
        ages, metrics
    )
    sums = np.broadcast_to(
        values[None, :, None, :] * counts[None, None, :, None],
        (branches, ages, blocks, metrics),
    ).copy()

    branch_means, draws, result = _crossed_bootstrap(
        sums,
        counts,
        samples=200,
        alpha=0.05,
        seed=9,
    )

    np.testing.assert_allclose(
        branch_means, np.broadcast_to(values, branch_means.shape)
    )
    np.testing.assert_allclose(draws, np.broadcast_to(values, draws.shape))
    np.testing.assert_allclose(result["point"], values)
    np.testing.assert_allclose(result["ordinary_ci"][..., 0], values)
    np.testing.assert_allclose(result["ordinary_ci"][..., 1], values)
    np.testing.assert_allclose(result["simultaneous_ci"][..., 0], values)
    np.testing.assert_allclose(result["simultaneous_ci"][..., 1], values)
