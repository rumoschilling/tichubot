from __future__ import annotations

import torch

from training.architecture_runtime_experiment import (
    _analyze_family,
    _architecture_config,
)
from training.train_history import _build_policy


def test_architecture_runtime_holds_training_batches_fixed():
    device = torch.device("cpu")
    standard = _architecture_config("standard", device)
    compact = _architecture_config("compact", device)

    for cfg in (standard, compact):
        assert cfg.games_per_update == 1024
        assert cfg.batch_size == 512
        assert cfg.optimizer_batch_size == 2048
        assert cfg.n_epochs == 2
        assert cfg.synchronous
        assert cfg.profile_update_timings

    assert not standard.split_actor_critic_history
    assert not standard.static_attention
    assert compact.split_actor_critic_history
    assert compact.static_attention


def test_compact_runtime_policy_really_has_fewer_parameters():
    device = torch.device("cpu")
    standard = _build_policy(_architecture_config("standard", device))
    compact = _build_policy(_architecture_config("compact", device))

    standard_parameters = sum(parameter.numel() for parameter in standard.parameters())
    compact_parameters = sum(parameter.numel() for parameter in compact.parameters())
    assert compact_parameters < standard_parameters


def test_paired_analysis_uses_independent_blocks_and_holm_adjustment():
    measurements = []
    for block in range(20):
        jitter = 1.0 + (block % 5) * 0.002
        measurements.append(
            {
                "standard": {"ppo_seconds": 2.0 * jitter},
                "compact": {"ppo_seconds": 1.0 * jitter},
            }
        )

    result = _analyze_family(
        measurements,
        ["ppo_seconds"],
        bootstrap_samples=2_000,
        permutation_samples=10_000,
        seed=7,
        alpha=0.05,
        equivalence_margin=0.02,
    )[0]

    assert result["paired_blocks"] == 20
    assert result["geometric_speedup_standard_over_compact"] == 2.0
    assert result["bootstrap_95_percent_speedup_ci"] == [2.0, 2.0]
    assert result["significant_after_holm"]
    assert result["direction_after_holm"] == "compact_faster"
