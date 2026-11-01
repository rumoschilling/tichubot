"""Regression coverage for the standalone passing collector."""

import multiprocessing as mp

import numpy as np
import pytest
import torch

from training import rollout_passing
from training.config import TrainConfig
from training.model import TichuPolicy
from training.model_embed import TichuPolicyEmbed
from training.model_gt import TichuGrandTichuPolicy
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder
from training.model_passing import TichuPassingPolicy


class ImmediatePool:
    def map(self, function, arguments):
        self.sizes = [args[3] for args in arguments]
        return [function(args) for args in arguments]


def test_passing_collection_skips_empty_worker_assignments():
    pool = ImmediatePool()
    result = rollout_passing.collect_passing_rollouts(
        TichuPassingPolicy(), TichuPolicyEmbed(), 1, torch.device("cpu"),
        n_workers=4, pool=pool,
    )
    assert pool.sizes == [1]
    assert result["hand_cards"].shape == (4, 14)
    assert all(len(value) == 4 for value in result.values())
    assert np.isfinite(result["rewards"]).all()


@pytest.mark.parametrize("kind", ["fixed", "embed", "history"])
def test_passing_workers_preserve_loaded_models_instead_of_training_defaults(monkeypatch, kind):
    if kind == "fixed":
        value = TichuPolicy(hidden_dim=16, n_layers=3)
    elif kind == "embed":
        value = TichuPolicyEmbed(hidden_dim=16, n_layers=3, embed_dim=8)
    else:
        value = TichuPolicyHistory(
            TransformerHistoryEncoder(d_model=8, nhead=2, n_layers=1, ffn_dim=16),
            hidden_dim=16, trunk_dims=(32, 16), mlp_activation="silu",
            mlp_layer_norm=True, embed_dim=8,
        )
    passing = TichuPassingPolicy(hidden_dim=8, n_layers=1, card_embed_dim=4)
    gt = TichuGrandTichuPolicy(hidden_dim=8, n_layers=1)
    observations = torch.randn(7, 196)
    expected = rollout_passing._get_values(value, observations, torch.device("cpu"))
    checked = []

    def check_models(restored_pass, restored_value, n_deals, device, gt_policy=None):
        assert n_deals == 1
        for original, restored in ((passing, restored_pass), (gt, gt_policy)):
            for key, weight in original.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], weight)
        actual = rollout_passing._get_values(restored_value, observations, device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        checked.append(True)
        return {"rewards": np.zeros(4, dtype=np.float32)}

    monkeypatch.setattr(rollout_passing, "_collect_passing_batched", check_models)
    result = rollout_passing.collect_passing_rollouts(
        passing, value, 2, torch.device("cpu"), n_workers=2,
        pool=ImmediatePool(), cfg=TrainConfig(), gt_policy=gt,
    )
    assert checked == [True, True]
    assert result["rewards"].shape == (8,)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_passing_collection_supports_counted_value_observations(n_workers):
    value = TichuPolicyEmbed(obs_size=200, hidden_dim=16, n_layers=1, embed_dim=8)
    passing = TichuPassingPolicy(hidden_dim=8, n_layers=1)
    result = rollout_passing.collect_passing_rollouts(
        passing, value, 2, torch.device("cpu"), n_workers=n_workers,
        pool=ImmediatePool(),
    )
    assert result["hand_cards"].shape == (8, 14)
    assert np.isfinite(result["rewards"]).all()


def test_passing_collection_round_trips_models_through_spawn_workers():
    value = TichuPolicyEmbed(obs_size=200, hidden_dim=16, n_layers=1, embed_dim=8)
    passing = TichuPassingPolicy(hidden_dim=8, n_layers=1, card_embed_dim=4)
    gt = TichuGrandTichuPolicy(hidden_dim=8, n_layers=1)
    with mp.get_context("spawn").Pool(2) as pool:
        result = rollout_passing.collect_passing_rollouts(
            passing, value, 2, torch.device("cpu"), n_workers=2,
            pool=pool, gt_policy=gt,
        )
    assert result["hand_cards"].shape == (8, 14)
    assert all(len(rows) == 8 for rows in result.values())
    assert np.isfinite(result["rewards"]).all()
