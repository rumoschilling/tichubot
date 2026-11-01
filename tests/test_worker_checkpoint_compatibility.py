"""Regression checks for dispatch isolation and checkpoint reconstruction."""

import numpy as np
import pytest
import torch

from training import rollout, rollout_history
from training.config import TrainConfig
from training.model import TichuPolicy
from training.model_embed import TichuPolicyEmbed
from training.model_gt import TichuGrandTichuPolicy
from training.model_passing import TichuPassingPolicy
from training.train_embed import load_checkpoint, load_gt_checkpoint
from training.train_history import _build_policy, _encoder_cfg, _model_cfg


class DeferredPool:
    def map_async(self, function, arguments):
        self.function = function
        self.arguments = arguments
        return self


def tiny_config():
    return TrainConfig(
        hidden_dim=16, n_layers=1, embed_dim=8,
        hist_d_model=8, hist_nhead=2, hist_n_layers=1, hist_ffn_dim=16,
    )


def dispatch(policy, cfg, gt, passing):
    pool = DeferredPool()
    result = rollout_history.collect_rollouts_history(
        policy, 2, torch.device("cpu"), pool=pool, n_workers=2,
        encoder_cfg=_encoder_cfg(cfg), model_cfg=_model_cfg(cfg),
        gt_policy=gt, pass_policy=passing, block=False,
    )
    assert result is pool
    return pool


def test_async_dispatch_freezes_all_three_policy_snapshots():
    cfg = tiny_config()
    models = (
        _build_policy(cfg),
        TichuGrandTichuPolicy(hidden_dim=8, n_layers=1),
        TichuPassingPolicy(hidden_dim=8, n_layers=1),
    )
    pool = dispatch(models[0], cfg, models[1], models[2])
    snapshots = pool.arguments[0][:3]
    expected = [{key: value.copy() for key, value in state.items()}
                for state in snapshots]

    with torch.no_grad():
        for model in models:
            for parameter in model.parameters():
                parameter.add_(1)

    for model, snapshot, original in zip(models, snapshots, expected):
        for key, value in snapshot.items():
            np.testing.assert_array_equal(value, original[key])
            assert not np.shares_memory(value, model.state_dict()[key].numpy())


def test_history_worker_restores_specialist_shapes_and_legacy_wish_head(monkeypatch):
    monkeypatch.setattr(rollout_history, "_WORKER_MODELS", {})
    cfg = tiny_config()
    expected_models = []
    checked = []

    def check_collect(policy, _n_games, device, **kwargs):
        actual = (policy, kwargs["gt_policy"], kwargs["pass_policy"])
        for source, restored in zip(expected_models, actual):
            assert source.state_dict().keys() == restored.state_dict().keys()
            for key, weight in source.state_dict().items():
                torch.testing.assert_close(weight, restored.state_dict()[key])
            assert next(restored.parameters()).device == device
            assert not restored.training
        checked.append(True)
        return {"checked": np.array([1])}, {}

    monkeypatch.setattr(rollout_history, "_collect_batched", check_collect)
    # Exercise two different specialist architectures in the same worker
    # cache, including switching from a legacy to a current wish head.
    for width, depth, wish_outputs in ((8, 1, 13), (16, 3, 14)):
        policy = _build_policy(cfg)
        policy.wish_head[4] = torch.nn.Linear(32, wish_outputs)
        gt = TichuGrandTichuPolicy(hidden_dim=width, n_layers=depth)
        passing = TichuPassingPolicy(
            hidden_dim=width, n_layers=depth, card_embed_dim=width,
        )
        expected_models[:] = [policy, gt, passing]
        pool = dispatch(policy, cfg, gt, passing)
        pool.function(pool.arguments[0])

    assert len(checked) == 2


@pytest.mark.parametrize("device_name", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable",
    )),
])
def test_fixed_worker_restores_custom_architecture_on_requested_device(monkeypatch, device_name):
    source = TichuPolicy(hidden_dim=16, n_layers=3)
    weights = {key: value.numpy().copy()
               for key, value in source.state_dict().items()}

    def check_collect(policy, n_games, device):
        assert n_games == 1
        assert next(policy.parameters()).device.type == device_name
        for key, weight in source.state_dict().items():
            torch.testing.assert_close(weight, policy.state_dict()[key].cpu())
        obs = torch.zeros(1, source.trunk[0].in_features, device=device)
        mask = torch.ones(1, source.policy_head.out_features, dtype=torch.bool, device=device)
        actions, _, _ = policy.act(obs, mask)
        assert actions.shape == (1,)
        return {"actions": actions.cpu().numpy()}

    monkeypatch.setattr(rollout, "_collect_batched", check_collect)
    result = rollout._worker_collect((weights, 1, device_name, 1))
    assert result["actions"].shape == (1,)


@pytest.mark.parametrize("legacy_classifier", [False, True])
@pytest.mark.parametrize("attached", [False, True])
def test_embed_load_restores_gt_moments_only_for_matching_head(tmp_path, legacy_classifier, attached):
    policy = TichuPolicyEmbed(hidden_dim=16, n_layers=1, embed_dim=8)
    optimizer = torch.optim.Adam(policy.parameters())
    gt = TichuGrandTichuPolicy(hidden_dim=8, n_layers=1)
    gt_optimizer = torch.optim.Adam(gt.parameters())
    gt_state = gt.state_dict()
    if legacy_classifier:
        gt_state["head.weight"] = gt_state.pop("adv_head.weight")
        gt_state["head.bias"] = gt_state.pop("adv_head.bias")
    moments = gt_optimizer.state_dict()
    bias_index = list(dict(gt.named_parameters())).index("adv_head.bias")
    moments["state"][bias_index] = {
        "step": torch.tensor(100.),
        "exp_avg": torch.full_like(gt.adv_head.bias, 9.),
        "exp_avg_sq": torch.full_like(gt.adv_head.bias, 81.),
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
        "gt_model": gt_state, "gt_optimizer": moments, "games_played": 123,
    }, checkpoint)

    restored_gt = TichuGrandTichuPolicy(hidden_dim=8, n_layers=1)
    restored_optimizer = torch.optim.Adam(restored_gt.parameters())
    if attached:
        load_gt_checkpoint(
            restored_gt, restored_optimizer, str(checkpoint),
            torch.device("cpu"), load_optimizer=True,
        )
    else:
        games = load_checkpoint(
            policy, optimizer, str(checkpoint), torch.device("cpu"),
            gt_policy=restored_gt, gt_optimizer=restored_optimizer,
        )
        assert games == 123
    assert restored_gt.migrated_from_classifier == legacy_classifier
    if legacy_classifier:
        assert not restored_optimizer.state
    else:
        torch.testing.assert_close(
            restored_optimizer.state[restored_gt.adv_head.bias]["exp_avg"],
            torch.tensor([9.]),
        )
