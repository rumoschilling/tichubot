"""Tests for the passing net's grand-tichu declaration input (56 -> 60)."""
import numpy as np
import torch

from tichu_env.encoding import PASS_OBS_SIZE, encode_pass_obs
from tichu_env.game_state import GameState
from training.checkpoints import migrate_legacy_pass_optimizer
from training.model_passing import (
    GT_DECL_DIM,
    NUM_CARDS,
    PASS_OBS_DIM,
    TichuPassingPolicy,
)
from training.rollout_passing import _collect_passing_batched


def test_pass_obs_dims_agree():
    assert PASS_OBS_SIZE == PASS_OBS_DIM == NUM_CARDS + GT_DECL_DIM


def test_encode_pass_obs_hand_and_relative_declaration_bits():
    gs = GameState()
    gs.reset(seed=7, with_calling=True, with_passing=True)
    for p in range(4):
        gs.declare_grand_tichu(p, p == 3)

    obs = encode_pass_obs(gs, 1)
    assert obs.shape == (PASS_OBS_SIZE,)
    hand = gs.get_passing_hand(1)
    assert len(hand) == 14
    assert obs[:NUM_CARDS].sum() == 14
    assert all(obs[c] == 1.0 for c in hand)
    # player 3 declared; relative to observer 1 that is offset (3 - 1) % 4 = 2
    assert obs[NUM_CARDS:].tolist() == [0.0, 0.0, 1.0, 0.0]

    obs_self = encode_pass_obs(gs, 3)
    assert obs_self[NUM_CARDS:].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_legacy_pass_model_loads_padded_and_behaves_identically():
    torch.manual_seed(0)
    reference = TichuPassingPolicy()
    legacy_state = {
        k: (v[:, :NUM_CARDS].clone() if k == "trunk.0.weight" else v.clone())
        for k, v in reference.state_dict().items()
    }

    migrated = TichuPassingPolicy()
    migrated.load_state_dict(legacy_state)
    assert migrated.trunk[0].weight.shape == (128, PASS_OBS_DIM)
    assert migrated.trunk[0].weight[:, NUM_CARDS:].abs().sum() == 0

    # With zeroed declaration columns the model must ignore the bits entirely
    obs = torch.zeros(2, PASS_OBS_DIM)
    hand_cards = torch.arange(14, dtype=torch.int64).repeat(2, 1)
    obs[:, :14] = 0.0
    for j in range(14):
        obs[:, j] = 1.0
    obs[1, NUM_CARDS:] = 1.0  # declaration bits differ between the rows
    choices = torch.tensor([[0, 1, 2], [0, 1, 2]])
    lp, ent = migrated.evaluate(obs, hand_cards, choices)
    assert torch.allclose(lp[0], lp[1])
    assert torch.allclose(ent[0], ent[1])


def test_fresh_model_initializes_declaration_columns_to_zero():
    policy = TichuPassingPolicy()
    assert policy.trunk[0].weight[:, NUM_CARDS:].abs().sum() == 0


def test_migrate_legacy_pass_optimizer_pads_only_trunk_input():
    optim_state = {
        "state": {
            0: {  # trunk.0.weight — must be padded
                "step": torch.tensor(10.0),
                "exp_avg": torch.ones(128, NUM_CARDS),
                "exp_avg_sq": torch.ones(128, NUM_CARDS),
            },
            5: {  # picked_proj.weight — same shape, must NOT be touched
                "step": torch.tensor(10.0),
                "exp_avg": torch.ones(128, NUM_CARDS),
                "exp_avg_sq": torch.ones(128, NUM_CARDS),
            },
        },
        "param_groups": [],
    }
    assert migrate_legacy_pass_optimizer(optim_state)
    assert optim_state["state"][0]["exp_avg"].shape == (128, PASS_OBS_DIM)
    assert optim_state["state"][0]["exp_avg"][:, NUM_CARDS:].abs().sum() == 0
    assert optim_state["state"][5]["exp_avg"].shape == (128, NUM_CARDS)
    # already-migrated state is a no-op
    assert not migrate_legacy_pass_optimizer(optim_state)


class _AlwaysDeclareGT:
    def act(self, h8_bin):
        n = h8_bin.shape[0]
        return torch.ones(n, dtype=torch.int64), torch.zeros(n)


class _ZeroValuePolicy(torch.nn.Module):
    def trunk(self, obs):
        return obs

    def value_head(self, feat):
        return torch.zeros(feat.shape[0], 1)


def test_passing_rollout_carries_declaration_bits():
    torch.manual_seed(0)
    pass_policy = TichuPassingPolicy()
    rollout = _collect_passing_batched(
        pass_policy, _ZeroValuePolicy(), n_deals=4,
        device=torch.device("cpu"), gt_policy=_AlwaysDeclareGT())

    obs = rollout["pass_obs"]
    assert obs.shape == (16, PASS_OBS_SIZE)
    assert obs[:, :NUM_CARDS].sum(axis=1).tolist() == [14.0] * 16
    # everyone declares (modulo the 2% explore eps), so declaration bits
    # should be nearly all set — and rewards must be finite
    assert obs[:, NUM_CARDS:].mean() > 0.8
    assert np.isfinite(rollout["rewards"]).all()


def test_passing_rollout_without_gt_policy_has_zero_bits():
    torch.manual_seed(0)
    pass_policy = TichuPassingPolicy()
    rollout = _collect_passing_batched(
        pass_policy, _ZeroValuePolicy(), n_deals=2,
        device=torch.device("cpu"))
    obs = rollout["pass_obs"]
    assert obs.shape == (8, PASS_OBS_SIZE)
    assert obs[:, NUM_CARDS:].sum() == 0
