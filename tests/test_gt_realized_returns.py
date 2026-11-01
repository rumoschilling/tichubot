"""Tests for the realized-return Grand Tichu training signal."""
import numpy as np
import torch

from training.config import TrainConfig
from training.model_gt import TichuGrandTichuPolicy
from training.ppo_gt import ppo_update_gt


def test_legacy_gt_state_dict_loads_with_fresh_baseline():
    torch.manual_seed(0)
    reference = TichuGrandTichuPolicy()
    # Oldest format: classifier head, no baseline head, no adv head.
    legacy = {k: v.clone() for k, v in reference.state_dict().items()
              if k.startswith("trunk.")}
    legacy["head.weight"] = torch.randn(1, 64) * 0.01
    legacy["head.bias"] = torch.zeros(1)

    migrated = TichuGrandTichuPolicy()
    migrated.load_state_dict(legacy)

    h8 = torch.zeros(3, 56)
    h8[:, :8] = 1.0
    # trunk carried over exactly
    assert torch.allclose(reference.trunk(h8), migrated.trunk(h8))
    # baseline starts neutral
    assert migrated.baseline(h8).abs().sum() == 0
    # classifier head is dropped; derived policy starts at the never-call
    # prior, and the trainer is told to skip stale optimizer state
    assert migrated.migrated_from_classifier
    with torch.no_grad():
        assert torch.sigmoid(migrated.call_logits(h8)).max() < 1e-6


def test_baseline_head_is_decision_blind_and_trains():
    torch.manual_seed(0)
    policy = TichuGrandTichuPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    # one hand type, returns average +0.3 regardless of decision
    h8 = torch.zeros(256, 56)
    h8[:, 10:18] = 1.0
    rets = torch.full((256,), 0.3)
    for _ in range(200):
        loss = ((policy.baseline(h8) - rets) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        assert abs(float(policy.baseline(h8[:1])) - 0.3) < 0.05


def _declare_prob(policy, h8_row):
    with torch.no_grad():
        logit = policy.call_logits(h8_row[None, :]).squeeze()
    return float(torch.sigmoid(logit))


def test_ppo_update_gt_learns_from_realized_returns():
    torch.manual_seed(0)
    np.random.seed(0)
    policy = TichuGrandTichuPolicy()
    # Start from an undecided policy: at the production never-call prior the
    # PPO path has (by design of the old objective) vanishing gradient —
    # that saturation blindness is exactly why probe mode regresses the
    # declare advantage instead. This test covers the update mechanics.
    with torch.no_grad():
        policy.adv_head.bias.zero_()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    cfg = TrainConfig()

    # A "monster" hand where declaring loses (like reality: -0.07 causal)
    # and a weak hand where declaring loses badly.
    monster = torch.zeros(56)
    monster[[51, 54, 55, 12, 25, 38, 11, 24]] = 1.0  # aces, phoenix, dragon
    weak = torch.zeros(56)
    weak[[0, 1, 13, 14, 26, 27, 39, 40]] = 1.0

    p_monster0 = _declare_prob(policy, monster)

    rng = np.random.default_rng(0)
    for _ in range(30):
        rows_h8, rows_decl, rows_lp, rows_ret = [], [], [], []
        for _ in range(256):
            is_monster = rng.random() < 0.5
            h8 = monster if is_monster else weak
            p = _declare_prob(policy, h8)
            declare = rng.random() < (0.98 * p + 0.01)
            p_act = p if declare else 1.0 - p
            behave = 0.98 * p_act + 0.02 * 0.5
            base = 0.14 if is_monster else -0.10  # quiet expectation
            causal = -0.07 if declare else 0.0    # declaring costs
            ret = base + causal + rng.normal(0, 0.4)
            rows_h8.append(h8.numpy())
            rows_decl.append(int(declare))
            rows_lp.append(float(np.log(behave)))
            rows_ret.append(float(ret))
        rollout = {
            "gt_h8": np.stack(rows_h8).astype(np.float32),
            "gt_declare": np.array(rows_decl, dtype=np.int64),
            "gt_log_probs": np.array(rows_lp, dtype=np.float32),
            "gt_returns": np.array(rows_ret, dtype=np.float32),
        }
        stats = ppo_update_gt(policy, opt, rollout, cfg, TorchDevice)

    assert set(stats) == {"gt_policy_loss", "gt_baseline_loss",
                          "gt_entropy", "gt_approx_kl"}
    # declaring loses on both hand types -> both probs must fall
    assert _declare_prob(policy, monster) < min(p_monster0, 0.25)
    assert _declare_prob(policy, weak) < 0.25
    # baseline must have learned the hand-quality gap (monster > weak)
    with torch.no_grad():
        b_monster = float(policy.baseline(monster[None, :]))
        b_weak = float(policy.baseline(weak[None, :]))
    assert b_monster > b_weak


TorchDevice = torch.device("cpu")
