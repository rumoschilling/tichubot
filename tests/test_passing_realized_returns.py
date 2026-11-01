"""Tests for the realized-return passing training signal.

The retired V(post)−V(pre) proxy was exactly invariant to the
left/partner/right assignment — the direction test here is the regression
guard that the new signal can do what the old one provably could not.
"""
import numpy as np
import torch

from training.config import TrainConfig
from training.model_gt import TichuGrandTichuPolicy
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder
from training.model_passing import TichuPassingPolicy
from training.ppo_passing import ppo_update_passing
from training.rollout_history import _collect_batched

DEVICE = torch.device("cpu")


def _tiny_policy():
    enc = TransformerHistoryEncoder(d_model=32, nhead=2, n_layers=1, ffn_dim=64)
    return TichuPolicyHistory(enc, hidden_dim=32, n_layers=2)


def test_legacy_pass_state_dict_loads_with_fresh_baseline():
    torch.manual_seed(0)
    reference = TichuPassingPolicy()
    legacy = {k: v.clone() for k, v in reference.state_dict().items()
              if not k.startswith("baseline_head.")}

    migrated = TichuPassingPolicy()
    migrated.load_state_dict(legacy)

    obs = torch.zeros(3, 60)
    obs[:, :14] = 1.0
    hand = torch.arange(14).repeat(3, 1)
    torch.manual_seed(1)
    ref_choices, ref_lp = reference.act(obs, hand)
    torch.manual_seed(1)
    mig_choices, mig_lp = migrated.act(obs, hand)
    assert torch.equal(ref_choices, mig_choices)
    assert torch.allclose(ref_lp, mig_lp)
    # baseline starts neutral
    assert migrated.baseline(obs).abs().sum() == 0


def test_normal_rollouts_harvest_pass_rows_with_realized_returns():
    torch.manual_seed(0)
    n_games = 8
    buf, _ = _collect_batched(
        _tiny_policy(), n_games, DEVICE,
        with_calling=True, gt_policy=TichuGrandTichuPolicy(),
        pass_policy=TichuPassingPolicy())

    assert buf["pass_train_obs"].shape == (4 * n_games, 60)
    assert buf["pass_train_hand_cards"].shape == (4 * n_games, 14)
    assert buf["pass_train_choices"].shape == (4 * n_games, 3)
    assert buf["pass_train_log_probs"].shape == (4 * n_games, 3)
    assert buf["pass_train_returns"].shape == (4 * n_games,)
    assert "pass_train_rewards" not in buf, "control-diff key is probe-only"

    # returns are team-signed round returns: the four rows of one game are
    # {r, -r, r, -r}, so each game's block sums to zero and shares |r|
    per_game = buf["pass_train_returns"].reshape(n_games, 4)
    assert np.allclose(per_game.sum(axis=1), 0.0, atol=1e-6)
    assert np.allclose(
        np.abs(per_game), np.abs(per_game[:, :1]), atol=1e-6)
    # the harvest must line up with the play stream's outcomes
    assert np.any(per_game != 0.0), "8 rounds with all-zero returns is wrong"


def test_ppo_update_passing_learns_direction_from_realized_returns():
    """The core capability the retired proxy lacked: route a specific card
    to the PARTNER slot when only the destination (not the shed set)
    changes the return."""
    torch.manual_seed(0)
    np.random.seed(0)
    policy = TichuPassingPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    cfg = TrainConfig()
    cfg.entropy_coef = 0.01
    cfg.batch_size = 512

    B = 512
    obs = torch.zeros(B, 60)
    obs[:, :14] = 1.0  # fixed hand: cards 0..13
    hand = torch.arange(14).repeat(B, 1)
    target = 7  # hand position that "supports the partner"

    def partner_rate():
        with torch.no_grad():
            choices, _ = policy.act(obs, hand)
        return float((choices[:, 1] == target).float().mean())

    rate0 = partner_rate()
    for _ in range(40):
        with torch.no_grad():
            choices, lps = policy.act(obs, hand)
        rets = torch.where(choices[:, 1] == target, 0.5, -0.5)
        rollout = {
            "pass_obs": obs.numpy(),
            "hand_cards": hand.numpy(),
            "choices": choices.numpy(),
            "log_probs": lps.numpy(),
            "returns": rets.numpy().astype(np.float32),
        }
        stats = ppo_update_passing(policy, opt, rollout, cfg, DEVICE)

    assert set(stats) == {"pass_policy_loss", "pass_baseline_loss",
                          "pass_entropy", "pass_approx_kl"}
    rate1 = partner_rate()
    assert rate1 > max(4 * rate0, 0.5), (
        f"partner-slot routing must be learned: {rate0:.3f} -> {rate1:.3f}")


def test_baseline_learns_mean_return():
    torch.manual_seed(0)
    np.random.seed(0)
    policy = TichuPassingPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    cfg = TrainConfig()
    cfg.batch_size = 256

    B = 256
    obs = torch.zeros(B, 60)
    obs[:, 14:28] = 1.0
    hand = torch.arange(14, 28).repeat(B, 1)

    for _ in range(60):
        with torch.no_grad():
            choices, lps = policy.act(obs, hand)
        rollout = {
            "pass_obs": obs.numpy(),
            "hand_cards": hand.numpy(),
            "choices": choices.numpy(),
            "log_probs": lps.numpy(),
            "returns": np.full(B, 0.3, dtype=np.float32),
        }
        ppo_update_passing(policy, opt, rollout, cfg, DEVICE)

    with torch.no_grad():
        assert abs(float(policy.baseline(obs[:1])) - 0.3) < 0.05


def test_advantages_path_skips_baseline():
    """The gt-probe practice channel passes control-differenced returns as
    ready-made advantages; the baseline must not train on them."""
    torch.manual_seed(0)
    np.random.seed(0)
    policy = TichuPassingPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    cfg = TrainConfig()

    B = 64
    obs = torch.zeros(B, 60)
    obs[:, :14] = 1.0
    hand = torch.arange(14).repeat(B, 1)
    with torch.no_grad():
        choices, lps = policy.act(obs, hand)

    stats = ppo_update_passing(
        policy, opt,
        {
            "pass_obs": obs.numpy(),
            "hand_cards": hand.numpy(),
            "choices": choices.numpy(),
            "log_probs": lps.numpy(),
            "advantages": np.random.randn(B).astype(np.float32),
        },
        cfg, DEVICE,
    )
    assert stats["pass_baseline_loss"] == 0.0
    # zero-init baseline stays exactly zero when only advantages are given
    assert policy.baseline_head.weight.abs().sum() == 0
    assert policy.baseline_head.bias.abs().sum() == 0
