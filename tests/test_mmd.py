"""Magnetic Mirror Descent: KL math, plateau test, PPO equivalence, controller."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_ppo_rollout_optimizations import _ppo_cfg, _synthetic_history_rollout, _tiny_history_policy
from training.mmd import MMDController, kl_plateaued, kl_to_magnet, magnet_log_probs
from training.ppo_history import ppo_update_history


def _batch(seed=0, rows=6, width=7):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, width, generator=g)
    n = torch.tensor([1, 2, 3, 5, 7, 4][:rows])
    valid = torch.arange(width)[None, :] < n[:, None]
    return logits, valid


def test_uniform_magnet_gradient_equals_entropy_bonus():
    logits, valid = _batch()
    a = logits.clone().requires_grad_(True)
    b = logits.clone().requires_grad_(True)
    alpha = 0.03
    (alpha * kl_to_magnet(a, valid, magnet_log_probs(None, valid, 0.2)).mean()).backward()
    log_pi = torch.log_softmax(b.masked_fill(~valid, float("-inf")), -1).masked_fill(~valid, 0.0)
    entropy = -(log_pi.exp() * log_pi).sum(-1)
    (-alpha * entropy.mean()).backward()
    torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-5)


def test_mixed_magnet_is_a_distribution_with_floor():
    logits, valid = _batch(1)
    magnet = torch.randn_like(logits) * 20  # very peaked
    rho = magnet_log_probs(magnet, valid, 0.2).exp() * valid
    torch.testing.assert_close(rho.sum(-1), torch.ones(len(rho)))
    n = valid.sum(-1, keepdim=True).float()
    assert torch.all(rho[valid] >= (0.2 / n.expand_as(rho))[valid] - 1e-7)


def test_plateau_test():
    rng = np.random.default_rng(0)
    rising = list(np.linspace(0.0, 1.0, 100) + rng.normal(0, 0.02, 100))
    flat = list(0.5 + rng.normal(0, 0.02, 100))
    assert not kl_plateaued(rising)[0]
    assert kl_plateaued(flat)[0]
    assert kl_plateaued(list(np.linspace(1.0, 0.5, 100)))[0]  # falling counts as converged


class _Holder:
    """Minimal stand-in for MMDController in round 0."""

    magnet_policy = None

    class state:
        round = 0


def test_round_zero_mmd_update_equals_entropy_update():
    policy = _tiny_history_policy(seed=3)
    rollout = _synthetic_history_rollout(policy)
    base = copy.deepcopy(policy.state_dict())

    def run(mmd_alpha, entropy, mmd):
        p = _tiny_history_policy(seed=3)
        p.load_state_dict(base)
        cfg = _ppo_cfg(use_ppo_bucketing=False)
        cfg.entropy_coef, cfg.mmd_alpha, cfg.mmd_epsilon = entropy, mmd_alpha, 0.2
        # SGD: updates proportional to gradients (Adam's first step is ~lr*sign(g),
        # which turns rounding noise on near-zero gradients into full steps)
        opt = torch.optim.SGD(p.parameters(), lr=0.1)
        np.random.seed(2024)  # minibatch order
        stats = ppo_update_history(p, opt, rollout, cfg, torch.device("cpu"), mmd=mmd)
        return p.state_dict(), stats

    with_entropy, _ = run(0.0, 0.03, None)
    # the dragon/wish heads keep entropy_coef in both runs; only the play loss differs
    with_mmd, stats = run(0.03, 0.03, _Holder())
    for key in base:
        torch.testing.assert_close(with_mmd[key], with_entropy[key], atol=1e-6, rtol=1e-5)
    assert stats["magnet_kl"] >= 0 and stats["mmd_round"] == 0


class _Cfg:
    def __init__(self, tmp):
        self.checkpoint_dir = str(tmp)
        self.mmd_alpha, self.mmd_epsilon = 0.03, 0.2
        self.mmd_window_updates, self.mmd_min_round_games, self.mmd_max_round_games = 10, 50, 10_000
        self.mmd_eval_deals, self.mmd_eval_procs, self.mmd_reference = 256, 1, None


def test_controller_moves_magnet_then_restarts_on_regression(tmp_path, monkeypatch):
    policy = _tiny_history_policy(seed=1)
    saved, loaded = [], []

    def save(games, path):
        torch.save({"model": policy.state_dict(), "games_played": games}, path)
        saved.append(Path(path).name)

    ctl = MMDController(_Cfg(tmp_path), torch.device("cpu"), save, lambda p: loaded.append(p), policy)
    ctl.restore(1000)
    verdicts = iter([{"mean": 3.0, "se": 1.0, "z": 3.0}, {"mean": -3.0, "se": 1.0, "z": -3.0}])
    monkeypatch.setattr(ctl, "_evaluate", lambda cand, opp, name: next(verdicts))

    games = 1000

    def feed_until_round_end(kl_of_step):
        nonlocal games
        before = (ctl.state.round, ctl.state.restarts)
        for k in range(200):
            games += 10
            ctl.after_update(games, {"magnet_kl": kl_of_step(k)})
            if (ctl.state.round, ctl.state.restarts) != before:
                return k
        raise AssertionError("round never ended")

    steps = feed_until_round_end(lambda k: min(k, 10) * 0.1 + 1e-4 * (k % 3))  # rises, then flat
    assert steps >= 10  # not before the KL stopped rising
    assert ctl.state.round == 1 and ctl.magnet_policy is not None  # round 0 ends without a check
    assert (tmp_path / "magnet.pt").exists() and "mmd_round_00.pt" in saved

    feed_until_round_end(lambda k: 0.5 + 1e-4 * (k % 3))
    assert ctl.state.round == 2 and ctl.state.restarts == 0  # check passed (z=+3): moved

    feed_until_round_end(lambda k: 0.5 + 1e-4 * (k % 3))
    assert ctl.state.round == 2 and ctl.state.restarts == 1  # regressed (z=-3): restarted
    assert loaded == [str(tmp_path / "magnet.pt")]
    state = json.loads((tmp_path / "mmd_state.json").read_text())
    assert [e["event"] for e in state["events"]] == ["start", "move", "move", "restart"]

    resumed = MMDController(_Cfg(tmp_path), torch.device("cpu"), save, lambda p: None, policy)
    resumed.restore(games)
    assert resumed.state.round == 2 and resumed.magnet_policy is not None
