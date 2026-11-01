"""Shared-control GT probe rollouts and the probe update (probe/adapt duality)."""

import numpy as np
import pytest
import torch

from tichu_env.game_state import GameState
from training.config import TrainConfig
from training.model_gt import TichuGrandTichuPolicy
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder
from training.ppo_gt import gt_probe_update
from training.rollout_history import _collect_batched


def _tiny_policy():
    enc = TransformerHistoryEncoder(d_model=32, nhead=2, n_layers=1, ffn_dim=64)
    return TichuPolicyHistory(enc, hidden_dim=32, n_layers=2)


def test_gt_probe_groups_share_deal_one_control_four_sole_declarers(monkeypatch):
    calls: dict[int, dict[int, bool]] = {}
    deal_at_decision: dict[int, tuple] = {}
    orig = GameState.declare_grand_tichu

    def spy(self, player, declare):
        key = id(self)
        if key not in calls:
            calls[key] = {}
            deal_at_decision[key] = tuple(
                tuple(sorted(h)) for h in self.hands)
        calls[key][player] = declare
        return orig(self, player, declare)

    monkeypatch.setattr(GameState, "declare_grand_tichu", spy)
    torch.manual_seed(0)
    buf, _ = _collect_batched(
        _tiny_policy(), 10, torch.device("cpu"),
        with_calling=True, gt_policy=TichuGrandTichuPolicy(), gt_probe=True)

    assert len(calls) == 10
    by_deal: dict[tuple, list[int]] = {}
    for key, deal in deal_at_decision.items():
        by_deal.setdefault(deal, []).append(key)
    assert len(by_deal) == 2, "each deal must appear in exactly five branches"
    for keys in by_deal.values():
        assert len(keys) == 5
        declarer_sets = []
        for key in keys:
            d = calls[key]
            assert len(d) == 4
            declarer_sets.append(frozenset(p for p in range(4) if d[p]))
        controls = [s for s in declarer_sets if len(s) == 0]
        singles = [s for s in declarer_sets if len(s) == 1]
        assert len(controls) == 1, "exactly one all-quiet control per deal"
        assert len(singles) == 4, "four sole-declarer branches per deal"
        assert set().union(*singles) == {0, 1, 2, 3}, (
            "the four branches must cover all four seats")

    assert buf["gt_probe_h8"].shape == (8, 56)
    assert np.all(buf["gt_probe_h8"].sum(axis=1) == 8)
    assert buf["gt_probe_diff"].shape == (8,)
    assert buf["gt_probe_quiet"].shape == (8,)
    assert "gt_h8" not in buf, "probe decisions must not enter PPO harvest"


def test_gt_probe_requires_group_multiple_and_calling():
    with pytest.raises(ValueError):
        _collect_batched(_tiny_policy(), 8, torch.device("cpu"),
                         with_calling=True, gt_policy=TichuGrandTichuPolicy(),
                         gt_probe=True)
    with pytest.raises(ValueError):
        _collect_batched(_tiny_policy(), 10, torch.device("cpu"),
                         with_calling=False, gt_policy=TichuGrandTichuPolicy(),
                         gt_probe=True)


def test_gt_probe_update_moves_prob_toward_effect_sign():
    torch.manual_seed(0)
    np.random.seed(0)
    gt = TichuGrandTichuPolicy()
    opt = torch.optim.Adam(gt.parameters(), lr=3e-3)
    cfg = TrainConfig()
    cfg.n_epochs = 40
    cfg.batch_size = 64
    cfg.entropy_coef = 0.01

    good = np.zeros(56, dtype=np.float32)
    good[:8] = 1.0
    bad = np.zeros(56, dtype=np.float32)
    bad[-8:] = 1.0
    n = 256
    rollout = {
        "gt_probe_h8": np.stack([good] * n + [bad] * n),
        "gt_probe_diff": np.concatenate(
            [np.full(n, 0.3, np.float32), np.full(n, -0.3, np.float32)]),
        "gt_probe_quiet": np.concatenate(
            [np.full(n, 0.2, np.float32), np.full(n, -0.1, np.float32)]),
    }
    probe_hands = torch.as_tensor(np.stack([good, bad]))

    with torch.no_grad():
        p0 = torch.sigmoid(gt.call_logits(probe_hands)).squeeze(-1)
    # The regression must escape the saturated never-call prior — the exact
    # spot where the old δ·p policy-gradient objective had ~zero gradient.
    assert p0.max() < 1e-6
    stats = gt_probe_update(gt, opt, rollout, cfg, torch.device("cpu"))
    with torch.no_grad():
        p1 = torch.sigmoid(gt.call_logits(probe_hands)).squeeze(-1)
        d1 = gt.declare_advantage(probe_hands)

    assert p1[0] > 0.9, "positive effect must raise p from saturation"
    assert p1[1] < 0.01, "negative effect must keep p at never-call"
    assert abs(d1[0].item() - 0.3) < 0.1, "d must regress toward E[diff|good]"
    assert abs(d1[1].item() + 0.3) < 0.1, "d must regress toward E[diff|bad]"
    assert stats["gt_diff_pos_frac"] == 0.5

    # baseline learns the quiet-branch conditional means
    with torch.no_grad():
        b = gt.baseline(probe_hands)
    assert abs(b[0].item() - 0.2) < 0.1
    assert abs(b[1].item() + 0.1) < 0.1


def test_gt_probe_block_practice_runs_play_update_per_chunk():
    from training.play_training import PlayTrainingBundle, _run_gt_probe_block

    torch.manual_seed(0)
    gt = TichuGrandTichuPolicy()
    play = _tiny_policy()
    bundle = PlayTrainingBundle(
        policy=play,
        optimizer=torch.optim.Adam(play.parameters()),
        gt_policy=gt,
        gt_optimizer=torch.optim.Adam(gt.parameters()),
    )
    cfg = TrainConfig()
    cfg.n_epochs = 1
    cfg.batch_size = 64
    cfg.gt_probe_games = 10
    cfg.gt_probe_chunk = 5

    collected: list[dict] = []

    def fake_collect(n_games):
        buf, _ = _collect_batched(
            play, n_games, torch.device("cpu"),
            with_calling=True, gt_policy=gt, gt_probe=True)
        collected.append(buf)
        return buf

    play_updates: list[dict] = []

    def fake_ppo_update(buf):
        play_updates.append(buf)
        return {"policy_loss": 0.0, "approx_kl": 0.0}

    _run_gt_probe_block(
        bundle, cfg, torch.device("cpu"), fake_collect,
        ppo_update=fake_ppo_update)
    assert len(collected) == 2, "10 game-equivalents in chunks of 5"
    assert play_updates == collected, (
        "practice must run one play PPO update per chunk on that chunk's "
        "full rollout buffer")

    # practice off: play never updates
    cfg.gt_probe_practice = False
    play_updates.clear()
    collected.clear()
    _run_gt_probe_block(
        bundle, cfg, torch.device("cpu"), fake_collect,
        ppo_update=fake_ppo_update)
    assert len(collected) == 2 and play_updates == []

    # an expired deadline cuts the block short before any chunk is collected
    import time
    collected.clear()
    _run_gt_probe_block(
        bundle, cfg, torch.device("cpu"), fake_collect,
        ppo_update=fake_ppo_update, deadline=time.monotonic() - 1.0)
    assert collected == []


def test_gt_probe_harvests_pass_practice_rows():
    from training.model_passing import TichuPassingPolicy
    torch.manual_seed(0)
    buf, _ = _collect_batched(
        _tiny_policy(), 10, torch.device("cpu"),
        with_calling=True, gt_policy=TichuGrandTichuPolicy(),
        pass_policy=TichuPassingPolicy(), gt_probe=True)

    # 2 groups x 4 declared branches x 4 passes; quiet controls not harvested
    assert buf["pass_train_obs"].shape == (32, 60)
    assert buf["pass_train_hand_cards"].shape == (32, 14)
    assert buf["pass_train_choices"].shape == (32, 3)
    assert buf["pass_train_log_probs"].shape == (32, 3)
    assert buf["pass_train_rewards"].shape == (32,)
    # every harvested pass obs sees exactly one (sole) declarer
    assert np.all(buf["pass_train_obs"][:, 56:60].sum(axis=1) == 1.0)
    # rewards are team-signed control diffs: the four seats of one branch
    # (consecutive rows, sequential passing order) cancel pairwise
    per_branch = buf["pass_train_rewards"].reshape(8, 4)
    assert np.allclose(per_branch.sum(axis=1), 0.0, atol=1e-6)
