"""Wiring tests for the GT teacher calibration probe."""
import random

import numpy as np
import torch

from tichu_env.game_state import GameState
from training.eval import _run_gt_phase
from training.gt_calibration import GTTeacherProbe


def _fresh_games(n):
    games = [GameState() for _ in range(n)]
    for gs in games:
        gs.reset(with_calling=True, with_passing=True)
    return games


def test_run_gt_phase_reports_wanted_players_empty_without_policies():
    games = _fresh_games(2)
    wanted = _run_gt_phase(games, None, None, torch.device("cpu"))
    for w in wanted:
        assert w == {"a": False, "b": False, "a_players": [], "b_players": []}
    for gs in games:
        assert gs.phase.name != "GRAND_TICHU"


class _AlwaysDeclareGT:
    def act(self, h8_bin):
        return torch.ones(1, dtype=torch.int64), torch.zeros(1)


class _ConstValuePolicy(torch.nn.Module):
    """No history_encoder attr -> _get_batch_values uses trunk/value_head."""

    def trunk(self, obs):
        return obs

    def value_head(self, feat):
        return torch.full((feat.shape[0], 1), 0.25)


def test_probe_records_declared_and_control_decisions():
    rng = random.Random(0)
    games = _fresh_games(1)
    gt = _AlwaysDeclareGT()
    wanted = _run_gt_phase(games, gt, gt, torch.device("cpu"),
                           act_a=True, act_b=False)
    assert wanted[0]["a_players"] == [0, 2]
    assert wanted[0]["b_players"] == [1, 3]
    # team A actually declared, team B stayed quiet (hypothetical only)
    assert games[0].tichu_declarations[0] == 2
    assert games[0].tichu_declarations[1] == 0

    for p in range(4):
        left, partner, right = rng.sample(games[0].get_passing_hand(p), 3)
        games[0].pass_cards(p, left, partner, right)

    probe = GTTeacherProbe(_ConstValuePolicy(), torch.device("cpu"))
    probe.post_pass(games, wanted)
    assert len(probe.pending) == 4
    # toggling the GT bit must not leak into the real state
    assert games[0].tichu_declarations[0] == 2
    assert games[0].tichu_declarations[1] == 0

    probe.post_round(games, wanted)
    assert probe.pending == []
    assert len(probe.records["a"]) == 2
    assert len(probe.records["b"]) == 2
    rec = probe.records["a"][0]
    assert rec["v1"] == rec["v0"] == 0.25
    assert np.isfinite(rec["realized"])
