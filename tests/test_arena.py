"""Tests for the model arena (training/arena.py)."""

import os

import numpy as np
import pytest
import torch

import training.arena as arena
from tichu_env.game_state import GameState
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder

TINY_CFG = {
    "hidden_dim": 64,
    "n_layers": 2,
    "embed_dim": 32,
    "hist_d_model": 32,
    "hist_nhead": 2,
    "hist_n_layers": 1,
    "hist_ffn_dim": 64,
    "hist_proj_gain": 0.1,
}


def _tiny_checkpoint(tmp_path, name: str, seed: int):
    torch.manual_seed(seed)
    enc = TransformerHistoryEncoder(
        d_model=TINY_CFG["hist_d_model"],
        nhead=TINY_CFG["hist_nhead"],
        n_layers=TINY_CFG["hist_n_layers"],
        ffn_dim=TINY_CFG["hist_ffn_dim"],
    )
    model = TichuPolicyHistory(
        enc,
        hidden_dim=TINY_CFG["hidden_dim"],
        n_layers=TINY_CFG["n_layers"],
        embed_dim=TINY_CFG["embed_dim"],
        history_proj_gain=TINY_CFG["hist_proj_gain"],
    )
    path = tmp_path / f"{name}.pt"
    torch.save(
        {"model": model.state_dict(), "config": TINY_CFG, "games_played": 0},
        path,
    )
    return path


@pytest.fixture(scope="module")
def two_models(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("arena_ckpts")
    device = torch.device("cpu")
    # load from tmp cwd so the repo's checkpoints/passing_fallback (untracked,
    # machine-dependent) is not picked up — these tests expect random passing
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        a = arena.load_arena_model(f"tinyA={_tiny_checkpoint(tmp_path, 'a', 1)}", device)
        b = arena.load_arena_model(f"tinyB={_tiny_checkpoint(tmp_path, 'b', 2)}", device)
    finally:
        os.chdir(cwd)
    return [a, b]


def test_load_arena_model_labels_and_kind(two_models):
    assert [m.label for m in two_models] == ["tinyA", "tinyB"]
    assert all(m.kind == "history" for m in two_models)
    # tiny checkpoints bundle no pass model -> random passing
    assert all(m.pass_policy is None for m in two_models)
    assert all(m.gt_policy is None for m in two_models)
    # specialist heads exist on history models
    assert all(m.decide_dragon is not None and m.decide_wish is not None
               for m in two_models)


def test_play_matchup_runs_and_scores(two_models):
    diffs = arena.play_matchup(
        two_models, (0, 1, 0, 1), n_games=4, batch=4,
        rng=__import__("random").Random(0),
    )
    assert diffs.shape == (4,)
    # Tichu round diffs are bounded and finite
    assert np.all(np.isfinite(diffs))
    assert np.all(np.abs(diffs) <= 800)


def test_player_model_ids_set_per_seating(two_models, monkeypatch):
    captured: list[list[int]] = []

    class RecordingGameState(GameState):
        def reset(self, *args, **kwargs):
            captured.append(list(self.player_model_ids))
            return super().reset(*args, **kwargs)

    monkeypatch.setattr(arena, "GameState", RecordingGameState)

    for seating in [(0, 1, 0, 1), (0, 1, 1, 0), (0, 0, 1, 0), (1, 1, 1, 1)]:
        captured.clear()
        arena.play_matchup(
            two_models, seating, n_games=2, batch=2,
            rng=__import__("random").Random(0),
        )
        assert captured == [list(seating)] * 2


def test_play_matchup_uses_explicit_deal_seeds_and_calling(two_models, monkeypatch):
    captured: list[tuple[int | None, bool]] = []

    class RecordingGameState(GameState):
        def reset(self, *args, **kwargs):
            captured.append((kwargs.get("seed"), kwargs.get("with_calling", False)))
            return super().reset(*args, **kwargs)

    monkeypatch.setattr(arena, "GameState", RecordingGameState)
    arena.play_matchup(
        two_models,
        (0, 1, 0, 1),
        n_games=2,
        batch=2,
        rng=__import__("random").Random(0),
        game_seeds=[11, 22],
        with_calling=True,
    )

    assert captured == [(11, True), (22, True)]


def test_team_swap():
    assert arena._team_swap((0, 1, 0, 1)) == (1, 0, 1, 0)
    assert arena._team_swap((0, 0, 1, 0)) == (0, 0, 0, 1)


def test_pooled_sign_correction_and_size(two_models):
    rng = __import__("random").Random(0)
    res = arena._pooled(
        two_models, (0, 1, 0, 1), (1, 0, 1, 0), n_games=3, batch=3,
        device=torch.device("cpu"), rng=rng,
    )
    assert res.n == 6
    assert np.isfinite(res.mean)
    assert 0.0 <= res.win_rate <= 1.0
    assert 0.0 <= res.draw_rate <= 1.0


def test_run_arena_end_to_end_structure(two_models):
    results = arena.run_arena(
        two_models, n_games=2, batch=2, include_coop=True, include_self=True,
        seed=0, progress=False,
    )
    assert results["labels"] == ["tinyA", "tinyB"]
    assert (0, 1) in results["versus"] and results["versus"][(0, 1)].n == 4
    assert (0, 1) in results["mixed"]
    # coop is ordered: both directions present
    assert (0, 1) in results["coop"] and (1, 0) in results["coop"]
    # self-pairs for both models
    assert set(results["self"].keys()) == {0, 1}

    report = arena.format_report(results)
    assert "tinyA" in report and "tinyB" in report
    assert "cooperation" in report.lower()

    payload = arena.results_to_json(results)
    assert payload["versus"]["tinyA|tinyB"]["n"] == 4
    # JSON payload must be serializable
    import json
    json.dumps(payload)
