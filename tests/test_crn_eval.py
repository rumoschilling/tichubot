"""Common random numbers and process sharding in the paired evaluator."""

import argparse
import json
import os
import random

import numpy as np
import pytest
import torch

import training.arena as arena
from training import paired_checkpoint_eval as pce
from training.config import TrainConfig
from training.model_gt import TichuGrandTichuPolicy
from training.model_passing import TichuPassingPolicy
from training.train_history import _build_policy


def _cfg():
    cfg = TrainConfig()
    cfg.hidden_dim, cfg.trunk_dims, cfg.embed_dim = 32, (48, 32), 16
    cfg.mlp_activation, cfg.mlp_layer_norm = "silu", True
    cfg.hist_d_model, cfg.hist_nhead, cfg.hist_n_layers, cfg.hist_ffn_dim = 24, 4, 1, 48
    cfg.hist_norm_first, cfg.hist_activation = True, "gelu"
    cfg.split_actor_critic_history = cfg.static_attention = True
    return cfg


def _save(path, seed, heads=True):
    torch.manual_seed(seed)
    cfg = _cfg()
    ckpt = {"model": _build_policy(cfg).state_dict(), "config": cfg, "games_played": seed}
    if heads:
        ckpt["pass_model"] = TichuPassingPolicy(hidden_dim=cfg.pass_hidden_dim, n_layers=cfg.pass_n_layers).state_dict()
        ckpt["gt_model"] = TichuGrandTichuPolicy(hidden_dim=cfg.gt_hidden_dim, n_layers=cfg.gt_n_layers).state_dict()
    torch.save(ckpt, path)
    return str(path)


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("crn")
    cwd = os.getcwd()
    os.chdir(tmp)  # keep the repo's machine-dependent passing fallback out
    yield tmp, _save(tmp / "a.pt", 1), _save(tmp / "b.pt", 2), _save(tmp / "bare.pt", 3, heads=False)
    os.chdir(cwd)


def _orientations(models, seeds, crn):
    out = []
    for k, seating in enumerate(((0, 1, 0, 1), (1, 0, 1, 0))):
        torch.manual_seed(10 + k)
        out.append(arena.play_matchup(models, seating, len(seeds), batch=8, rng=random.Random(20 + k),
                                      game_seeds=seeds, with_calling=True, common_random_numbers=crn))
    return out


@pytest.mark.parametrize("which", ["a", "bare"])
def test_crn_self_play_cancels_exactly(ckpts, which):
    tmp, a, _b, bare = ckpts
    path = a if which == "a" else bare
    models = [arena.load_arena_model(f"x={path}", torch.device("cpu")),
              arena.load_arena_model(f"y={path}", torch.device("cpu"))]
    seeds = np.random.default_rng(5).integers(0, 2**63, size=12, dtype=np.int64)
    first, second = _orientations(models, seeds, crn=True)
    assert np.array_equal(first, second)  # identical policies -> identical games, effect 0 per deal
    plain_a, plain_b = _orientations(models, seeds, crn=False)
    assert not np.array_equal(plain_a, plain_b)  # without CRN the sampling noise is there


def _args(tmp, a, b, procs, crn):
    return argparse.Namespace(
        candidate=f"a={a}", opponents=[f"b={b}"], output=str(tmp / f"out_{procs}_{crn}.json"),
        device="cpu", deals=16, batch=8, block_size=4, bootstrap_samples=200, sign_flip_samples=200,
        alpha=0.05, seed=7, no_gt=False, procs=procs, common_random_numbers=crn,
    )


def test_sharded_crn_matches_in_process(ckpts):
    tmp, a, b, _bare = ckpts
    one = pce.run_experiment(_args(tmp, a, b, 1, True))["comparisons"][0]
    two = pce.run_experiment(_args(tmp, a, b, 2, True))["comparisons"][0]
    assert one["block_means"] == two["block_means"]
    stored = json.loads((tmp / "out_2_True.json").read_text())
    assert stored["design"]["procs"] == 2 and stored["design"]["common_random_numbers"] is True
