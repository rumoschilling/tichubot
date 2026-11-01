"""End-to-end check of the saturation pipeline on tiny checkpoints."""

import os

import pytest
import torch

from training import saturation
from training.config import TrainConfig
from training.model_gt import TichuGrandTichuPolicy
from training.model_passing import TichuPassingPolicy
from training.train_history import _build_policy


def _tiny_cfg(**flags):
    cfg = TrainConfig()
    cfg.hidden_dim, cfg.trunk_dims, cfg.embed_dim = 32, (48, 32), 16
    cfg.mlp_activation, cfg.mlp_layer_norm = "silu", True
    cfg.hist_d_model, cfg.hist_nhead, cfg.hist_n_layers, cfg.hist_ffn_dim = 24, 4, 1, 48
    cfg.hist_norm_first, cfg.hist_activation = True, "gelu"
    cfg.split_actor_critic_history = cfg.static_attention = True
    for key, value in flags.items():
        setattr(cfg, key, value)
    return cfg


def _save(path, cfg, games):
    torch.manual_seed(games)
    torch.save({
        "model": _build_policy(cfg).state_dict(),
        "pass_model": TichuPassingPolicy(hidden_dim=cfg.pass_hidden_dim, n_layers=cfg.pass_n_layers).state_dict(),
        "gt_model": TichuGrandTichuPolicy(hidden_dim=cfg.gt_hidden_dim, n_layers=cfg.gt_n_layers).state_dict(),
        "config": cfg, "games_played": games,
    }, path)
    return str(path)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sat")
    run = tmp / "run" / "checkpoints"
    run.mkdir(parents=True)
    new = _save(run / "ckpt_00001024.pt", _tiny_cfg(), 1024)
    _save(run / "ckpt_00002048.pt", _tiny_cfg(), 2048)
    _save(run / "ckpt_01000448.pt", _tiny_cfg(), 1000448)
    old = _save(tmp / "old.pt", _tiny_cfg(hist_final_norm=False, hist_pool_norm=False), 5)
    cwd = os.getcwd()
    os.chdir(tmp)  # keep the repo's machine-dependent passing fallback out
    try:
        bank = saturation.build_bank(24, 3, generator=new)
    finally:
        os.chdir(cwd)
    bank["meta"]["id"] = "test-bank"
    return tmp, run, new, old, bank


def test_probe_records_all_sections(setup):
    tmp, _, new, old, bank = setup
    rec = saturation.probe(new, bank, "tiny")
    assert rec["label"] == "tiny" and not rec["legacy_history"]
    assert rec["arch"] == {"final_norm": True, "pool_norm": True}
    assert {"trunk.2", "history_fuse.2", "critic_history_fuse.2"} <= set(rec["units"])
    assert "history_encoder.encoder.layers.0.ffn" in rec["units"]
    for stats in rec["units"].values():
        assert all(0.0 <= v <= 1.0 for k, v in stats.items() if k != "units")
    h = rec["history"]
    assert 0.0 <= h["actor_argmax_flip"] <= 1.0 and h["actor_kl"] >= 0.0
    assert 0.0 <= h["actor_history_share"] <= 1.0
    p = rec["policy"]["entropy"]
    assert p["lo"] <= p["mean"] <= p["hi"]
    assert 0.0 <= rec["gt"]["decided"] <= 1.0

    legacy = saturation.probe(old, bank, "old")
    assert legacy["label"] == "old-noLN" and legacy["legacy_history"]


def test_cache_and_run_dir_selection(setup):
    tmp, run, new, _, bank = setup
    records = tmp / "records"
    first, fresh = saturation.probe_cached(new, "tiny", bank, records)
    assert fresh and (records / "tiny.json").exists()
    again, fresh = saturation.probe_cached(new, "tiny", bank, records)
    assert not fresh and again == first
    picked = saturation.run_dir_checkpoints(str(run), every=1_000_000)
    assert [g for g, _ in picked] == [1024, 1000448]
    text = saturation.report([first])
    assert "tiny" in text and "HISTORY BRANCH" in text
