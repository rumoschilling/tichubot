"""History-branch norms (--hist-final-norm / --hist-pool-norm) and the
resume path from a checkpoint trained without them."""

import torch

from tichu_env.encoding import PLAY_TOKEN_SIZE, STATIC_OBS_SIZE
from training.config import TrainConfig
from training.train_history import (
    _build_policy,
    _make_optimizer,
    _optimizer_param_names,
    load_checkpoint,
    reset_history_fuse,
)


def _cfg(**flags):
    cfg = TrainConfig()
    cfg.hidden_dim = 32
    cfg.trunk_dims = (48, 32)
    cfg.embed_dim = 16
    cfg.mlp_activation = "silu"
    cfg.mlp_layer_norm = True
    cfg.hist_d_model = 24
    cfg.hist_nhead = 4
    cfg.hist_n_layers = 2
    cfg.hist_ffn_dim = 48
    cfg.hist_norm_first = True
    cfg.hist_activation = "gelu"
    cfg.split_actor_critic_history = True
    cfg.static_attention = True
    for key, value in flags.items():
        setattr(cfg, key, value)
    return cfg


def _inputs(batch=6, length=9):
    torch.manual_seed(0)
    obs = torch.rand(batch, STATIC_OBS_SIZE)
    history = torch.randn(batch, length, PLAY_TOKEN_SIZE) * 5
    lens = torch.tensor([0, 1, 3, 5, 9, 9][:batch])
    return obs, history, lens


def test_norms_are_default_and_legacy_adds_none():
    base = set(_build_policy(_cfg(hist_final_norm=False, hist_pool_norm=False)).state_dict())
    assert set(_build_policy(_cfg()).state_dict()) == set(
        _build_policy(_cfg(hist_final_norm=True, hist_pool_norm=True)).state_dict()
    )
    normed = set(_build_policy(_cfg()).state_dict())
    assert normed - base == {
        "history_encoder.encoder.norm.weight", "history_encoder.encoder.norm.bias",
        "history_pool_norm.weight", "history_pool_norm.bias",
        "critic_history_pool_norm.weight", "critic_history_pool_norm.bias",
    }


def test_pool_norm_bounds_history_scale():
    policy = _build_policy(_cfg(hist_final_norm=True, hist_pool_norm=True))
    with torch.no_grad():
        for p in policy.history_encoder.parameters():
            p.mul_(30.0)  # emulate the grown encoder of late checkpoints
    seen = {}
    policy.history_fuse[0].register_forward_hook(lambda m, i, o: seen.setdefault("x", i[0]))
    obs, history, lens = _inputs()
    with torch.no_grad():
        policy.actor_critic_features(obs, history, lens)
    hist_part = seen["x"][:, 32:]
    # LayerNorm output: per-row RMS ~1 (empty histories are zero before the norm)
    assert hist_part.pow(2).mean(-1).sqrt().max() < 1.5


def test_resume_from_unnormed_checkpoint_keeps_adam_state(tmp_path):
    old_cfg = _cfg(hist_final_norm=False, hist_pool_norm=False)
    old = _build_policy(old_cfg)
    old_opt = _make_optimizer(old, old_cfg)
    obs, history, lens = _inputs()
    actor, critic = old.actor_critic_features(obs, history, lens)
    (actor.sum() + critic.sum()).backward()
    old_opt.step()
    path = tmp_path / "old.pt"
    torch.save({"model": old.state_dict(), "optimizer": old_opt.state_dict(),
                "config": old_cfg, "games_played": 1234}, path)

    new_cfg = _cfg(hist_final_norm=True, hist_pool_norm=True)
    new = _build_policy(new_cfg)
    new_opt = _make_optimizer(new, new_cfg)
    assert load_checkpoint(new, new_opt, str(path), torch.device("cpu")) == 1234

    new_params = dict(new.named_parameters())
    old_params = dict(old.named_parameters())
    for name in ("trunk.0.weight", "history_encoder.input_proj.weight", "critic_pool_query"):
        assert torch.equal(new_params[name], old_params[name])
        assert torch.equal(new_opt.state[new_params[name]]["exp_avg"], old_opt.state[old_params[name]]["exp_avg"])
    assert new_params["history_pool_norm.weight"] not in new_opt.state
    assert len(_optimizer_param_names(new, new_opt)) == len(old_params) + 6


def test_reset_history_fuse_makes_features_static_only():
    cfg = _cfg(hist_final_norm=True, hist_pool_norm=True)
    policy = _build_policy(cfg)
    optimizer = _make_optimizer(policy, cfg)
    assert reset_history_fuse(policy, optimizer) == ["history_fuse.1", "critic_history_fuse.1"]
    obs, history, lens = _inputs()
    with torch.no_grad():
        actor, critic = policy.actor_critic_features(obs, history, lens)
        static = policy._static_features(obs, None, None)
    assert torch.equal(actor, static) and torch.equal(critic, static)


def test_pickled_legacy_config_reads_norms_off():
    from training.checkpoints import _cfg_value

    legacy = TrainConfig()
    del legacy.__dict__["hist_final_norm"], legacy.__dict__["hist_pool_norm"]  # as unpickled
    assert legacy.hist_final_norm is True  # class default leaks through getattr...
    assert _cfg_value(legacy, "hist_final_norm") is False  # ...but not through _cfg_value
    assert _cfg_value(legacy, "hist_pool_norm") is False
    assert _cfg_value(TrainConfig(), "hist_pool_norm") is True


def test_legacy_checkpoint_loads_as_trained_and_is_tagged(tmp_path):
    from training.checkpoints import load_policy_checkpoint, tag_legacy_label

    cfg = _cfg(hist_final_norm=False, hist_pool_norm=False)
    legacy = _build_policy(cfg)
    path = tmp_path / "legacy.pt"
    torch.save({"model": legacy.state_dict(), "config": cfg, "games_played": 7}, path)
    loaded = load_policy_checkpoint(str(path), torch.device("cpu"), strict=True)
    assert loaded.policy.history_pool_norm is None
    assert loaded.policy.history_encoder.encoder.norm is None
    assert tag_legacy_label("old", loaded.checkpoint) == "old-noLN"
    assert tag_legacy_label("old-noLN", loaded.checkpoint) == "old-noLN"

    current = _build_policy(_cfg())
    torch.save({"model": current.state_dict(), "config": _cfg(), "games_played": 7}, path)
    assert tag_legacy_label("new", load_policy_checkpoint(str(path), torch.device("cpu")).checkpoint) == "new"


def test_resuming_legacy_checkpoint_upgrades_and_resets_fuse(tmp_path):
    old_cfg = _cfg(hist_final_norm=False, hist_pool_norm=False)
    old = _build_policy(old_cfg)
    path = tmp_path / "old.pt"
    torch.save({"model": old.state_dict(), "optimizer": _make_optimizer(old, old_cfg).state_dict(),
                "config": old_cfg, "games_played": 5}, path)
    new = _build_policy(_cfg())
    load_checkpoint(new, _make_optimizer(new, _cfg()), str(path), torch.device("cpu"))
    assert torch.count_nonzero(new.history_fuse[1].weight) == 0
    assert torch.count_nonzero(new.critic_history_fuse[1].weight) == 0
    # resuming a checkpoint that already has the norms leaves the fuse alone
    with torch.no_grad():
        new.history_fuse[1].weight.fill_(0.5)
    torch.save({"model": new.state_dict(), "optimizer": _make_optimizer(new, _cfg()).state_dict(),
                "config": _cfg(), "games_played": 6}, path)
    again = _build_policy(_cfg())
    load_checkpoint(again, _make_optimizer(again, _cfg()), str(path), torch.device("cpu"))
    assert torch.all(again.history_fuse[1].weight == 0.5)
