from __future__ import annotations

import torch

from training.checkpoints import load_policy_checkpoint
from training.config import TrainConfig
from training.model_embed import ACTION_DIM
from training.play import load_play_bundle
from training.train_history import _build_policy


def _config() -> TrainConfig:
    return TrainConfig(
        hidden_dim=32,
        n_layers=3,
        embed_dim=16,
        action_hidden_dim=20,
        trunk_dims=(48, 40, 32),
        mlp_activation="silu",
        mlp_layer_norm=True,
        hist_d_model=24,
        hist_nhead=4,
        hist_n_layers=1,
        hist_ffn_dim=48,
        hist_norm_first=True,
        hist_activation="gelu",
        split_actor_critic_history=True,
        static_attention=True,
    )


def _inputs():
    generator = torch.Generator().manual_seed(17)
    obs = torch.randn(6, 196, generator=generator)
    history = torch.randn(6, 9, 63, generator=generator)
    history_len = torch.tensor([1, 2, 4, 6, 8, 9], dtype=torch.int32)
    actions = torch.randn(6, 5, ACTION_DIM, generator=generator)
    pass_ids = torch.full((6, 3), 56, dtype=torch.int64)
    return obs, actions, history, history_len, pass_ids


def _linear(module: torch.nn.Sequential) -> torch.nn.Linear:
    return next(layer for layer in module if isinstance(layer, torch.nn.Linear))


def test_split_paths_change_only_their_own_output():
    policy = _build_policy(_config()).eval()
    obs, actions, history, history_len, pass_ids = _inputs()

    with torch.no_grad():
        logits_before, value_before = policy._forward_scores(
            obs, actions, history, history_len, pass_ids, pass_ids
        )
        _linear(policy.critic_history_fuse).weight[:, 0].add_(0.1)
        logits_after, value_after = policy._forward_scores(
            obs, actions, history, history_len, pass_ids, pass_ids
        )

    torch.testing.assert_close(logits_after, logits_before)
    assert not torch.allclose(value_after, value_before)


def test_actor_and_critic_readout_gradients_are_isolated():
    policy = _build_policy(_config())
    obs, actions, history, history_len, pass_ids = _inputs()

    logits, _ = policy._forward_scores(
        obs, actions, history, history_len, pass_ids, pass_ids
    )
    logits.square().mean().backward()
    assert policy.history_encoder.pool_query.grad is not None
    assert policy.critic_pool_query.grad is None
    assert _linear(policy.history_fuse).weight.grad is not None
    assert _linear(policy.critic_history_fuse).weight.grad is None

    policy.zero_grad(set_to_none=True)
    _, value = policy._forward_scores(
        obs, actions, history, history_len, pass_ids, pass_ids
    )
    value.square().mean().backward()
    assert policy.history_encoder.pool_query.grad is None
    assert policy.critic_pool_query.grad is not None
    assert _linear(policy.history_fuse).weight.grad is None
    assert _linear(policy.critic_history_fuse).weight.grad is not None


def test_compact_split_checkpoint_round_trip(tmp_path):
    cfg = _config()
    policy = _build_policy(cfg).eval()
    checkpoint = tmp_path / "compact.pt"
    torch.save(
        {"model": policy.state_dict(), "config": cfg, "games_played": 0},
        checkpoint,
    )

    loaded = load_policy_checkpoint(checkpoint)
    assert loaded.kind == "history"
    assert loaded.policy.split_actor_critic_history
    assert loaded.policy.static_attention
    assert not loaded.policy.history_encoder.use_checkpointing
    assert hasattr(loaded.policy, "critic_pool_attn")
    assert loaded.policy.action_encoder.residual[0].out_features == 20

    obs, actions, history, history_len, pass_ids = _inputs()
    with torch.no_grad():
        expected = policy._forward_scores(
            obs, actions, history, history_len, pass_ids, pass_ids
        )
        actual = loaded.policy._forward_scores(
            obs, actions, history, history_len, pass_ids, pass_ids
        )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_play_loader_preserves_compact_split_architecture(tmp_path):
    cfg = _config()
    expected = _build_policy(cfg).eval()
    checkpoint = tmp_path / "compact.pt"
    torch.save(
        {"model": expected.state_dict(), "config": cfg, "games_played": 123},
        checkpoint,
    )

    policy, pass_policy, gt_policy, kind, info = load_play_bundle(
        str(checkpoint), None, torch.device("cpu")
    )

    assert kind == "history"
    assert policy.split_actor_critic_history
    assert policy.static_attention
    assert [
        layer.out_features for layer in policy.trunk if isinstance(layer, torch.nn.Linear)
    ] == [48, 40, 32]
    assert pass_policy is None
    assert gt_policy is None
    assert info["games_played"] == 123


def test_static_state_conditions_actor_and_critic_pool_queries():
    policy = _build_policy(_config()).eval()
    obs, _, _, _, pass_ids = _inputs()

    with torch.no_grad():
        static_feat = policy._static_features(obs, pass_ids, pass_ids)
        actor_query = policy._history_pool_query(static_feat)
        critic_query = policy._history_pool_query(static_feat, critic=True)

    assert actor_query.shape == (obs.shape[0], 1, policy.history_encoder.output_dim)
    assert critic_query.shape == actor_query.shape
    assert not torch.allclose(actor_query[0], actor_query[1])
    assert not torch.allclose(critic_query[0], critic_query[1])
    assert not torch.allclose(actor_query, critic_query)


def test_critic_static_attention_changes_only_value_output():
    policy = _build_policy(_config()).eval()
    obs, actions, history, history_len, pass_ids = _inputs()

    with torch.no_grad():
        logits_before, value_before = policy._forward_scores(
            obs, actions, history, history_len, pass_ids, pass_ids
        )
        policy.critic_history_static_query.weight.add_(0.1)
        logits_after, value_after = policy._forward_scores(
            obs, actions, history, history_len, pass_ids, pass_ids
        )

    torch.testing.assert_close(logits_after, logits_before)
    assert not torch.allclose(value_after, value_before)
