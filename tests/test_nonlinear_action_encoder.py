from __future__ import annotations

import copy

import torch

from training.checkpoints import migrate_legacy_action_optimizer
from training.model_embed import ACTION_DIM, TichuPolicyEmbed
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder


def _embed_policy() -> TichuPolicyEmbed:
    return TichuPolicyEmbed(
        hidden_dim=24,
        n_layers=1,
        embed_dim=8,
        action_hidden_dim=16,
    )


def _history_policy() -> TichuPolicyHistory:
    encoder = TransformerHistoryEncoder(
        d_model=16,
        nhead=4,
        n_layers=1,
        ffn_dim=32,
    )
    return TichuPolicyHistory(
        encoder,
        hidden_dim=24,
        n_layers=1,
        embed_dim=8,
        action_hidden_dim=16,
    )


def test_new_residual_encoder_starts_as_exact_legacy_dot_product():
    torch.manual_seed(3)
    policy = _embed_policy().eval()
    obs = torch.randn(4, policy.obs_size)
    actions = torch.randn(4, 7, ACTION_DIM)

    with torch.no_grad():
        feat = policy.trunk(obs)
        query = policy.query_head(feat)
        linear_keys = policy.action_embed(actions)
        legacy_logits = (query.unsqueeze(1) * linear_keys).sum(dim=-1)
        logits, _ = policy._forward_scores(obs, actions)

    torch.testing.assert_close(logits, legacy_logits, rtol=0, atol=0)


def test_nonlinear_action_encoder_can_assign_pair_specific_features():
    policy = TichuPolicyEmbed(
        hidden_dim=8,
        n_layers=1,
        embed_dim=1,
        action_hidden_dim=2,
    )
    residual = policy.action_encoder.residual
    with torch.no_grad():
        for parameter in residual.parameters():
            parameter.zero_()
        residual[0].weight[0, 4] = 1.0
        residual[0].weight[0, 17] = 1.0
        residual[0].bias[0] = -1.5
        residual[2].weight[0, 0] = 1.0
        residual[4].weight[0, 0] = 1.0

    actions = torch.zeros(1, 3, ACTION_DIM)
    actions[0, 0, 4] = 1.0
    actions[0, 1, 17] = 1.0
    actions[0, 2, [4, 17]] = 1.0
    linear_keys = torch.zeros(1, 3, 1)

    keys = policy.action_encoder(actions, linear_keys)
    singles_sum = keys[0, 0] + keys[0, 1]
    assert not torch.allclose(keys[0, 2], singles_sum)


def test_nonlinear_keys_and_query_receive_dot_product_gradients():
    torch.manual_seed(5)
    policy = _embed_policy()
    with torch.no_grad():
        policy.action_encoder.residual[-1].weight.fill_(0.1)

    query = torch.randn(2, 8, requires_grad=True)
    actions = torch.randn(2, 5, ACTION_DIM, requires_grad=True)
    linear_keys = policy.action_embed(actions)
    keys = policy.action_encoder(actions, linear_keys)
    logits = (query.unsqueeze(1) * keys).sum(dim=-1)
    logits.square().mean().backward()

    assert query.grad is not None and query.grad.norm() > 0
    assert actions.grad is not None and actions.grad.norm() > 0
    assert policy.action_encoder.residual[0].weight.grad.norm() > 0


def test_legacy_embed_checkpoint_loads_strictly_without_changing_logits():
    torch.manual_seed(7)
    reference = _embed_policy().eval()
    legacy_state = {
        key: value.clone()
        for key, value in reference.state_dict().items()
        if not key.startswith("action_encoder.")
    }
    legacy_state["action_embed.weight"] = legacy_state[
        "action_embed.weight"
    ][:, :-1]
    obs = torch.randn(3, reference.obs_size)
    actions = torch.randn(3, 6, ACTION_DIM)
    actions[..., -1] = 0.0
    with torch.no_grad():
        expected = reference._forward_scores(obs, actions)

    torch.manual_seed(11)
    migrated = _embed_policy().eval()
    migrated.load_state_dict(legacy_state, strict=True)
    with torch.no_grad():
        actual = migrated._forward_scores(obs, actions)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert torch.count_nonzero(migrated.action_embed.weight[:, -1]) == 0


def test_legacy_history_checkpoint_loads_strictly():
    reference = _history_policy()
    legacy_state = {
        key: value.clone()
        for key, value in reference.state_dict().items()
        if not key.startswith("action_encoder.")
    }

    migrated = _history_policy()
    result = migrated.load_state_dict(legacy_state, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys


def test_legacy_optimizer_migration_preserves_old_moments():
    policy = _embed_policy()
    legacy_parameters = [
        parameter
        for name, parameter in policy.named_parameters()
        if not name.startswith("action_encoder.")
    ]
    legacy_optimizer = torch.optim.Adam(legacy_parameters, lr=3e-4)
    for parameter in legacy_parameters:
        parameter.grad = torch.ones_like(parameter)
    legacy_optimizer.step()
    legacy_state = copy.deepcopy(legacy_optimizer.state_dict())
    old_state_count = len(legacy_state["state"])

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    assert migrate_legacy_action_optimizer(policy, optimizer, legacy_state)
    optimizer.load_state_dict(legacy_state)

    assert len(optimizer.state) == old_state_count
    assert len(legacy_state["param_groups"][0]["params"]) == len(
        tuple(policy.parameters())
    )


def test_57_wide_optimizer_moments_are_zero_padded():
    legacy = TichuPolicyEmbed(
        hidden_dim=24,
        n_layers=1,
        embed_dim=8,
        action_dim=57,
        action_hidden_dim=16,
    )
    legacy_optimizer = torch.optim.Adam(legacy.parameters(), lr=3e-4)
    for parameter in legacy.parameters():
        parameter.grad = torch.ones_like(parameter)
    legacy_optimizer.step()
    state = copy.deepcopy(legacy_optimizer.state_dict())

    policy = _embed_policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    assert migrate_legacy_action_optimizer(policy, optimizer, state)
    optimizer.load_state_dict(state)

    action_state = optimizer.state[policy.action_embed.weight]
    assert action_state["exp_avg"].shape[1] == ACTION_DIM
    assert action_state["exp_avg_sq"].shape[1] == ACTION_DIM
    assert torch.count_nonzero(action_state["exp_avg"][:, -1]) == 0
    assert torch.count_nonzero(action_state["exp_avg_sq"][:, -1]) == 0
