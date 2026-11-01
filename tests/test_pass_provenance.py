import copy

import pytest
import torch

from tichu_env.encoding import (
    encode_pass_source_ids,
    encode_pass_target_ids,
)
from tichu_env.game_state import GameState
from training.checkpoints import revive_dead_pass_provenance
from training.model_embed import ACTION_DIM, TichuPolicyEmbed
from training.model_history import (
    TichuPolicyHistory,
    TransformerHistoryEncoder,
)


def _complete_fixed_pass(seed: int = 42):
    game = GameState()
    game.reset(seed=seed, with_passing=True)
    submitted = {}
    for player in range(4):
        hand = game.get_passing_hand(player)
        submitted[player] = tuple(hand[:3])
        assert game.pass_cards(player, *submitted[player])
    return game, submitted


def test_pass_provenance_uses_observer_relative_left_partner_right_order():
    game, submitted = _complete_fixed_pass()

    for observer in range(4):
        left = (observer + 1) % 4
        partner = (observer + 2) % 4
        right = (observer + 3) % 4

        assert encode_pass_source_ids(game, observer).tolist() == [
            submitted[left][2],
            submitted[partner][1],
            submitted[right][0],
        ]
        assert encode_pass_target_ids(game, observer).tolist() == list(
            submitted[observer]
        )


def _build_embed():
    return TichuPolicyEmbed(hidden_dim=32, n_layers=1, embed_dim=8)


def _build_history():
    encoder = TransformerHistoryEncoder(
        d_model=16,
        nhead=4,
        n_layers=1,
        ffn_dim=32,
    )
    return TichuPolicyHistory(
        encoder,
        hidden_dim=32,
        n_layers=1,
        embed_dim=8,
    )


def _forward(policy, source_ids, target_ids):
    obs = torch.randn(2, 196)
    action_vecs = torch.randn(2, 4, ACTION_DIM)
    if isinstance(policy, TichuPolicyHistory):
        history = torch.zeros(2, 1, 63)
        history_lens = torch.zeros(2, dtype=torch.int32)
        return policy._forward_scores(
            obs,
            action_vecs,
            history,
            history_lens,
            source_ids,
            target_ids,
        )
    return policy._forward_scores(
        obs,
        action_vecs,
        source_ids,
        target_ids,
    )


@pytest.mark.parametrize("factory", [_build_embed, _build_history])
def test_fresh_provenance_branch_is_neutral_but_has_nonzero_card_embeddings(factory):
    torch.manual_seed(8)
    policy = factory()
    source_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    target_ids = torch.tensor([[7, 8, 9], [10, 11, 12]])

    torch.manual_seed(17)
    with_provenance = _forward(policy, source_ids, target_ids)
    torch.manual_seed(17)
    without_provenance = _forward(policy, None, None)

    for actual, expected in zip(with_provenance, without_provenance):
        assert torch.equal(actual, expected)
    assert torch.count_nonzero(policy.pass_source_embed.weight) > 0
    assert torch.count_nonzero(policy.pass_target_embed.weight) > 0
    assert torch.count_nonzero(policy.pass_source_proj.weight) == 0
    assert torch.count_nonzero(policy.pass_target_proj.weight) == 0


@pytest.mark.parametrize("factory", [_build_embed, _build_history])
def test_fresh_provenance_projection_receives_card_specific_gradient(factory):
    torch.manual_seed(9)
    policy = factory()
    source_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    target_ids = torch.tensor([[7, 8, 9], [10, 11, 12]])

    logits, values = _forward(policy, source_ids, target_ids)
    (logits.square().mean() + values.square().mean()).backward()

    assert policy.pass_source_proj.weight.grad.norm() > 0
    assert policy.pass_target_proj.weight.grad.norm() > 0


def test_dead_checkpoint_migration_preserves_outputs_and_resets_only_branch_state():
    torch.manual_seed(10)
    policy = _build_embed()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    source_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    target_ids = torch.tensor([[7, 8, 9], [10, 11, 12]])

    with torch.no_grad():
        policy.pass_source_embed.weight.zero_()
        policy.pass_target_embed.weight.zero_()
        policy.pass_source_proj.weight.zero_()
        policy.pass_target_proj.weight.zero_()

    logits, values = _forward(policy, source_ids, target_ids)
    (logits.square().mean() + values.square().mean()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    before_state = {
        parameter: copy.deepcopy(state)
        for parameter, state in optimizer.state.items()
    }
    torch.manual_seed(23)
    expected = _forward(policy, source_ids, target_ids)

    assert revive_dead_pass_provenance(
        policy,
        optimizer,
        context="test checkpoint",
    ) == ("pass_source", "pass_target")

    torch.manual_seed(23)
    actual = _forward(policy, source_ids, target_ids)
    for migrated, original in zip(actual, expected):
        assert torch.equal(migrated, original)

    reset_parameters = {
        policy.pass_source_embed.weight,
        policy.pass_source_proj.weight,
        policy.pass_target_embed.weight,
        policy.pass_target_proj.weight,
    }
    assert all(parameter not in optimizer.state for parameter in reset_parameters)
    for parameter, old_state in before_state.items():
        if parameter in reset_parameters:
            continue
        assert parameter in optimizer.state
        assert optimizer.state[parameter].keys() == old_state.keys()
        for key, old_value in old_state.items():
            current = optimizer.state[parameter][key]
            if torch.is_tensor(old_value):
                assert torch.equal(current, old_value)
            else:
                assert current == old_value

    torch.manual_seed(31)
    logits, values = _forward(policy, source_ids, target_ids)
    (logits.square().mean() + values.square().mean()).backward()
    optimizer.step()

    assert torch.count_nonzero(policy.pass_source_proj.weight) > 0
    assert torch.count_nonzero(policy.pass_target_proj.weight) > 0

    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(37)
    learned = _forward(policy, source_ids, target_ids)
    (learned[0].square().mean() + learned[1].square().mean()).backward()
    assert policy.pass_source_embed.weight.grad.norm() > 0
    assert policy.pass_target_embed.weight.grad.norm() > 0

    torch.manual_seed(41)
    original_ids = _forward(policy, source_ids, target_ids)
    torch.manual_seed(41)
    changed_ids = _forward(policy, source_ids.flip(1), target_ids.flip(1))
    assert any(
        not torch.equal(original, changed)
        for original, changed in zip(original_ids, changed_ids)
    )
