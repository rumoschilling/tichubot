import copy

import numpy as np
import torch

from training.checkpoints import migrate_legacy_pass_optimizer
from training.model_embed import TichuPolicyEmbed
from training.model_passing import (
    HAND_SIZE,
    NUM_CARDS,
    PASS_OBS_DIM,
    TichuPassingPolicy,
)
from training.rollout_embed import collect_rollouts_embed


def _fixed_inputs(batch_size: int = 4):
    obs = torch.zeros(batch_size, PASS_OBS_DIM)
    hand = torch.arange(HAND_SIZE).repeat(batch_size, 1)
    obs[:, :HAND_SIZE] = 1.0
    choices = torch.tensor([[0, 1, 2]]).repeat(batch_size, 1)
    return obs, hand, choices


class _AlwaysDeclareGT:
    def act(self, h8_binary):
        batch_size = len(h8_binary)
        return (
            torch.ones(
                batch_size, dtype=torch.int64, device=h8_binary.device
            ),
            torch.zeros(batch_size, device=h8_binary.device),
        )


def test_legacy_positional_checkpoint_keeps_identical_policy():
    torch.manual_seed(4)
    reference = TichuPassingPolicy()
    legacy_state = {
        key: value.clone()
        for key, value in reference.state_dict().items()
        if not key.startswith(("card_embed.", "card_query."))
    }

    torch.manual_seed(99)
    migrated = TichuPassingPolicy()
    migrated.load_state_dict(legacy_state)

    obs, hand, choices = _fixed_inputs()
    ref_lp, ref_entropy = reference.evaluate(obs, hand, choices)
    new_lp, new_entropy = migrated.evaluate(obs, hand, choices)

    assert torch.equal(ref_lp, new_lp)
    assert torch.equal(ref_entropy, new_entropy)
    assert migrated.card_query.weight.abs().sum() == 0
    assert migrated.card_query.bias.abs().sum() == 0


def test_candidate_scorer_tracks_card_identity_across_hand_positions():
    policy = TichuPassingPolicy(card_embed_dim=4)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.card_query.bias[0] = 1.0
        policy.card_embed.weight[10, 0] = 2.0

    hand_cards = torch.tensor([
        list(range(14)),
        [10, *range(14, 27)],
    ])
    trunk_feat = torch.zeros(2, policy.hidden_dim)
    picked = torch.zeros(2, NUM_CARDS)
    mask = torch.ones(2, HAND_SIZE, dtype=torch.bool)

    logits = policy._pointer_logits(
        trunk_feat, 0, picked, hand_cards, mask
    )

    assert logits.argmax(dim=1).tolist() == [10, 0]


def test_optimizer_migration_preserves_old_moments_and_adds_new_parameters():
    policy = TichuPassingPolicy()
    legacy_parameters = list(policy.parameters())[:-3]
    legacy_optimizer = torch.optim.Adam(legacy_parameters, lr=3e-4)
    for parameter in legacy_parameters:
        parameter.grad = torch.ones_like(parameter)
    legacy_optimizer.step()
    state = copy.deepcopy(legacy_optimizer.state_dict())
    old_state_count = len(state["state"])

    assert migrate_legacy_pass_optimizer(
        state, expected_params=len(tuple(policy.parameters()))
    )
    assert len(state["param_groups"][0]["params"]) == len(
        tuple(policy.parameters())
    )

    migrated_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    migrated_optimizer.load_state_dict(state)
    assert len(migrated_optimizer.state) == old_state_count


def test_embed_rollout_executes_and_harvests_passing_decisions():
    torch.manual_seed(7)
    n_games = 2
    rollout = collect_rollouts_embed(
        TichuPolicyEmbed(hidden_dim=32, n_layers=1, embed_dim=8),
        n_games,
        torch.device("cpu"),
        with_calling=True,
        gt_policy=_AlwaysDeclareGT(),
        pass_policy=TichuPassingPolicy(hidden_dim=32, n_layers=1),
    )

    assert rollout["pass_train_obs"].shape == (4 * n_games, PASS_OBS_DIM)
    assert rollout["pass_train_hand_cards"].shape == (
        4 * n_games, HAND_SIZE
    )
    assert rollout["pass_train_choices"].shape == (4 * n_games, 3)
    assert rollout["pass_train_log_probs"].shape == (4 * n_games, 3)
    assert rollout["pass_train_returns"].shape == (4 * n_games,)
    assert np.all(rollout["pass_train_obs"][:, :NUM_CARDS].sum(axis=1) == 14)
    assert np.all(rollout["pass_train_obs"][:, NUM_CARDS:] == 1)

    per_game_returns = rollout["pass_train_returns"].reshape(n_games, 4)
    assert np.allclose(per_game_returns.sum(axis=1), 0.0, atol=1e-6)
    assert np.allclose(
        np.abs(per_game_returns),
        np.abs(per_game_returns[:, :1]),
        atol=1e-6,
    )
