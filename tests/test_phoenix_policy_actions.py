from __future__ import annotations

import copy
import random

import numpy as np
import torch

from tichu_env.cards import PHOENIX_INDEX, Rank, Suit, card_index
from tichu_env.combinations import (
    CombinationType,
    binary_phoenix_interpretations,
)
from tichu_env.game_state import GameState
from training.model_embed import ACTION_DIM
from training.play import ai_select_action
from training.rollout_embed import _action_vec


J, P, S, W = Suit.JADE, Suit.PAGODA, Suit.STAR, Suit.SWORD


def ci(suit: Suit, rank: Rank) -> int:
    return card_index(suit, rank)


def game_with_hands(h0, h1, h2, h3, current_player=0) -> GameState:
    game = GameState()
    game.hands = [set(h0), set(h1), set(h2), set(h3)]
    game.current_player = current_player
    game.finish_order = []
    game.won_cards = [[] for _ in range(4)]
    game.played_cards = set()
    game.round_over = False
    return game


def _matching(actions, cards, *, call_tichu=False):
    wanted = frozenset(cards)
    return [
        action
        for action in actions
        if frozenset(action[0]) == wanted and action[1] is call_tichu
    ]


def test_ambiguous_straight_has_lower_and_higher_policy_actions():
    cards = [
        ci(J, Rank.THREE),
        ci(P, Rank.FOUR),
        ci(S, Rank.FIVE),
        ci(W, Rank.SIX),
        PHOENIX_INDEX,
    ]
    game = game_with_hands(
        cards + [ci(J, Rank.ACE)],
        [ci(P, Rank.TWO)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )

    variants = _matching(game.get_policy_legal_actions(0), cards)
    assert [action[2] for action in variants] == [False, True]

    ranks = []
    phoenix_ranks = []
    for action in variants:
        trial = copy.deepcopy(game)
        assert trial.play(0, action)
        ranks.append(trial.trick.leading_combo.rank)
        phoenix_ranks.append(trial.trick.leading_combo.phoenix_as)
    assert ranks == [6.0, 7.0]
    assert phoenix_ranks == [2, 7]


def test_ambiguous_full_house_selects_lower_or_higher_triple():
    cards = [
        ci(J, Rank.FIVE),
        ci(P, Rank.FIVE),
        ci(S, Rank.NINE),
        ci(W, Rank.NINE),
        PHOENIX_INDEX,
    ]
    interpretations = binary_phoenix_interpretations(cards)
    assert interpretations is not None
    assert [combo.type for combo in interpretations] == [
        CombinationType.FULL_HOUSE,
        CombinationType.FULL_HOUSE,
    ]
    assert [combo.rank for combo in interpretations] == [5.0, 9.0]

    game = game_with_hands(
        cards + [ci(J, Rank.ACE)],
        [ci(P, Rank.TWO)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )
    variants = _matching(game.get_policy_legal_actions(0), cards)
    assert [action[2] for action in variants] == [False, True]

    for action, expected_rank in zip(variants, (5.0, 9.0), strict=True):
        trial = copy.deepcopy(game)
        assert trial.play(0, action)
        assert trial.trick.leading_combo.rank == expected_rank


def test_only_beating_phoenix_interpretation_is_offered_when_following():
    lead = [
        ci(J, Rank.TWO),
        ci(P, Rank.THREE),
        ci(S, Rank.FOUR),
        ci(W, Rank.FIVE),
        ci(J, Rank.SIX),
    ]
    response = [
        ci(S, Rank.THREE),
        ci(W, Rank.FOUR),
        ci(J, Rank.FIVE),
        ci(P, Rank.SIX),
        PHOENIX_INDEX,
    ]
    game = game_with_hands(
        lead + [ci(J, Rank.ACE)],
        response + [ci(P, Rank.ACE)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )
    assert game.play(0, lead)

    variants = _matching(game.get_policy_legal_actions(1), response)
    assert len(variants) == 1
    assert variants[0][2] is True
    assert game.play(1, variants[0])
    assert game.trick.leading_combo.rank == 7.0


def test_phoenix_choice_changes_only_the_final_action_bit():
    cards = [
        ci(J, Rank.THREE),
        ci(P, Rank.FOUR),
        ci(S, Rank.FIVE),
        ci(W, Rank.SIX),
        PHOENIX_INDEX,
    ]
    lower = _action_vec((cards, False, False))
    higher = _action_vec((cards, False, True))

    assert lower.shape == (ACTION_DIM,)
    np.testing.assert_array_equal(lower[:-1], higher[:-1])
    assert lower[-1] == 0
    assert higher[-1] == 1


def test_unambiguous_phoenix_pair_has_no_high_variant():
    cards = [ci(J, Rank.FIVE), PHOENIX_INDEX]
    game = game_with_hands(
        cards + [ci(J, Rank.ACE)],
        [ci(P, Rank.TWO)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )
    variants = _matching(game.get_policy_legal_actions(0), cards)
    assert variants == [(sorted(cards), False, False)]


def test_deployed_embed_path_can_select_and_execute_higher_interpretation():
    class PreferHighPolicy:
        obs_size = 196

        def act(self, obs, action_vecs, n_legal, *unused):
            scores = action_vecs[..., -1]
            action = scores.argmax(dim=1)
            zeros = torch.zeros(obs.shape[0], device=obs.device)
            return action, zeros, zeros

    cards = [
        ci(J, Rank.THREE),
        ci(P, Rank.FOUR),
        ci(S, Rank.FIVE),
        ci(W, Rank.SIX),
        PHOENIX_INDEX,
    ]
    game = game_with_hands(
        cards + [ci(J, Rank.ACE)],
        [ci(P, Rank.TWO)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )

    action = ai_select_action(
        PreferHighPolicy(),
        game,
        0,
        torch.device("cpu"),
        random.Random(0),
        "embed",
    )
    assert action[2] is True
    assert game.play(0, action)
    assert game.trick.leading_combo.rank == 7.0
