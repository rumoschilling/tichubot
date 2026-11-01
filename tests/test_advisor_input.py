import copy
import random

import torch

from tichu_env.cards import (
    DOG_INDEX,
    DRAGON_INDEX,
    MAHJONG_INDEX,
    PHOENIX_INDEX,
    Rank,
    Suit,
    card_index,
)
from tichu_env.game_state import GamePhase, GameState
from training.advisor import (
    UNDO,
    force_play,
    legal_phoenix_meanings,
    parse_card,
    parse_cards,
    parse_wish,
    suggest,
)
from training.model import TichuPolicy


def test_special_card_shortcuts_match_advisor_input_convention():
    assert parse_card("m") == MAHJONG_INDEX
    assert parse_card("d") == DRAGON_INDEX
    assert parse_card("w") == DOG_INDEX
    assert parse_card("p") == PHOENIX_INDEX


def test_repeated_bare_ranks_pair_positionally_with_suits():
    assert parse_cards("2 2 3 3 r g s b") == [
        card_index(Suit.STAR, Rank.TWO),
        card_index(Suit.JADE, Rank.TWO),
        card_index(Suit.SWORD, Rank.THREE),
        card_index(Suit.PAGODA, Rank.THREE),
    ]


def test_rank_range_expands_before_pairing_with_suits():
    assert parse_cards("2-7 r g b s g s") == [
        card_index(Suit.STAR, Rank.TWO),
        card_index(Suit.JADE, Rank.THREE),
        card_index(Suit.PAGODA, Rank.FOUR),
        card_index(Suit.SWORD, Rank.FIVE),
        card_index(Suit.JADE, Rank.SIX),
        card_index(Suit.SWORD, Rank.SEVEN),
    ]


def test_one_rank_broadcasts_over_compact_suits():
    assert parse_cards("2 rgb") == [
        card_index(Suit.STAR, Rank.TWO),
        card_index(Suit.JADE, Rank.TWO),
        card_index(Suit.PAGODA, Rank.TWO),
    ]


def test_one_suit_broadcasts_over_a_rank_range():
    assert parse_cards("2-7 r") == [
        card_index(Suit.STAR, rank)
        for rank in (
            Rank.TWO,
            Rank.THREE,
            Rank.FOUR,
            Rank.FIVE,
            Rank.SIX,
            Rank.SEVEN,
        )
    ]


def test_rank_bomb_expands_to_all_four_suits():
    assert parse_cards("J bomb") == [
        card_index(Suit.STAR, Rank.JACK),
        card_index(Suit.JADE, Rank.JACK),
        card_index(Suit.PAGODA, Rank.JACK),
        card_index(Suit.SWORD, Rank.JACK),
    ]


def test_compact_groups_can_be_mixed_with_normal_and_special_cards():
    assert parse_cards("m, 2 3 r g, d w 10b") == [
        MAHJONG_INDEX,
        card_index(Suit.STAR, Rank.TWO),
        card_index(Suit.JADE, Rank.THREE),
        DRAGON_INDEX,
        DOG_INDEX,
        card_index(Suit.PAGODA, Rank.TEN),
    ]


def test_compact_group_rejects_mismatched_rank_and_suit_counts():
    assert parse_cards("2-7 r g b") is None
    assert parse_cards("2 3 4 r g") is None
    assert parse_cards("2-3 bomb") is None


def test_force_play_rejection_is_transactional():
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 1
    gs.hands = [{55}, {0, 1, 2}, {14}, {28}]
    gs._incoming_pass_source[1][2] = 0
    gs._outgoing_pass_target[2][1] = 0
    gs._invalidate_cache()
    _ = gs.sorted_hands

    before = copy.deepcopy(gs)
    random.seed(417)
    random_state = random.getstate()

    assert not force_play(gs, 1, [14, 28], 0, call_tichu=True)

    assert gs.hands == before.hands
    assert gs.tichu_declarations == before.tichu_declarations
    assert gs.current_player == before.current_player
    assert gs.played_cards == before.played_cards
    assert gs.play_log == before.play_log
    assert gs.trick_history == before.trick_history
    assert gs._incoming_pass_source == before._incoming_pass_source
    assert gs._outgoing_pass_target == before._outgoing_pass_target
    assert gs._sorted_hands_cache == before._sorted_hands_cache
    assert random.getstate() == random_state


def test_force_play_rejects_card_with_different_known_pass_recipient():
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 2
    gs.hands = [{55}, {MAHJONG_INDEX, 0}, {14, 15}, {28}]
    gs._outgoing_pass_target[0][1] = MAHJONG_INDEX
    gs._incoming_pass_source[1][0] = MAHJONG_INDEX
    before = copy.deepcopy(gs)

    assert not force_play(gs, 2, [MAHJONG_INDEX], 0)

    assert gs.hands == before.hands
    assert gs.current_player == before.current_player
    assert gs.play_log == before.play_log
    assert gs._incoming_pass_source == before._incoming_pass_source
    assert gs._outgoing_pass_target == before._outgoing_pass_target


def test_force_play_uses_unknown_replacement_and_preserves_known_pass():
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 2
    gs.hands = [{55}, {0}, {14, 15}, {28}]
    gs._outgoing_pass_target[0][2] = 14
    gs._incoming_pass_source[2][0] = 14

    assert force_play(gs, 2, [28], 0)

    assert [len(hand) for hand in gs.hands] == [1, 1, 1, 1]
    assert gs.hands[2] == {14}
    assert gs.hands[3] == {15}
    assert gs._outgoing_pass_target[0][2] == 14
    assert gs._incoming_pass_source[2][0] == 14


def test_force_play_without_unknown_replacement_rejects_transactionally():
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 2
    gs.hands = [{55}, {0}, {14}, {28}]
    gs._outgoing_pass_target[0][2] = 14
    gs._incoming_pass_source[2][0] = 14
    before = copy.deepcopy(gs)
    random.seed(932)
    random_state = random.getstate()

    assert not force_play(gs, 2, [28], 0)

    assert gs.hands == before.hands
    assert gs.current_player == before.current_player
    assert gs.play_log == before.play_log
    assert gs._incoming_pass_source == before._incoming_pass_source
    assert gs._outgoing_pass_target == before._outgoing_pass_target
    assert random.getstate() == random_state



def test_fixed_policy_advisor_suggestions_unpack_forward_result():
    gs = GameState()
    gs.reset(seed=7)
    policy = TichuPolicy(hidden_dim=16, n_layers=1)

    suggestions = suggest(
        policy,
        gs,
        gs.current_player,
        torch.device("cpu"),
        top_n=3,
        kind="fixed",
    )

    assert len(suggestions) == 3
    assert all(len(item) == 4 for item in suggestions)


def _wish_state(wish_rank: int, p1_hand: set[int]) -> GameState:
    """P3 led the Mahjong with a wish; the human seat 0 has passed."""
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 3
    gs.hands = [
        {DRAGON_INDEX, card_index(Suit.JADE, Rank.TWO)},
        set(p1_hand),
        {card_index(Suit.STAR, Rank.TWO), card_index(Suit.STAR, Rank.THREE)},
        {MAHJONG_INDEX, card_index(Suit.SWORD, Rank.TWO)},
    ]
    gs._invalidate_cache()
    assert gs.play(3, [MAHJONG_INDEX])
    gs.set_mahjong_wish(wish_rank)
    assert gs.play(0, [])
    assert gs.current_player == 1
    return gs


def test_force_play_accepts_opponent_pass_despite_guessed_wished_card():
    # The simulated P1 hand happens to hold the wished 7. The real P1 passed,
    # so they could not fulfil the wish: the guess must not veto the pass.
    seven = card_index(Suit.JADE, Rank.SEVEN)
    gs = _wish_state(7, {seven, card_index(Suit.JADE, Rank.NINE)})
    assert not copy.deepcopy(gs).play(1, [])  # the engine alone rejects it

    assert force_play(gs, 1, [], 0)

    assert gs.current_player == 2
    assert gs.mahjong_wish == 7


def test_force_play_accepts_opponent_play_that_leaves_wish_open():
    seven = card_index(Suit.JADE, Rank.SEVEN)
    nine = card_index(Suit.JADE, Rank.NINE)
    gs = _wish_state(7, {seven, nine})

    assert force_play(gs, 1, [nine], 0)

    assert gs.current_player == 2
    assert gs.mahjong_wish == 7


def test_force_play_fulfilling_the_wish_clears_it():
    seven = card_index(Suit.JADE, Rank.SEVEN)
    gs = _wish_state(7, {seven, card_index(Suit.JADE, Rank.NINE)})

    assert force_play(gs, 1, [seven], 0)

    assert gs.mahjong_wish is None


def _phoenix_full_house_state(lead_ten_full_house: bool) -> tuple[GameState, list[int]]:
    queens_nines = [
        card_index(Suit.JADE, Rank.QUEEN),
        card_index(Suit.PAGODA, Rank.QUEEN),
        card_index(Suit.JADE, Rank.NINE),
        card_index(Suit.PAGODA, Rank.NINE),
        PHOENIX_INDEX,
    ]
    tens_threes = [
        card_index(Suit.JADE, Rank.TEN),
        card_index(Suit.PAGODA, Rank.TEN),
        card_index(Suit.STAR, Rank.TEN),
        card_index(Suit.JADE, Rank.THREE),
        card_index(Suit.PAGODA, Rank.THREE),
    ]
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 3 if lead_ten_full_house else 0
    gs.hands = [
        set(queens_nines) | {card_index(Suit.SWORD, Rank.TWO)},
        {card_index(Suit.STAR, Rank.TWO)},
        {card_index(Suit.STAR, Rank.FOUR)},
        set(tens_threes) | {card_index(Suit.SWORD, Rank.FOUR)},
    ]
    gs._invalidate_cache()
    if lead_ten_full_house:
        assert gs.play(3, tens_threes)
    return gs, queens_nines


def test_typed_phoenix_full_house_uses_the_only_legal_meaning():
    gs, cards = _phoenix_full_house_state(lead_ten_full_house=True)

    meanings = legal_phoenix_meanings(gs, 0, 0, cards)

    # Queens-full beats tens-full; nines-full (the lower meaning) does not.
    assert [combo.phoenix_as for combo in meanings] == [int(Rank.QUEEN)]
    assert gs.play(0, (cards, False, False), phoenix_as=meanings[0].phoenix_as)


def test_typed_phoenix_full_house_leading_offers_both_meanings():
    gs, cards = _phoenix_full_house_state(lead_ten_full_house=False)
    before = copy.deepcopy(gs)

    meanings = legal_phoenix_meanings(gs, 0, 0, cards)

    assert [combo.phoenix_as for combo in meanings] == [int(Rank.NINE), int(Rank.QUEEN)]
    assert gs.hands == before.hands
    assert gs.current_player == before.current_player


def test_wish_input_accepts_rank_letters_and_rejects_garbage():
    assert parse_wish("J") == int(Rank.JACK)
    assert parse_wish("a") == int(Rank.ACE)
    assert parse_wish("10") == 10
    assert parse_wish("") is None
    assert parse_wish("0") is None
    assert parse_wish("u") == UNDO
    assert parse_wish("x") == ""
    assert parse_wish("15") == ""
