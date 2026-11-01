from __future__ import annotations

import copy

import numpy as np

from tichu_env.cards import (
    DRAGON_INDEX,
    DOG_INDEX,
    MAHJONG_INDEX,
    PHOENIX_INDEX,
    Rank,
    Suit,
    card_index,
)
from tichu_env.combinations import (
    CombinationType,
    classify_cards,
    phoenix_interpretations,
)
from tichu_env.env import TichuEnv
from tichu_env.game_state import GamePhase, GameState, TrickState


J, P, S, W = Suit.JADE, Suit.PAGODA, Suit.STAR, Suit.SWORD


def ci(suit: Suit, rank: Rank) -> int:
    return card_index(suit, rank)


def game_with_hands(h0, h1, h2, h3, *, current_player=0, track_history=False):
    game = GameState(track_history=track_history)
    if track_history:
        game.reset(seed=0)
        game.play_log = []
        game.history_tokens.fill(0)
        game.player_indices.fill(0)
    game.hands = [set(h0), set(h1), set(h2), set(h3)]
    game.current_player = current_player
    game.finish_order = []
    game.finished = set()
    game.won_cards = [[] for _ in range(4)]
    game.trick = TrickState()
    game.tichu_declarations = [0, 0, 0, 0]
    game.mahjong_wish = None
    game.dragon_give_pending = None
    game.played_cards = set()
    game.trick_history = []
    game.trick_num = 0
    game._has_played = set()
    game.round_over = False
    game.round_scores = (0, 0)
    game.phase = GamePhase.PLAY
    game._invalidate_cache()
    return game


def state_fingerprint(game: GameState):
    return (
        tuple(frozenset(hand) for hand in game.hands),
        game.current_player,
        tuple(game.finish_order),
        frozenset(game.finished),
        tuple(tuple(cards) for cards in game.won_cards),
        copy.deepcopy(game.trick),
        tuple(game.tichu_declarations),
        game.mahjong_wish,
        game.dragon_give_pending,
        frozenset(game.played_cards),
        tuple((p, tuple(cards), kind) for p, cards, kind in game.trick_history),
        tuple((p, tuple(cards), n, opener) for p, cards, n, opener in game.play_log),
        game.trick_num,
        frozenset(game._has_played),
        game.round_over,
        game.round_scores,
    )


def test_phoenix_can_replace_a_straight_rank_still_present_in_hand():
    hand = [
        ci(J, Rank.TWO),
        ci(J, Rank.THREE),
        ci(J, Rank.FOUR),
        ci(P, Rank.FOUR),
        ci(J, Rank.FIVE),
        ci(J, Rank.SIX),
        PHOENIX_INDEX,
    ]
    selected = {
        ci(J, Rank.TWO),
        ci(J, Rank.THREE),
        PHOENIX_INDEX,
        ci(J, Rank.FIVE),
        ci(J, Rank.SIX),
    }
    game = game_with_hands(
        hand,
        [ci(P, Rank.TWO)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )

    assert classify_cards(sorted(selected)).type == CombinationType.STRAIGHT
    for actions in (
        game.get_legal_actions(0),
        game.get_policy_legal_actions(0),
    ):
        assert any(set(action[0]) == selected for action in actions)


def test_phoenix_can_replace_a_steps_card_still_present_in_hand():
    hand = [
        ci(J, Rank.TWO),
        ci(P, Rank.TWO),
        ci(J, Rank.THREE),
        ci(P, Rank.THREE),
        PHOENIX_INDEX,
    ]
    selected = {
        ci(J, Rank.TWO),
        PHOENIX_INDEX,
        ci(J, Rank.THREE),
        ci(P, Rank.THREE),
    }
    game = game_with_hands(
        hand,
        [ci(P, Rank.FOUR)],
        [ci(S, Rank.FOUR)],
        [ci(W, Rank.FOUR)],
    )

    assert classify_cards(sorted(selected)).type == CombinationType.STEPS
    for actions in (
        game.get_legal_actions(0),
        game.get_policy_legal_actions(0),
    ):
        assert any(set(action[0]) == selected for action in actions)


def test_every_emitted_phoenix_action_executes_and_wish_filter_is_preserved():
    game = game_with_hands(
        [
            ci(J, Rank.TWO), ci(P, Rank.TWO),
            ci(J, Rank.THREE), ci(P, Rank.THREE),
            ci(J, Rank.FOUR), ci(P, Rank.FOUR),
            ci(J, Rank.FIVE), ci(J, Rank.SIX),
            PHOENIX_INDEX,
        ],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
        [ci(S, Rank.THREE)],
    )
    game.mahjong_wish = Rank.FOUR.value

    for getter in (GameState.get_legal_actions, GameState.get_policy_legal_actions):
        actions = getter(game, 0)
        assert actions
        for action in actions:
            cards = action[0]
            assert any(
                card < 52 and card % 13 + 2 == Rank.FOUR.value
                for card in cards
            )
            if PHOENIX_INDEX in cards:
                trial = copy.deepcopy(game)
                assert trial.play(0, action), action


def test_phoenix_cannot_represent_mahjong_rank():
    cards = [
        ci(J, Rank.TWO),
        ci(P, Rank.THREE),
        ci(S, Rank.FOUR),
        ci(W, Rank.FIVE),
        PHOENIX_INDEX,
    ]

    assert classify_cards(cards, phoenix_as=1) is None
    assert [(combo.rank, combo.phoenix_as) for combo in phoenix_interpretations(cards)] == [
        (6.0, 6)
    ]
    assert classify_cards([MAHJONG_INDEX, PHOENIX_INDEX]) is None


def test_duplicate_card_action_is_rejected_without_state_changes():
    five = ci(J, Rank.FIVE)
    game = game_with_hands(
        [five, ci(J, Rank.ACE)],
        [ci(P, Rank.TWO)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )
    before = state_fingerprint(game)

    assert classify_cards([five, five]) is None
    assert not game.play(0, ([five, five], True))
    assert state_fingerprint(game) == before


def test_invalid_tichu_play_is_rejected_without_state_changes():
    game = game_with_hands(
        [ci(J, Rank.TWO), ci(J, Rank.ACE)],
        [ci(P, Rank.TWO)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )
    before = state_fingerprint(game)

    assert not game.play(0, ([ci(J, Rank.TWO), ci(J, Rank.ACE)], True))
    assert state_fingerprint(game) == before


def test_leading_pass_with_tichu_is_atomic():
    game = game_with_hands(
        [ci(J, Rank.TWO)],
        [ci(P, Rank.TWO)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )
    before = state_fingerprint(game)

    assert not game.play(0, ([], True))
    assert state_fingerprint(game) == before


def test_following_dog_with_tichu_is_atomic():
    game = game_with_hands(
        [ci(J, Rank.THREE), ci(J, Rank.ACE)],
        [DOG_INDEX, ci(P, Rank.FOUR)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )
    assert game.play(0, [ci(J, Rank.THREE)])
    before = state_fingerprint(game)

    assert not game.play(1, ([DOG_INDEX], True))
    assert state_fingerprint(game) == before


def test_wish_forces_manual_pass_before_tichu_declaration():
    game = game_with_hands(
        [ci(J, Rank.THREE), ci(J, Rank.ACE)],
        [ci(P, Rank.FIVE), ci(P, Rank.KING)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )
    assert game.play(0, [ci(J, Rank.THREE)])
    game.mahjong_wish = Rank.FIVE.value
    before = state_fingerprint(game)

    assert not game.play(1, ([], True))
    assert state_fingerprint(game) == before


def test_wish_forces_manual_dog_before_tichu_declaration():
    game = game_with_hands(
        [DOG_INDEX, ci(J, Rank.FIVE), ci(J, Rank.ACE)],
        [ci(P, Rank.TWO)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
    )
    game.mahjong_wish = Rank.FIVE.value
    before = state_fingerprint(game)

    assert not game.play(0, ([DOG_INDEX], True))
    assert state_fingerprint(game) == before


def test_card_actions_are_rejected_outside_play_phase():
    for reset_kwargs, expected_phase in (
        ({"with_calling": True}, GamePhase.GRAND_TICHU),
        ({"with_passing": True}, GamePhase.PASSING),
    ):
        game = GameState()
        game.reset(seed=42, **reset_kwargs)
        player = game.current_player
        card = next(iter(game.hands[player]))
        before = state_fingerprint(game)

        assert game.phase == expected_phase
        assert game.get_legal_actions(player) == []
        assert game.get_policy_legal_actions(player) == []
        assert not game.play(player, [card])
        assert state_fingerprint(game) == before


def test_dragon_gift_advances_history_trick_number():
    game = game_with_hands(
        [ci(J, Rank.KING), ci(J, Rank.ACE)],
        [DRAGON_INDEX, ci(P, Rank.TWO)],
        [ci(S, Rank.TWO), ci(S, Rank.THREE)],
        [ci(W, Rank.TWO), ci(W, Rank.THREE)],
        track_history=True,
    )

    assert game.play(0, [ci(J, Rank.KING)])
    assert game.play(1, [DRAGON_INDEX])
    assert game.play(2, [])
    assert game.play(3, [])
    assert game.play(0, [])
    assert game.dragon_give_pending == 1
    assert game.trick_num == 0

    assert game.give_dragon_trick(1, 0)
    assert game.trick_num == 1
    assert game.play(1, [ci(P, Rank.TWO)])
    assert game.play_log[-1][2] == 1


def test_invalid_dragon_recipients_are_rejected_atomically():
    game = game_with_hands(
        [ci(J, Rank.TWO)],
        [ci(P, Rank.TWO)],
        [ci(S, Rank.TWO)],
        [ci(W, Rank.TWO)],
        current_player=1,
    )
    game.dragon_give_pending = 1
    game.trick = TrickState(
        leading_combo=classify_cards([DRAGON_INDEX]),
        leading_player=1,
        cards_on_table=[DRAGON_INDEX],
        passes=3,
    )
    before = state_fingerprint(game)

    for recipient in (-1, 4, False, True, 0.0, 1.0):
        assert not game.give_dragon_trick(1, recipient)
        assert state_fingerprint(game) == before

    assert game.give_dragon_trick(1, np.int64(0))
    assert game.won_cards[0] == [DRAGON_INDEX]


def test_pettingzoo_mahjong_play_opens_wish_decision():
    env = TichuEnv(seed=0)
    env.reset(seed=0)
    env.game = game_with_hands(
        [MAHJONG_INDEX, ci(J, Rank.ACE)],
        [ci(P, Rank.TWO), ci(P, Rank.THREE)],
        [ci(S, Rank.TWO), ci(S, Rank.THREE)],
        [ci(W, Rank.TWO), ci(W, Rank.THREE)],
    )
    env._sync_agent_to_game()
    env._update_legal_actions()
    action_index = next(
        index
        for index, action in enumerate(env._legal_actions)
        if action == ([MAHJONG_INDEX], False)
    )

    env.step(action_index)

    assert env._pending_wish
    assert env.agent_selection == "player_0"
    assert env._legal_actions == [[rank] for rank in range(14)]
    assert env.observe("player_0")["action_mask"].sum() == 14
    assert env.observe("player_1")["action_mask"].sum() == 0

    env.step(5)  # The Mahjong player wishes for rank 6.

    assert not env._pending_wish
    assert env.game.mahjong_wish == 6
    assert env.agent_selection == "player_1"
    assert env.observe("player_0")["action_mask"].sum() == 0
    assert env.observe("player_1")["action_mask"].sum() > 0


def test_history_observation_space_accepts_trick_indices_above_one():
    env = TichuEnv(seed=0, track_history=True)
    env.reset(seed=0)
    player = env.game.current_player
    env.game._record_play(player, [], 41, False)
    agent = f"player_{player}"

    observation = env.observe(agent)

    assert observation["history"][0, 61] == np.float32(41 / 25)
    assert observation["history"][0, 61] > 1.0
    assert env.observation_space(agent).contains(observation)


def test_bomb_beating_dragon_keeps_the_completed_trick():
    bomb = [ci(suit, Rank.FIVE) for suit in (J, P, S, W)]
    game = game_with_hands(
        [DRAGON_INDEX, ci(J, Rank.TWO)],
        bomb + [ci(J, Rank.THREE)],
        [ci(P, Rank.TWO), ci(P, Rank.THREE)],
        [ci(S, Rank.TWO), ci(S, Rank.THREE)],
        track_history=True,
    )
    assert game.play(0, [DRAGON_INDEX])
    assert game.play(1, bomb)
    for player in (2, 3, 0):
        assert game.play(player, [])

    assert game.dragon_give_pending is None
    assert game.current_player == 1
    assert set(game.won_cards[1]) == {DRAGON_INDEX, *bomb}
    assert game.trick_num == 1
    assert game.trick_history[-1][0] == 1


def test_round_ending_bomb_keeps_dragon_points_for_bomb_winners_team():
    bomb = [ci(suit, Rank.FIVE) for suit in (J, P, S, W)]
    game = game_with_hands(
        [], [], [DRAGON_INDEX, ci(J, Rank.TWO)], bomb, current_player=2,
    )
    game.finish_order = [0, 1]
    game.finished = {0, 1}
    assert game.play(2, [DRAGON_INDEX])
    assert game.play(3, bomb)

    assert game.round_over
    assert game.finish_order == [0, 1, 3, 2]
    assert set(game.won_cards[3]) == {DRAGON_INDEX, *bomb}
    assert game.round_scores == (0, 45)


def test_round_ending_dragon_still_gives_points_to_opposing_team():
    game = game_with_hands(
        [], [], [DRAGON_INDEX], [ci(J, Rank.TWO)], current_player=2,
    )
    game.finish_order = [0, 1]
    game.finished = {0, 1}
    assert game.play(2, [DRAGON_INDEX])

    assert game.round_over
    assert DRAGON_INDEX in game.won_cards[1]
    assert game.round_scores == (0, 25)


def test_final_dog_is_accounted_for_in_normal_round_scoring():
    game = game_with_hands(
        [],
        [],
        [DOG_INDEX],
        [ci(W, Rank.TWO)],
        current_player=2,
    )
    game.finish_order = [1, 0]
    game.finished = {0, 1}

    assert game.play(2, [DOG_INDEX])

    assert game.round_over
    assert game.finish_order == [1, 0, 2, 3]
    assert game.round_scores == (0, 0)
    assert game.current_player == 2  # No lead is transferred after round end.
    assert DOG_INDEX in game.won_cards[0]
    assert sum(pile.count(DOG_INDEX) for pile in game.won_cards) == 1


def test_final_dog_is_accounted_for_in_double_finish():
    game = game_with_hands(
        [],
        [ci(P, Rank.TWO)],
        [DOG_INDEX],
        [ci(W, Rank.THREE)],
        current_player=2,
    )
    game.finish_order = [0]
    game.finished = {0}

    assert game.play(2, [DOG_INDEX])

    assert game.round_over
    assert game.finish_order == [0, 2]
    assert game.round_scores == (200, 0)
    assert game.current_player == 2  # No lead is transferred after round end.
    assert DOG_INDEX in game.won_cards[0]
    assert sum(pile.count(DOG_INDEX) for pile in game.won_cards) == 1
