import io
import random

import pytest
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
from tichu_env.encoding import encode_pass_source_ids, encode_pass_target_ids
from tichu_env.game_state import GamePhase, GameState
from training import advisor
from training.web import (
    WEB_ROOT,
    LocalGame,
    TichuWebHandler,
    WebError,
    _card_payload,
)


def test_request_parser_rejects_negative_content_length_without_reading_body():
    handler = object.__new__(TichuWebHandler)
    handler.headers = {"Content-Length": "-1"}
    handler.rfile = io.BytesIO(b'{"unbounded": true}')

    with pytest.raises(WebError, match="Invalid request body length"):
        handler._read_json()

    assert handler.rfile.tell() == 0



def _leading_web_game(hand: list[int]) -> LocalGame:
    """Build a minimal human lead without running a complete deal."""

    occupied = set(hand)
    assert DOG_INDEX not in occupied
    reserve = [card for card in range(56) if card not in occupied and card != DOG_INDEX]
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 0
    gs.hands = [set(hand + [DOG_INDEX]), {reserve[0]}, {reserve[1]}, {reserve[2]}]
    gs._invalidate_cache()
    game = LocalGame(mode="play", seat=0, device=torch.device("cpu"), gs=gs)
    game._start_timeline("Permutation test.")
    return game


@pytest.mark.parametrize(
    ("cards", "orders"),
    [
        (
            [card_index(Suit.JADE, Rank.TWO), card_index(Suit.PAGODA, Rank.TWO)],
            ((1, 0), (0, 1)),
        ),
        (
            [
                card_index(Suit.JADE, Rank.TWO),
                card_index(Suit.PAGODA, Rank.TWO),
                card_index(Suit.STAR, Rank.THREE),
                card_index(Suit.SWORD, Rank.THREE),
            ],
            ((3, 2, 1, 0), (1, 3, 0, 2), (2, 0, 3, 1)),
        ),
        (
            [card_index(Suit.JADE, Rank.FOUR), PHOENIX_INDEX],
            ((1, 0), (0, 1)),
        ),
        (
            [
                MAHJONG_INDEX,
                card_index(Suit.JADE, Rank.TWO),
                card_index(Suit.PAGODA, Rank.THREE),
                card_index(Suit.STAR, Rank.FOUR),
                card_index(Suit.SWORD, Rank.FIVE),
            ],
            ((4, 3, 2, 1, 0), (2, 0, 4, 1, 3), (1, 3, 0, 4, 2)),
        ),
    ],
)
def test_visual_play_accepts_card_combinations_in_any_selection_order(cards, orders):
    for order in orders:
        game = _leading_web_game(cards)
        selected = [cards[index] for index in order]

        game.play_cards(selected)

        assert game.gs is not None
        assert set(cards).isdisjoint(game.gs.hands[0])
        assert set(game.gs.trick.cards_on_table) == set(cards)
        assert {card["index"] for card in game.timeline[-1]["cards"]} == set(cards)


@pytest.mark.parametrize("call_tichu", [False, True])
def test_visual_play_permutation_matching_preserves_tichu_variant(call_tichu):
    cards = [
        card_index(Suit.JADE, Rank.KING),
        card_index(Suit.PAGODA, Rank.KING),
    ]
    game = _leading_web_game(cards)

    game.play_cards(list(reversed(cards)), call_tichu=call_tichu)

    assert game.gs is not None
    assert game.gs.tichu_declarations[0] == int(call_tichu)


def test_visual_play_prompts_for_phoenix_straight_then_applies_exact_choice():
    cards = [
        card_index(Suit.JADE, Rank.THREE),
        card_index(Suit.PAGODA, Rank.FOUR),
        card_index(Suit.STAR, Rank.FIVE),
        card_index(Suit.SWORD, Rank.SIX),
        PHOENIX_INDEX,
    ]
    game = _leading_web_game(cards)
    assert game.gs is not None
    game.gs.mahjong_wish = 4

    game.play_cards(list(reversed(cards)))

    state = game.state()
    assert [choice["value"] for choice in state["phoenix_choices"]] == [2, 7]
    assert state["human_turn"] is False
    assert set(cards).issubset(game.gs.hands[0])

    game.choose_phoenix(2)

    combo = game.gs.trick.leading_combo
    assert combo is not None
    assert (combo.rank, combo.phoenix_as) == (6.0, 2)
    assert game.gs.mahjong_wish is None
    assert "Phoenix as Two" in game.timeline[-2]["text"]


def test_visual_play_prompts_for_phoenix_full_house_triple_choice():
    cards = [
        card_index(Suit.JADE, Rank.FIVE),
        card_index(Suit.PAGODA, Rank.FIVE),
        card_index(Suit.STAR, Rank.NINE),
        card_index(Suit.SWORD, Rank.NINE),
        PHOENIX_INDEX,
    ]
    game = _leading_web_game(cards)

    game.play_cards([cards[index] for index in (2, 4, 0, 3, 1)])

    assert [
        choice["value"] for choice in game.state()["phoenix_choices"]
    ] == [5, 9]
    game.choose_phoenix(5)

    assert game.gs is not None
    combo = game.gs.trick.leading_combo
    assert combo is not None
    assert (combo.rank, combo.phoenix_as) == (5.0, 5)
    assert "Phoenix as Five" in game.timeline[-1]["text"]


def test_visual_play_filters_phoenix_choices_against_current_trick():
    lead = [
        card_index(Suit.JADE, Rank.TWO),
        card_index(Suit.PAGODA, Rank.THREE),
        card_index(Suit.STAR, Rank.FOUR),
        card_index(Suit.SWORD, Rank.FIVE),
        card_index(Suit.JADE, Rank.SIX),
    ]
    response = [
        card_index(Suit.STAR, Rank.THREE),
        card_index(Suit.SWORD, Rank.FOUR),
        card_index(Suit.JADE, Rank.FIVE),
        card_index(Suit.PAGODA, Rank.SIX),
        PHOENIX_INDEX,
    ]
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 3
    occupied = set(lead + response + [DOG_INDEX])
    reserve = [card for card in range(56) if card not in occupied]
    gs.hands = [
        set(response + [DOG_INDEX]),
        {reserve[0]},
        {reserve[1]},
        set(lead + [reserve[2]]),
    ]
    gs._invalidate_cache()
    assert gs.play(3, lead)
    game = LocalGame(mode="play", seat=0, device=torch.device("cpu"), gs=gs)
    game._start_timeline("Phoenix response test.")

    game.play_cards(list(reversed(response)))

    assert game.state()["phoenix_choices"] == []
    combo = gs.trick.leading_combo
    assert combo is not None
    assert (combo.rank, combo.phoenix_as) == (7.0, 7)


def test_visual_phoenix_prompt_can_be_cancelled_without_playing():
    cards = [
        card_index(Suit.JADE, rank)
        for rank in (Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX)
    ] + [PHOENIX_INDEX]
    game = _leading_web_game(cards)

    game.play_cards(cards)
    game.cancel_phoenix()

    assert game.pending_phoenix_play is None
    assert game.state()["human_turn"] is True
    assert game.gs is not None
    assert set(cards).issubset(game.gs.hands[0])


def test_visual_ui_exposes_explicit_phoenix_choice_endpoint():
    javascript = (WEB_ROOT / "app.js").read_text()

    assert 'request("/api/phoenix", { phoenix_as: choice.value })' in javascript
    assert 'request("/api/phoenix/cancel", {})' in javascript


def test_visual_state_api_marks_still_held_pass_cards_by_relative_source():
    seat = 3
    incoming = [
        card_index(Suit.JADE, Rank.TWO),
        card_index(Suit.PAGODA, Rank.THREE),
        card_index(Suit.STAR, Rank.FOUR),
    ]
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 0
    gs.hands = [
        {card_index(Suit.JADE, Rank.FIVE)},
        {card_index(Suit.PAGODA, Rank.SIX)},
        {card_index(Suit.STAR, Rank.SEVEN)},
        set(incoming + [DOG_INDEX]),
    ]
    gs._incoming_pass_source[seat] = {0: incoming[0], 1: incoming[1], 2: incoming[2]}
    gs._invalidate_cache()
    game = LocalGame(mode="play", seat=seat, device=torch.device("cpu"), gs=gs)

    marked = {
        card["index"]: card["received_from"]
        for card in game.state()["hand"]
        if "received_from" in card
    }

    assert marked == {
        incoming[0]: {
            "player": 0,
            "relation": "left",
            "label": "←",
            "description": "Received from left opponent (P0)",
        },
        incoming[1]: {
            "player": 1,
            "relation": "partner",
            "label": "↑",
            "description": "Received from partner/across (P1)",
        },
        incoming[2]: {
            "player": 2,
            "relation": "right",
            "label": "→",
            "description": "Received from right opponent (P2)",
        },
    }

    gs.hands[seat].remove(incoming[0])
    assert incoming[0] not in {
        card["index"] for card in game.state()["hand"]
    }


def test_live_undo_restores_own_play_and_removes_revealed_ai_replies():
    cards = [
        card_index(Suit.JADE, Rank.TWO),
        card_index(Suit.PAGODA, Rank.TWO),
    ]
    game = _leading_web_game(cards)
    assert game.gs is not None
    game.gs._incoming_pass_source[0][1] = cards[0]
    hand_before = set(game.gs.hands[0])
    timeline_before = list(game.timeline)
    rng_before = game.rng.getstate()
    global_rng_before = random.getstate()
    torch_rng_before = torch.get_rng_state().clone()

    game.play_cards(list(reversed(cards)))
    random.random()
    torch.rand(3)
    pending = game.pending_ai_play
    assert pending is not None
    assert game.advance_ai_step(pending.id)
    assert game.state()["can_undo"] is True

    game.undo_live()

    assert game.gs.current_player == 0
    assert game.gs.hands[0] == hand_before
    assert game.gs.trick.cards_on_table == []
    assert game.timeline == timeline_before
    assert game.pending_ai_play is None
    assert game.rng.getstate() == rng_before
    assert random.getstate() == global_rng_before
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert game.state()["can_undo"] is False
    restored_card = next(
        card for card in game.state()["hand"] if card["index"] == cards[0]
    )
    assert restored_card["received_from"]["relation"] == "left"


def test_live_undo_restores_gt_then_passing_before_immediate_ai_work():
    game = LocalGame(
        mode="play",
        seat=0,
        device=torch.device("cpu"),
        rng=random.Random(23),
    )
    game.new_round(seed=31)
    assert game.gs is not None
    gt_timeline = list(game.timeline)

    game.declare_grand_tichu(False)
    assert game.gs.phase == GamePhase.PASSING
    assert set(game.recorded_gt_decisions) == {0, 1, 2, 3}
    game.undo_live()

    assert game.gs.phase == GamePhase.GRAND_TICHU
    assert game.gs.current_player == 0
    assert game.gs.tichu_declarations == [0, 0, 0, 0]
    assert game.recorded_gt_decisions == {}
    assert game.timeline == gt_timeline

    game.declare_grand_tichu(False)
    passing_hand = set(game.gs.hands[0])
    passing_timeline = list(game.timeline)
    cards = game.gs.get_passing_hand(0)[:3]
    game.submit_pass(cards)
    assert game.gs.phase == GamePhase.PLAY
    assert game.pending_ai_play is not None

    game.undo_live()

    assert game.gs.phase == GamePhase.PASSING
    assert game.gs.current_player == 0
    assert game.gs.hands[0] == passing_hand
    assert game.timeline == passing_timeline
    assert game.pending_ai_play is None


def test_live_undo_treats_mahjong_play_and_wish_as_one_transaction():
    reserve = [card for card in range(56) if card not in {MAHJONG_INDEX, DOG_INDEX}]
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 0
    gs.hands = [
        {MAHJONG_INDEX, DOG_INDEX},
        {reserve[0], reserve[1]},
        {reserve[2], reserve[3]},
        {reserve[4], reserve[5]},
    ]
    gs._invalidate_cache()
    game = LocalGame(mode="play", seat=0, device=torch.device("cpu"), gs=gs)
    game._start_timeline("Mahjong undo test.")
    timeline_before = list(game.timeline)

    game.play_cards([MAHJONG_INDEX])
    assert game.waiting_for_wish
    game.set_wish(8)
    assert game.gs.mahjong_wish == 8

    game.undo_live()

    assert MAHJONG_INDEX in game.gs.hands[0]
    assert game.gs.current_player == 0
    assert game.gs.mahjong_wish is None
    assert game.waiting_for_wish is False
    assert game.timeline == timeline_before
    assert game.state()["can_undo"] is False


def test_round_ending_human_mahjong_does_not_prompt_for_wish():
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 0
    gs.finish_order = [2]
    gs.finished = {2}
    gs.hands = [
        {MAHJONG_INDEX},
        {card_index(Suit.JADE, Rank.TWO)},
        set(),
        {card_index(Suit.PAGODA, Rank.TWO)},
    ]
    gs._invalidate_cache()
    game = LocalGame(mode="play", seat=0, device=torch.device("cpu"), gs=gs)
    game._start_timeline("Terminal human Mahjong test.")

    game.play_cards([MAHJONG_INDEX])

    assert gs.round_over is True
    assert game.waiting_for_wish is False
    assert game.wish_player is None
    assert game.state()["waiting_for_wish"] is False
    assert game.timeline[-1]["type"] == "round_end"



def test_phoenix_prompt_only_adds_undo_when_interpretation_is_committed():
    cards = [
        card_index(Suit.JADE, rank)
        for rank in (Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX)
    ] + [PHOENIX_INDEX]
    game = _leading_web_game(cards)

    game.play_cards(cards)
    assert game.pending_phoenix_play is not None
    assert game.state()["can_undo"] is False
    game.cancel_phoenix()
    assert game.state()["can_undo"] is False

    game.play_cards(list(reversed(cards)))
    game.choose_phoenix(2)
    assert game.state()["can_undo"] is True
    game.undo_live()

    assert game.pending_phoenix_play is None
    assert game.gs is not None
    assert set(cards).issubset(game.gs.hands[0])
    assert game.gs.current_player == 0


def test_advisor_rejects_alternative_play_while_phoenix_choice_is_pending():
    phoenix_cards = [
        card_index(Suit.JADE, Rank.THREE),
        card_index(Suit.PAGODA, Rank.FOUR),
        card_index(Suit.STAR, Rank.FIVE),
        card_index(Suit.SWORD, Rank.SIX),
        PHOENIX_INDEX,
    ]
    alternative = card_index(Suit.JADE, Rank.TWO)
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 1
    gs.hands = [
        {DRAGON_INDEX},
        set(phoenix_cards + [alternative]),
        {card_index(Suit.STAR, Rank.TWO)},
        {card_index(Suit.PAGODA, Rank.TWO)},
    ]
    gs._invalidate_cache()
    game = LocalGame(mode="advisor", seat=0, device=torch.device("cpu"), gs=gs)
    game._start_timeline("Pending Phoenix test.")

    game.advisor_opponent_play(1, "3g 4b 5r 6s p")
    pending = game.pending_phoenix_play
    pending_undo = game.pending_phoenix_undo
    hands_before = [set(hand) for hand in gs.hands]
    timeline_before = list(game.timeline)

    with pytest.raises(WebError, match="pending Phoenix interpretation"):
        game.advisor_opponent_play(1, "2g")

    assert game.pending_phoenix_play is pending
    assert game.pending_phoenix_undo is pending_undo
    assert gs.current_player == 1
    assert gs.hands == hands_before
    assert game.timeline == timeline_before

    game.cancel_phoenix()
    game.advisor_opponent_play(1, "2g")

    assert alternative not in gs.hands[1]
    assert gs.current_player == 2



def test_live_undo_restores_dragon_recipient_decision():
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 3
    gs.hands = [
        {DRAGON_INDEX, card_index(Suit.JADE, Rank.TWO)},
        {card_index(Suit.PAGODA, Rank.TWO)},
        {card_index(Suit.STAR, Rank.TWO)},
        {card_index(Suit.SWORD, Rank.KING), card_index(Suit.SWORD, Rank.TWO)},
    ]
    gs._invalidate_cache()
    assert gs.play(3, [card_index(Suit.SWORD, Rank.KING)])
    assert gs.play(0, [DRAGON_INDEX])
    assert gs.play(1, [])
    assert gs.play(2, [])
    assert gs.play(3, [])
    assert gs.dragon_give_pending == 0
    game = LocalGame(mode="play", seat=0, device=torch.device("cpu"), gs=gs)
    game._start_timeline("Dragon undo test.")
    table_before = list(gs.trick.cards_on_table)

    game.give_dragon(1)
    assert DRAGON_INDEX in gs.won_cards[1]
    game.undo_live()

    assert game.gs.dragon_give_pending == 0
    assert game.gs.trick.cards_on_table == table_before
    assert DRAGON_INDEX not in game.gs.won_cards[1]


def test_live_undo_restores_advisor_opponent_play():
    game = LocalGame(mode="advisor", seat=0, device=torch.device("cpu"))
    game.setup_advisor("2-a g d")
    assert game.gs is not None
    player = game.gs.current_player
    assert player != 0
    hand_before = set(game.gs.hands[player])
    timeline_before = list(game.timeline)

    game.advisor_opponent_play(player, "m")
    assert game.waiting_for_wish
    game.set_wish(6)
    game.undo_live()

    assert game.gs.current_player == player
    assert game.gs.hands[player] == hand_before
    assert game.timeline == timeline_before
    assert game.waiting_for_wish is False


def test_round_ending_advisor_mahjong_does_not_prompt_for_wish():
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 1
    gs.finish_order = [3]
    gs.finished = {3}
    gs.hands = [
        {DRAGON_INDEX},
        {MAHJONG_INDEX},
        {card_index(Suit.STAR, Rank.TWO)},
        set(),
    ]
    gs._invalidate_cache()
    game = LocalGame(mode="advisor", seat=0, device=torch.device("cpu"), gs=gs)
    game._start_timeline("Terminal advisor Mahjong test.")

    game.advisor_opponent_play(1, "m")

    assert gs.round_over is True
    assert game.waiting_for_wish is False
    assert game.wish_player is None
    assert game.state()["waiting_for_wish"] is False
    assert game.timeline[-1]["type"] == "round_end"



def test_advisor_rejects_next_opponent_play_until_wish_is_recorded():
    next_card = card_index(Suit.JADE, Rank.TWO)
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 1
    gs.hands = [
        {DRAGON_INDEX},
        {MAHJONG_INDEX, card_index(Suit.STAR, Rank.TWO)},
        {next_card, card_index(Suit.JADE, Rank.THREE)},
        {card_index(Suit.PAGODA, Rank.TWO)},
    ]
    gs._invalidate_cache()
    game = LocalGame(mode="advisor", seat=0, device=torch.device("cpu"), gs=gs)
    game._start_timeline("Pending wish test.")

    game.advisor_opponent_play(1, "m")
    hands_before = [set(hand) for hand in gs.hands]
    timeline_before = list(game.timeline)

    with pytest.raises(WebError, match="pending Mahjong wish"):
        game.advisor_opponent_play(2, "2g")

    assert game.waiting_for_wish is True
    assert gs.current_player == 2
    assert gs.hands == hands_before
    assert game.timeline == timeline_before

    game.set_wish(None)
    game.advisor_opponent_play(2, "2g")

    assert next_card not in gs.hands[2]
    assert gs.current_player == 3



def test_live_undo_walks_back_advisor_staged_setup():
    game = LocalGame(mode="advisor", seat=0, device=torch.device("cpu"))

    game.advisor_set_first_eight("2-9 g")
    assert game.state()["can_undo"] is True
    game.undo_live()
    assert game.advisor_stage == "first_eight"
    assert game.advisor_first_eight == ()

    game.advisor_set_first_eight("2-9 g")
    game.advisor_set_remaining_six("10-a g p", [0, 2])
    game.undo_live()
    assert game.advisor_stage == "remaining_six"
    assert game.gs is None

    game.advisor_set_remaining_six("10-a g p", [0, 2])
    assert game.gs is not None
    outgoing = sorted(game.gs.hands[0])[:3]
    game.submit_pass(outgoing)
    game.undo_live()
    assert game.advisor_stage == "passing"
    assert game.advisor_outgoing is None

    game.submit_pass(outgoing)
    hands_before_received = [set(hand) for hand in game.gs.hands]
    game.advisor_receive_pass("2-4 b")
    game.undo_live()
    assert game.advisor_stage == "received"
    assert game.advisor_outgoing == tuple(outgoing)
    assert game.gs.hands == hands_before_received

    game.advisor_receive_pass("2-4 b")
    assert game.advisor_stage == "mahjong_holder"
    game.advisor_set_mahjong_holder(1)
    game.undo_live()
    assert game.advisor_stage == "mahjong_holder"
    assert game.gs.current_player == -1


def test_live_undo_is_cleared_and_disabled_when_ai_replay_starts():
    game = LocalGame(
        mode="play",
        seat=0,
        device=torch.device("cpu"),
        rng=random.Random(41),
    )
    game.new_round(seed=43)
    game.declare_grand_tichu(False)
    assert game.state()["can_undo"] is True
    assert game.gs is not None
    game.gs.round_over = True

    game.start_ai_replay()

    assert game.spectating is True
    assert game.undo_stack == []
    assert game.state()["can_undo"] is False


def test_visual_ui_undo_stops_ai_schedule_before_requesting_restore():
    javascript = (WEB_ROOT / "app.js").read_text()
    undo_block = javascript[javascript.index('const undo = makeButton("Undo"'):]
    undo_block = undo_block[:undo_block.index("undo.disabled")]

    assert undo_block.index("stopAiSchedule()") < undo_block.index('request("/api/undo"')


def test_advisor_opponent_entry_prompts_for_phoenix_value_independent_of_order():
    cards = [
        card_index(Suit.JADE, Rank.THREE),
        card_index(Suit.PAGODA, Rank.FOUR),
        card_index(Suit.STAR, Rank.FIVE),
        card_index(Suit.SWORD, Rank.SIX),
        PHOENIX_INDEX,
    ]
    occupied = set(cards + [DOG_INDEX])
    reserve = [card for card in range(56) if card not in occupied]
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 1
    gs.hands = [{DOG_INDEX}, set(cards + [reserve[0]]), {reserve[1]}, {reserve[2]}]
    gs._invalidate_cache()
    game = LocalGame(mode="advisor", seat=0, device=torch.device("cpu"), gs=gs)
    game._start_timeline("Advisor Phoenix entry test.")

    game.advisor_opponent_play(1, "p 6s 3g 5r 4b")

    assert [choice["value"] for choice in game.state()["phoenix_choices"]] == [2, 7]
    assert set(cards).issubset(gs.hands[1])
    game.choose_phoenix(2)

    combo = gs.trick.leading_combo
    assert combo is not None
    assert (combo.rank, combo.phoenix_as) == (6.0, 2)
    assert "Phoenix as Two" in game.timeline[-1]["text"]


def test_card_payload_uses_original_tichu_deck_coordinates_for_all_cards():
    cards = [_card_payload(card) for card in range(56)]

    assert {card["index"] for card in cards} == set(range(56))
    assert all(0 <= card["sprite_row"] <= 4 for card in cards)
    assert all(0 <= card["sprite_col"] <= 12 for card in cards)
    assert cards[0]["sprite_row"] == 0
    assert cards[0]["sprite_col"] == 0
    assert cards[12]["sprite_row"] == 0
    assert cards[12]["sprite_col"] == 12
    assert cards[13]["sprite_row"] == 1
    assert cards[52]["special"]
    assert [cards[index]["sprite_col"] for index in range(52, 56)] == [0, 1, 2, 3]
    assert cards[55]["name"] == "Dragon"


def test_visual_ui_uses_project_owned_tichu_deck_assets():
    css = (WEB_ROOT / "app.css").read_text()

    assert (WEB_ROOT / "assets" / "tichu-original-deck.svg").is_file()
    assert (WEB_ROOT / "assets" / "card-back.svg").is_file()
    assert "tichu-original-deck.svg" in css
    assert "english-pattern-deck-cc0.svg" not in css


def test_visual_deck_uses_blue_pagodas_and_black_swords():
    deck = (WEB_ROOT / "assets" / "tichu-original-deck.svg").read_text()

    assert '<g id="pagoda">' in deck
    assert 'fill="#5b7fbd" stroke="#263f70"' in deck
    assert '<g id="sword">' in deck
    assert 'fill="#3d444b" stroke="#14191e"' in deck
    assert deck.count("#31558f") == 13
    assert deck.count("#252b30") == 13


def test_visual_mode_menu_switches_cleanly_between_play_and_advisor():
    game = LocalGame(
        mode="play",
        seat=0,
        show_mode_menu=True,
        device=torch.device("cpu"),
    )

    assert game.state()["show_mode_menu"] is True
    game.choose_mode("advisor")
    advisor_state = game.state()
    assert advisor_state["mode"] == "advisor"
    assert advisor_state["show_mode_menu"] is False
    assert advisor_state["advisor_stage"] == "first_eight"

    game.open_mode_menu()
    game.choose_mode("play")
    play_state = game.state()
    assert play_state["mode"] == "play"
    assert play_state["show_mode_menu"] is False
    assert play_state["ready"] is False
    assert play_state["advisor_stage"] is None


def test_visual_ui_only_elevates_cards_in_the_human_hand():
    css = (WEB_ROOT / "app.css").read_text()
    javascript = (WEB_ROOT / "app.js").read_text()

    assert ".hand-cards .card:not(:disabled):hover, .hand-cards .card.selected" in css
    assert ".table-cards .card.selected" in css
    assert "z-index: auto;" in css
    assert ".pass-source-marker" in css
    assert "position: absolute;" in css
    assert "receivedFrom.description" in javascript
    assert 'marker.setAttribute("aria-hidden", "true")' in javascript


def test_visual_play_controller_resolves_ai_grand_tichu_and_passing_immediately():
    game = LocalGame(
        mode="play",
        seat=0,
        device=torch.device("cpu"),
        rng=random.Random(7),
    )
    game.new_round(seed=19)

    assert game.gs is not None
    assert game.gs.phase == GamePhase.GRAND_TICHU
    assert game.gs.current_player == 0

    game.declare_grand_tichu(False)
    # GT and passing are administrative actions: they should not wait for the
    # card-reveal timer or model-thinking animation.
    assert game.gs.phase == GamePhase.PASSING
    assert game.gs.current_player == 0
    assert [event["type"] for event in game.state()["history"][-3:]] == [
        "declaration", "declaration", "declaration",
    ]

    hand = game.gs.get_passing_hand(0)
    game.submit_pass(hand[:3])
    assert game.gs.phase == GamePhase.PLAY
    assert len(game.gs.hands[0]) == 14


def test_visual_play_precomputes_then_reveals_exactly_one_ai_play():
    game = LocalGame(
        mode="play",
        seat=0,
        device=torch.device("cpu"),
        rng=random.Random(7),
    )
    game.new_round(seed=1)
    game.declare_grand_tichu(False)
    hand = game.gs.get_passing_hand(0)
    game.submit_pass(hand[:3])
    state = game.state()
    pass_event = next(
        event for event in reversed(state["history"])
        if event["type"] == "pass_submission" and event["player"] == game.seat
    )

    assert pass_event["type"] == "pass_submission"
    assert "P1 (left)" in pass_event["text"]
    assert {card["index"] for card in pass_event["cards"]} == set(hand[:3])
    # The AI selected its normal play during submit_pass(), but the real game
    # state and timeline stay untouched until the browser reveals that id.
    assert state["ai_pending"] is True
    assert state["ai_delay_ms"] == 500
    assert state["ai_action_id"] == game.pending_ai_play.id
    assert state["ai_player"] == game.gs.current_player
    pending = game.pending_ai_play
    assert pending is not None
    timeline_before = list(state["history"])
    hand_before = set(game.gs.hands[pending.player])
    assert set(pending.cards).issubset(hand_before)

    assert game.advance_ai_step(pending.id) is True
    after = game.state()
    assert len(after["history"]) > len(timeline_before)
    assert any(
        event["type"] in {"play", "pass"} and event["player"] == pending.player
        for event in after["history"][len(timeline_before):]
    )
    assert set(pending.cards).isdisjoint(game.gs.hands[pending.player])

    # An old browser timer must not reveal the next precomputed action.
    next_pending = game.pending_ai_play
    if next_pending is not None:
        assert game.advance_ai_step(pending.id) is False
        assert game.pending_ai_play == next_pending


def test_completed_deal_can_be_replayed_with_all_four_ai_seats():
    game = LocalGame(
        mode="play",
        seat=0,
        device=torch.device("cpu"),
        rng=random.Random(13),
    )
    game.new_round(seed=23)
    assert game.replay_start_state is not None
    original_cards = set().union(
        *(hand | set(remaining) for hand, remaining in zip(
            game.replay_start_state.hands,
            game.replay_start_state._remaining_cards or [],
        ))
    )
    assert len(original_cards) == 56

    # The replay action is deliberately available only after the played round
    # completes; the source deal remains frozen independently of live state.
    assert game.gs is not None
    game.gs.round_over = True
    game.start_ai_replay()

    state = game.state()
    assert state["spectating"] is True
    assert state["human_turn"] is False
    assert state["ai_pending"] is True
    assert state["history"][0]["text"].startswith("AI replay started")
    assert set().union(*game.gs.hands) == original_cards

    # In spectator mode P0 (the former human seat) is also model-controlled,
    # and every queued move is revealed one at a time through the same API.
    saw_seat_zero = False
    for _ in range(2_000):
        if game.gs.round_over:
            break
        pending = game.pending_ai_play
        assert pending is not None
        saw_seat_zero |= pending.player == game.seat
        assert game.advance_ai_step(pending.id) is True
    else:
        raise AssertionError("AI replay did not finish")

    final_state = game.state()
    assert saw_seat_zero
    assert final_state["round_over"] is True
    assert final_state["spectating"] is True
    assert final_state["can_replay"] is True
    assert any(event["type"] == "round_end" for event in final_state["history"])
    assert all("You" not in event["text"] for event in final_state["history"])


def test_ai_replay_reuses_opponent_gt_yes_no_and_passes(monkeypatch):
    original_opponent_gt = iter((True, False, True))
    monkeypatch.setattr(
        "training.web.ai_gt",
        lambda _policy, _hand, _device: next(original_opponent_gt),
    )
    game = LocalGame(
        mode="play",
        seat=0,
        device=torch.device("cpu"),
        rng=random.Random(37),
    )
    game.new_round(seed=43)
    assert game.gs is not None

    game.declare_grand_tichu(False)
    assert game.recorded_gt_decisions == {0: False, 1: True, 2: False, 3: True}
    game.submit_pass(game.gs.get_passing_hand(0)[:3])
    original_passes = dict(game.recorded_passes)
    assert set(original_passes) == {0, 1, 2, 3}

    replay_gt_calls = []

    def replay_gt(_policy, hand, _device):
        replay_gt_calls.append(tuple(hand))
        return True

    monkeypatch.setattr("training.web.ai_gt", replay_gt)
    game.gs.round_over = True
    game.start_ai_replay()

    # Only the compared seat asks the current model. Opponent calls and
    # explicit no-calls come from the played round's recorded decisions.
    assert len(replay_gt_calls) == 1
    assert game.gs.tichu_declarations == [2, 2, 0, 2]
    declaration_events = [
        event for event in game.timeline if event["type"] == "declaration"
    ]
    assert [event["text"] for event in declaration_events] == [
        "P0 called Grand Tichu.",
        "P1 called Grand Tichu (same as played round).",
        "P2 declined Grand Tichu (same as played round).",
        "P3 called Grand Tichu (same as played round).",
    ]

    for player in (1, 2, 3):
        targets = game.gs._outgoing_pass_target[player]
        recipients = ((player + 1) % 4, (player + 2) % 4, (player + 3) % 4)
        assert tuple(targets[recipient] for recipient in recipients) == original_passes[player]

    frame = game.replay_frames[game.replay_cursor]
    assert frame.recorded_gt_decisions == game.recorded_gt_decisions
    assert frame.recorded_passes == game.recorded_passes


def test_ai_replay_navigation_restores_plays_and_final_score():
    game = LocalGame(
        mode="play",
        seat=0,
        device=torch.device("cpu"),
        rng=random.Random(29),
    )
    game.new_round(seed=41)
    assert game.gs is not None
    game.gs.round_over = True
    game.start_ai_replay()

    initial = game.state()
    assert initial["replay_cursor"] == 0
    assert initial["replay_can_back"] is False
    assert initial["replay_can_forward"] is True

    assert game.step_ai_replay(1) is True
    first_play = game.state()
    assert first_play["replay_cursor"] == 1
    assert first_play["replay_can_back"] is True
    first_history = first_play["history"]
    first_hand = first_play["hand"]
    first_table = first_play["table"]

    assert game.step_ai_replay(1) is True
    assert game.step_ai_replay(-1) is True
    restored = game.state()
    assert restored["history"] == first_history
    assert restored["hand"] == first_hand
    assert restored["table"] == first_table

    assert game.step_ai_replay(1) is True
    assert game.finish_ai_replay() is True
    final = game.state()
    final_scores = final["scores"]
    assert final["round_over"] is True
    assert final["replay_can_forward"] is False
    assert any(event["type"] == "round_end" for event in final["history"])

    assert game.step_ai_replay(-1) is True
    assert game.state()["replay_can_forward"] is True
    assert game.step_ai_replay(1) is True
    restored_final = game.state()
    assert restored_final["round_over"] is True
    assert restored_final["scores"] == final_scores


def test_precomputed_ai_mahjong_applies_its_wish_when_the_play_is_revealed():
    class FixedWishPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wish_head = torch.nn.Identity()

        def act(self, obs, actions, n_legal, history, history_len, pass_src, pass_tgt):
            batch = obs.shape[0]
            return torch.zeros(batch, dtype=torch.long), None, None

        def fused_features(self, obs, history, history_len, pass_src, pass_tgt):
            return torch.zeros((obs.shape[0], 4))

        def get_mahjong_wish_logits(self, features):
            logits = torch.full((features.shape[0], 14), -1_000.0)
            logits[:, 7] = 1_000.0
            return logits

    # Give an AI one legal lead, so the precomputed action must be Mahjong.
    # This keeps the test independent of a sampled full deal.
    gs = GameState(track_history=True)
    gs.phase = GamePhase.PLAY
    gs.current_player = 1
    gs.hands = [{0}, {MAHJONG_INDEX}, {1}, {2}]
    game = LocalGame(
        mode="play",
        seat=0,
        policy=FixedWishPolicy(),
        kind="history",
        device=torch.device("cpu"),
        rng=random.Random(11),
        gs=gs,
    )
    game._start_timeline("Test round.")
    game._prepare_ai()

    pending = game.pending_ai_play
    assert pending is not None
    assert pending.cards == (MAHJONG_INDEX,)
    assert gs.mahjong_wish is None

    assert game.advance_ai_step(pending.id) is True
    assert gs.mahjong_wish == 8
    event_types = [event["type"] for event in game.timeline]
    assert event_types[-1] == "wish"
    assert "play" in event_types[:-1]


def test_visual_state_exposes_model_value_only_for_human_play_turn():
    class FixedValuePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Identity()
            self.value_head = torch.nn.Linear(196, 1)

        def forward(self, obs, mask):
            logits = torch.zeros_like(mask, dtype=torch.float32)
            return logits.masked_fill(~mask, float("-inf"))

    game = LocalGame(
        mode="play",
        seat=0,
        policy=FixedValuePolicy(),
        kind="fixed",
        device=torch.device("cpu"),
        rng=random.Random(3),
    )
    game.new_round(seed=5)

    # Grand-Tichu setup is outside main-play critic training, so it should not
    # be shown as a stable point estimate in the browser.
    assert game.state()["value_available"] is False
    assert game.state()["predicted_value"] is None

    assert game.gs is not None
    game.gs.phase = GamePhase.PLAY
    game.gs.current_player = game.seat
    state = game.state()

    assert state["value_available"] is True
    assert isinstance(state["predicted_value"], float)
    # No card reveal is pending while the human is deciding a play.
    assert state["ai_delay_ms"] == 0


def test_random_ai_has_no_model_value():
    game = LocalGame(mode="play", seat=0, device=torch.device("cpu"))
    game.new_round(seed=5)

    state = game.state()
    assert state["has_model_value"] is False
    assert state["value_available"] is False
    assert state["predicted_value"] is None


def test_advisor_controller_uses_existing_parser_and_setup():
    cards = "2g 3g 4g 5g 6g 7g 8g 9g 10g Jg Qg Kg Ag Mahjong"
    game = LocalGame(mode="advisor", seat=0, device=torch.device("cpu"))

    game.setup_advisor(cards)
    state = game.state()

    assert state["ready"] is True
    assert state["mode"] == "advisor"
    assert len(state["hand"]) == 14


def test_advisor_controller_records_an_opponent_tichu_call():
    # Deliberately omit the Mahjong so an opponent leads with it.
    cards = "2g 3g 4g 5g 6g 7g 8g 9g 10g Jg Qg Kg Ag 2b"
    game = LocalGame(mode="advisor", seat=0, device=torch.device("cpu"))
    game.setup_advisor(cards)

    assert game.gs is not None
    player = game.gs.current_player
    assert player != game.seat
    assert MAHJONG_INDEX in game.gs.hands[player]

    game.advisor_opponent_play(player, "Mahjong", call_tichu=True)

    assert game.gs.tichu_declarations[player] == 1


def test_advisor_staged_setup_exposes_gt_and_records_directed_provenance():
    class FixedGtPolicy:
        def evaluate(self, hand, declare):
            probability = torch.full(
                (hand.shape[0],),
                0.75,
                dtype=hand.dtype,
                device=hand.device,
            )
            return probability.log(), torch.zeros_like(probability)

    game = LocalGame(
        mode="advisor",
        seat=0,
        gt_policy=FixedGtPolicy(),
        device=torch.device("cpu"),
        rng=random.Random(17),
    )

    game.advisor_set_first_eight("2-9 g")
    state = game.state()
    assert state["ready"] is False
    assert state["advisor_stage"] == "remaining_six"
    assert state["gt_probability"] == 0.75
    assert len(state["first_eight"]) == 8

    game.advisor_set_remaining_six("10-a g p", [0, 2])
    assert game.gs is not None
    assert game.advisor_stage == "passing"
    assert game.gs.phase == GamePhase.PASSING
    assert game.gs.tichu_declarations == [2, 0, 2, 0]

    outgoing = [
        card_index(Suit.JADE, Rank.TWO),
        card_index(Suit.JADE, Rank.THREE),
        card_index(Suit.JADE, Rank.FOUR),
    ]
    incoming = [
        card_index(Suit.PAGODA, Rank.TWO),
        card_index(Suit.PAGODA, Rank.THREE),
        card_index(Suit.PAGODA, Rank.FOUR),
    ]
    game.submit_pass(outgoing)
    assert game.advisor_stage == "received"
    game.advisor_receive_pass("2-4 b")

    assert game.gs.phase == GamePhase.PLAY
    assert encode_pass_source_ids(game.gs, 0).tolist() == incoming
    assert encode_pass_target_ids(game.gs, 0).tolist() == outgoing
    marked = {
        card["index"]: card["received_from"]["relation"]
        for card in game.state()["hand"]
        if "received_from" in card
    }
    assert marked == {
        incoming[0]: "left",
        incoming[1]: "partner",
        incoming[2]: "right",
    }
    assert [len(hand) for hand in game.gs.hands] == [14, 14, 14, 14]
    assert game.advisor_stage == "mahjong_holder"

    game.advisor_set_mahjong_holder(1)
    assert game.advisor_stage == "play"
    assert game.gs.current_player == 1
    assert MAHJONG_INDEX in game.gs.hands[1]


def test_advisor_derives_mahjong_holder_from_own_directed_pass():
    game = LocalGame(
        mode="advisor",
        seat=0,
        device=torch.device("cpu"),
        rng=random.Random(29),
    )
    game.gs = advisor.setup_game(list(range(13)) + [MAHJONG_INDEX], 0)
    game.gs.phase = GamePhase.PASSING
    game.gs.current_player = 0
    game.advisor_stage = "passing"

    game.submit_pass([MAHJONG_INDEX, 0, 1])
    game.advisor_receive_pass("2r 2b 2s")

    assert game.advisor_stage == "play"
    assert game.gs.current_player == 1
    assert MAHJONG_INDEX in game.gs.hands[1]
    assert game.gs._outgoing_pass_target[0][1] == MAHJONG_INDEX
    assert "Mahjong you passed" in game.last_event


def test_advisor_records_opponent_dragon_gift():
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 1
    gs.hands = [
        {card_index(Suit.JADE, Rank.TWO)},
        {DRAGON_INDEX, card_index(Suit.PAGODA, Rank.TWO)},
        {card_index(Suit.STAR, Rank.TWO)},
        {card_index(Suit.SWORD, Rank.TWO)},
    ]
    gs._invalidate_cache()
    game = LocalGame(mode="advisor", seat=0, device=torch.device("cpu"), gs=gs)
    game._start_timeline("Opponent Dragon test.")
    game.advisor_opponent_play(1, "d")
    game.advisor_opponent_play(2, "")
    game.advisor_opponent_play(3, "")
    assert gs.play(0, [])
    assert gs.dragon_give_pending == 1

    with pytest.raises(WebError, match="must go to an opponent"):
        game.give_dragon(3)
    game.give_dragon(2)

    assert DRAGON_INDEX in gs.won_cards[2]
    assert gs.dragon_give_pending is None
    assert gs.current_player == 1
    assert game.timeline[-1]["text"] == "P1 gave the Dragon trick to P2."
    game.undo_live()
    assert game.gs.dragon_give_pending == 1


def test_play_mode_still_rejects_giving_an_ai_dragon_trick():
    gs = GameState()
    gs.phase = GamePhase.PLAY
    gs.current_player = 1
    gs.hands = [
        {card_index(Suit.JADE, Rank.TWO)},
        {DRAGON_INDEX, card_index(Suit.PAGODA, Rank.TWO)},
        {card_index(Suit.STAR, Rank.TWO)},
        {card_index(Suit.SWORD, Rank.TWO)},
    ]
    gs._invalidate_cache()
    for player, cards in ((1, [DRAGON_INDEX]), (2, []), (3, []), (0, [])):
        assert gs.play(player, cards)
    game = LocalGame(mode="play", seat=0, device=torch.device("cpu"), gs=gs)

    with pytest.raises(WebError, match="do not have a Dragon trick"):
        game.give_dragon(2)
