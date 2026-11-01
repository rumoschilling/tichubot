"""A small local browser UI for playing or advising a Tichu round.

The UI deliberately uses only the Python standard library.  It is a thin
controller over :class:`tichu_env.game_state.GameState`, plus the policy
helpers already used by ``training.play`` and ``training.advisor``.  That
keeps the terminal programs useful while giving them a visual, local option.

Run one of:

    python -m training.web --mode play --checkpoint path/to/checkpoint.pt
    python -m training.web --mode advisor --checkpoint path/to/checkpoint.pt

The server only listens on localhost and is intended for one local player.
"""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import random
import threading
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import torch

from tichu_env.cards import (
    DOG_INDEX,
    DRAGON_INDEX,
    MAHJONG_INDEX,
    NUM_CARDS,
    PHOENIX_INDEX,
    Rank,
    card_name,
    card_rank,
    card_suit,
)
from tichu_env.combinations import Combination, classify_cards, phoenix_interpretations
from tichu_env.display import COMBO_NAMES
from tichu_env.encoding import (
    encode_history,
    encode_pass_source_ids,
    encode_pass_target_ids,
    encode_static,
)
from tichu_env.game_state import GamePhase, GameState
from training import advisor
from training.checkpoints import DEFAULT_PLAY_CHECKPOINT
from training.model_embed import ACTION_DIM
from training.observations import encode_policy_static
from training.play import (
    ai_gt,
    ai_pass,
    ai_select_action,
    ai_select_dragon_recipient,
    ai_select_mahjong_wish,
    load_play_bundle,
)


WEB_ROOT = Path(__file__).with_name("web_static")
DEFAULT_CHECKPOINT = DEFAULT_PLAY_CHECKPOINT
Mode = Literal["play", "advisor"]
AdvisorStage = Literal[
    "first_eight",
    "remaining_six",
    "passing",
    "received",
    "mahjong_holder",
    "play",
]


class WebError(ValueError):
    """An invalid browser action that can be displayed to the user."""


@dataclass(frozen=True)
class PendingAiPlay:
    """A model decision that is waiting to be shown in the browser.

    The state transition is intentionally *not* applied until the browser's
    short reveal timer expires.  Keeping the selected action separate means
    model inference never waits for that timer, while the player still gets a
    chance to see each opponent move arrive at the table.
    """

    id: int
    player: int
    cards: tuple[int, ...]
    call_tichu: bool
    phoenix_high: bool = False


@dataclass(frozen=True)
class PendingPhoenixPlay:
    """A human play waiting for an explicit Phoenix interpretation."""

    cards: tuple[int, ...]
    call_tichu: bool
    choices: tuple[Combination, ...]
    player: int
    advisor_opponent: bool = False


@dataclass
class ReplayFrame:
    """One reversible browser-visible point in an AI replay."""

    gs: GameState
    timeline: list[dict[str, Any]]
    last_event: str
    pending_ai_play: PendingAiPlay | None
    waiting_for_wish: bool
    wish_player: int | None
    next_ai_play_id: int
    next_event_id: int
    round_end_recorded: bool
    recorded_gt_decisions: dict[int, bool]
    recorded_passes: dict[int, tuple[int, int, int]]
    rng_state: object


@dataclass
class LiveUndoFrame:
    """Complete browser-controller state before one accepted manual action."""

    gs: GameState | None
    replay_start_state: GameState | None
    timeline: list[dict[str, Any]]
    last_event: str
    pending_ai_play: PendingAiPlay | None
    pending_phoenix_play: PendingPhoenixPlay | None
    waiting_for_wish: bool
    wish_player: int | None
    next_ai_play_id: int
    next_event_id: int
    round_end_recorded: bool
    advisor_stage: AdvisorStage
    advisor_first_eight: tuple[int, ...]
    advisor_gt_probability: float | None
    advisor_gt_callers: set[int]
    advisor_outgoing: tuple[int, int, int] | None
    recorded_gt_decisions: dict[int, bool]
    recorded_passes: dict[int, tuple[int, int, int]]
    show_mode_menu: bool
    rng_state: object
    global_random_state: object
    torch_rng_state: torch.Tensor
    cuda_rng_states: list[torch.Tensor] | None


def _sort_cards(cards: set[int] | list[int]) -> list[int]:
    """Use the same rank-first ordering as the terminal display."""

    def key(card: int) -> tuple[int, int]:
        if card == MAHJONG_INDEX:
            return (0, 0)
        if card == DOG_INDEX:
            return (99, 0)
        if card == PHOENIX_INDEX:
            return (99, 1)
        if card == DRAGON_INDEX:
            return (99, 2)
        rank = card_rank(card)
        suit = card_suit(card)
        return (rank.value if rank is not None else 99, suit.value if suit is not None else 0)

    return sorted(cards, key=key)


def _card_payload(card: int) -> dict[str, Any]:
    """A card record understood by the original Tichu deck-sheet renderer.

    Rows zero through three are Jade, Pagoda, Star, and Sword.  Each puts
    ranks 2--Ace in columns zero through twelve.  Row four starts with the
    four distinct special cards.  This is an original project asset, not a
    crop of a commercial Tichu deck.
    """

    if card < 52:
        rank = card_rank(card)
        suit = card_suit(card)
        assert rank is not None and suit is not None
        sprite_row = suit.value
        sprite_col = rank.value - Rank.TWO.value
        return {
            "index": card,
            "name": card_name(card),
            "sprite_row": sprite_row,
            "sprite_col": sprite_col,
            "special": False,
        }

    special_sprites = {
        MAHJONG_INDEX: (4, 0),
        DOG_INDEX: (4, 1),
        PHOENIX_INDEX: (4, 2),
        DRAGON_INDEX: (4, 3),
    }
    row, col = special_sprites[card]
    return {
        "index": card,
        "name": card_name(card),
        "sprite_row": row,
        "sprite_col": col,
        "special": True,
    }


def _cards_payload(cards: set[int] | list[int]) -> list[dict[str, Any]]:
    return [_card_payload(card) for card in _sort_cards(cards)]


def _hand_cards_payload(gs: GameState, seat: int) -> list[dict[str, Any]]:
    """Return the visible hand with any still-held incoming pass sources."""

    source_by_card = {
        card: source
        for source, card in gs._incoming_pass_source[seat].items()
        if card in gs.hands[seat]
    }
    relation_by_offset = {
        1: ("left", "←", "left opponent"),
        2: ("partner", "↑", "partner/across"),
        3: ("right", "→", "right opponent"),
    }
    payloads = _cards_payload(gs.hands[seat])
    for payload in payloads:
        source = source_by_card.get(payload["index"])
        if source is None:
            continue
        relation, label, description = relation_by_offset[(source - seat) % 4]
        payload["received_from"] = {
            "player": source,
            "relation": relation,
            "label": label,
            "description": f"Received from {description} (P{source})",
        }
    return payloads


def _combo_name(cards: list[int]) -> str:
    combo = classify_cards(cards) if cards else None
    return COMBO_NAMES.get(combo.type, "") if combo else ("Pass" if not cards else "")


def _rank_name(rank: int) -> str:
    return "1" if rank == 1 else Rank(rank).name.title()


def _canonical_play_action(
    cards: list[int],
    call_tichu: bool,
    legal_actions: list[tuple[list[int], bool]],
) -> tuple[list[int], bool] | None:
    """Resolve a browser selection to the game's canonical legal action."""

    selected = tuple(sorted(cards))
    matches = [
        (legal_cards, legal_call_tichu)
        for legal_cards, legal_call_tichu in legal_actions
        if (
            legal_call_tichu == call_tichu
            and tuple(sorted(legal_cards)) == selected
        )
    ]
    if not matches:
        return None

    # Avoid relying on click order or the enumerator's insertion order.
    return min(matches, key=lambda action: tuple(action[0]))


@dataclass
class LocalGame:
    """Single local game/controller exposed through the browser API."""

    mode: Mode
    seat: int
    show_mode_menu: bool = False
    policy: Any = None
    pass_policy: Any = None
    gt_policy: Any = None
    kind: str = "history"
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    rng: random.Random = field(default_factory=random.Random)
    gs: GameState | None = None
    # A frozen copy of the dealt round before any Grand Tichu, passing, or
    # play decisions. The setup choices alongside it keep opponent GT and
    # passing fixed while the replayed seat tries the selected AI instead.
    replay_start_state: GameState | None = None
    recorded_gt_decisions: dict[int, bool] = field(default_factory=dict)
    recorded_passes: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    spectating: bool = False
    replay_frames: list[ReplayFrame] = field(default_factory=list)
    replay_cursor: int = -1
    waiting_for_wish: bool = False
    wish_player: int | None = None
    # AI model inference runs as soon as an opponent reaches a normal play.
    # The selected action waits here only for the browser's visual reveal.
    pending_ai_play: PendingAiPlay | None = None
    pending_phoenix_play: PendingPhoenixPlay | None = None
    undo_stack: list[LiveUndoFrame] = field(default_factory=list)
    pending_phoenix_undo: LiveUndoFrame | None = None
    _next_ai_play_id: int = 1
    last_event: str = ""
    timeline: list[dict[str, Any]] = field(default_factory=list)
    _next_event_id: int = 1
    _round_end_recorded: bool = False
    advisor_stage: AdvisorStage = "first_eight"
    advisor_first_eight: tuple[int, ...] = ()
    advisor_gt_probability: float | None = None
    advisor_gt_callers: set[int] = field(default_factory=set)
    advisor_outgoing: tuple[int, int, int] | None = None

    def _capture_live_undo(self) -> LiveUndoFrame:
        cuda_rng_states = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else None
        )
        return LiveUndoFrame(
            gs=copy.deepcopy(self.gs),
            replay_start_state=copy.deepcopy(self.replay_start_state),
            timeline=copy.deepcopy(self.timeline),
            last_event=self.last_event,
            pending_ai_play=self.pending_ai_play,
            pending_phoenix_play=self.pending_phoenix_play,
            waiting_for_wish=self.waiting_for_wish,
            wish_player=self.wish_player,
            next_ai_play_id=self._next_ai_play_id,
            next_event_id=self._next_event_id,
            round_end_recorded=self._round_end_recorded,
            advisor_stage=self.advisor_stage,
            advisor_first_eight=self.advisor_first_eight,
            advisor_gt_probability=self.advisor_gt_probability,
            advisor_gt_callers=set(self.advisor_gt_callers),
            advisor_outgoing=self.advisor_outgoing,
            recorded_gt_decisions=dict(self.recorded_gt_decisions),
            recorded_passes=dict(self.recorded_passes),
            show_mode_menu=self.show_mode_menu,
            rng_state=self.rng.getstate(),
            global_random_state=random.getstate(),
            torch_rng_state=torch.get_rng_state().clone(),
            cuda_rng_states=cuda_rng_states,
        )

    def _push_live_undo(self, frame: LiveUndoFrame) -> None:
        self.undo_stack.append(frame)

    def _clear_live_undo(self) -> None:
        self.undo_stack = []
        self.pending_phoenix_undo = None

    def _restore_live_undo(self, frame: LiveUndoFrame) -> None:
        self.gs = copy.deepcopy(frame.gs)
        if self.gs is not None:
            self.gs._invalidate_cache()
        self.replay_start_state = copy.deepcopy(frame.replay_start_state)
        if self.replay_start_state is not None:
            self.replay_start_state._invalidate_cache()
        self.timeline = copy.deepcopy(frame.timeline)
        self.last_event = frame.last_event
        self.pending_ai_play = frame.pending_ai_play
        self.pending_phoenix_play = frame.pending_phoenix_play
        self.pending_phoenix_undo = None
        self.waiting_for_wish = frame.waiting_for_wish
        self.wish_player = frame.wish_player
        self._next_ai_play_id = frame.next_ai_play_id
        self._next_event_id = frame.next_event_id
        self._round_end_recorded = frame.round_end_recorded
        self.advisor_stage = frame.advisor_stage
        self.advisor_first_eight = frame.advisor_first_eight
        self.advisor_gt_probability = frame.advisor_gt_probability
        self.advisor_gt_callers = set(frame.advisor_gt_callers)
        self.advisor_outgoing = frame.advisor_outgoing
        self.recorded_gt_decisions = dict(frame.recorded_gt_decisions)
        self.recorded_passes = dict(frame.recorded_passes)
        self.show_mode_menu = frame.show_mode_menu
        self.rng.setstate(frame.rng_state)
        random.setstate(frame.global_random_state)
        torch.set_rng_state(frame.torch_rng_state)
        if frame.cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(frame.cuda_rng_states)

    def undo_live(self) -> None:
        if self.spectating:
            raise WebError("Live undo is unavailable during AI replay.")
        if not self.undo_stack:
            raise WebError("There is no live action to undo.")
        self._restore_live_undo(self.undo_stack.pop())

    def _can_undo(self) -> bool:
        return bool(self.undo_stack) and not self.spectating

    def _clear_replay_navigation(self) -> None:
        self.replay_frames = []
        self.replay_cursor = -1

    def _capture_replay_frame(self) -> None:
        if not self.spectating or self.gs is None:
            return
        if self.replay_cursor < len(self.replay_frames) - 1:
            self.replay_frames = self.replay_frames[:self.replay_cursor + 1]
        self.replay_frames.append(
            ReplayFrame(
                gs=copy.deepcopy(self.gs),
                timeline=copy.deepcopy(self.timeline),
                last_event=self.last_event,
                pending_ai_play=self.pending_ai_play,
                waiting_for_wish=self.waiting_for_wish,
                wish_player=self.wish_player,
                next_ai_play_id=self._next_ai_play_id,
                next_event_id=self._next_event_id,
                round_end_recorded=self._round_end_recorded,
                recorded_gt_decisions=dict(self.recorded_gt_decisions),
                recorded_passes=dict(self.recorded_passes),
                rng_state=self.rng.getstate(),
            )
        )
        self.replay_cursor = len(self.replay_frames) - 1

    def _restore_replay_frame(self, cursor: int) -> None:
        frame = self.replay_frames[cursor]
        self.gs = copy.deepcopy(frame.gs)
        self.gs._invalidate_cache()
        self.timeline = copy.deepcopy(frame.timeline)
        self.last_event = frame.last_event
        self.pending_ai_play = frame.pending_ai_play
        self.waiting_for_wish = frame.waiting_for_wish
        self.wish_player = frame.wish_player
        self._next_ai_play_id = frame.next_ai_play_id
        self._next_event_id = frame.next_event_id
        self._round_end_recorded = frame.round_end_recorded
        self.recorded_gt_decisions = dict(frame.recorded_gt_decisions)
        self.recorded_passes = dict(frame.recorded_passes)
        self.rng.setstate(frame.rng_state)
        self.replay_cursor = cursor

    def choose_mode(self, mode: Mode) -> None:
        """Switch workflows and discard state owned by the previous mode."""
        if mode not in ("play", "advisor"):
            raise WebError("Mode must be play or advisor.")
        self.mode = mode
        self._clear_live_undo()
        self.show_mode_menu = False
        self.gs = None
        self.replay_start_state = None
        self.spectating = False
        self._clear_replay_navigation()
        self.waiting_for_wish = False
        self.wish_player = None
        self.pending_ai_play = None
        self.pending_phoenix_play = None
        self._next_ai_play_id = 1
        self.timeline = []
        self._next_event_id = 1
        self._round_end_recorded = False
        self.advisor_stage = "first_eight"
        self.advisor_first_eight = ()
        self.advisor_gt_probability = None
        self.advisor_gt_callers = set()
        self.advisor_outgoing = None
        self.recorded_gt_decisions = {}
        self.recorded_passes = {}
        self.last_event = (
            "Start a visual round."
            if mode == "play"
            else "Enter your first eight cards."
        )

    def open_mode_menu(self) -> None:
        self.show_mode_menu = True
        self.last_event = "Choose Play or Advisor."

    def _player_name(self, player: int) -> str:
        return "You" if player == self.seat and not self.spectating else f"P{player}"

    def _record_event(
        self,
        event_type: str,
        text: str,
        *,
        player: int | None = None,
        cards: list[int] | None = None,
        recipient: int | None = None,
    ) -> None:
        """Append one browser-visible event and make it the status message."""
        event = {
            "id": self._next_event_id,
            "type": event_type,
            "text": text,
            "player": player,
            "recipient": recipient,
            "cards": _cards_payload(cards or []),
        }
        self._next_event_id += 1
        self.timeline.append(event)
        self.last_event = text

    def _start_timeline(self, message: str) -> None:
        self.timeline = []
        self._next_event_id = 1
        self._round_end_recorded = False
        self._record_event("round", message)

    def _record_pass_submission(self, player: int, cards: list[int]) -> None:
        """Record the directed pass without leaking an opponent's hand."""
        if self.mode == "play" and not self.spectating:
            self.recorded_passes[player] = tuple(cards)
        recipients = [
            (player + 1) % 4,
            (player + 2) % 4,
            (player + 3) % 4,
        ]
        directions = ", ".join(
            f"P{recipient} ({label})"
            for recipient, label in zip(recipients, ("left", "partner", "right"))
        )
        if player == self.seat and not self.spectating:
            directed_cards = ", ".join(
                f"{card_name(card)} → P{recipient} ({label})"
                for card, recipient, label in zip(
                    cards, recipients, ("left", "partner", "right")
                )
            )
            text = f"You submitted directed passes: {directed_cards}."
        else:
            text = f"P{player} submitted directed passes to {directions}."
        self._record_event(
            "pass_submission",
            text,
            player=player,
            # Passing choices remain secret for AI seats, but the local
            # player can review their own sent cards in the timeline.
            cards=cards if player == self.seat and not self.spectating else None,
        )

    def _record_new_finishes(self, finished_before: set[int]) -> None:
        """Surface finishes caused by a play or Dragon gift, including cleanup."""
        gs = self._require_game()
        for position, player in enumerate(gs.finish_order, start=1):
            if player not in finished_before:
                self._record_event(
                    "finish",
                    f"{self._player_name(player)} finished #{position}.",
                    player=player,
                )
        if gs.round_over and not self._round_end_recorded:
            self._round_end_recorded = True
            self._record_event(
                "round_end",
                "Round over. "
                f"Team A (P0 + P2) {gs.round_scores[0]:+d}; "
                f"Team B (P1 + P3) {gs.round_scores[1]:+d}.",
            )

    def _record_play(
        self,
        player: int,
        cards: list[int],
        *,
        call_tichu: bool,
        previous_wish: int | None,
        finished_before: set[int],
    ) -> None:
        """Record public play information after the game-state transition."""
        gs = self._require_game()
        if call_tichu:
            self._record_event(
                "declaration",
                f"{self._player_name(player)} called Tichu.",
                player=player,
            )
        if cards:
            combo = gs.trick.leading_combo
            phoenix_detail = (
                f" with Phoenix as {_rank_name(combo.phoenix_as)}"
                if (
                    PHOENIX_INDEX in cards
                    and combo is not None
                    and combo.phoenix_as is not None
                )
                else ""
            )
            self._record_event(
                "play",
                f"{self._player_name(player)} played {_combo_name(cards)}"
                f"{phoenix_detail}.",
                player=player,
                cards=cards,
            )
        else:
            self._record_event("pass", f"{self._player_name(player)} passed.", player=player)
        if previous_wish is not None and gs.mahjong_wish is None:
            rank = Rank(previous_wish).name.title()
            self._record_event("wish", f"Mahjong wish for {rank} was fulfilled.")
        self._record_new_finishes(finished_before)

    def new_round(self, seed: int | None = None) -> None:
        """Start a normal visual-play round, including GT and passing phases."""
        if self.mode != "play":
            raise WebError("New dealt rounds are available in play mode only.")
        self._clear_live_undo()
        self.gs = GameState(track_history=(self.kind == "history"))
        # Browser play is human-vs-AI, so this matches training.play exactly.
        self.gs.reset(seed=seed, with_calling=True, with_passing=True)
        self.replay_start_state = copy.deepcopy(self.gs)
        self.recorded_gt_decisions = {}
        self.recorded_passes = {}
        self.spectating = False
        self._clear_replay_navigation()
        self.waiting_for_wish = False
        self.wish_player = None
        self.pending_ai_play = None
        self.pending_phoenix_play = None
        self._start_timeline("Round dealt. Choose whether to call Grand Tichu for your first eight cards.")
        self._prepare_ai()

    def setup_advisor(self, card_text: str) -> None:
        """Create a post-pass advisor state as a backward-compatible shortcut."""
        if self.mode != "advisor":
            raise WebError("Advisor setup is only available in advisor mode.")
        cards = advisor.parse_cards(card_text)
        if cards is None:
            raise WebError("Could not parse cards. Use entries such as '3g Jb 10r As Phoenix'.")
        if len(cards) != 14 or len(set(cards)) != 14:
            raise WebError("Enter exactly 14 distinct cards (your hand after passing).")
        undo_frame = self._capture_live_undo()
        self.gs = advisor.setup_game(cards, self.seat, track_history=(self.kind == "history"))
        self.replay_start_state = None
        self.recorded_gt_decisions = {}
        self.recorded_passes = {}
        self.spectating = False
        self._clear_replay_navigation()
        self.waiting_for_wish = False
        self.wish_player = None
        self.pending_ai_play = None
        self.pending_phoenix_play = None
        self.advisor_stage = "play"
        self.advisor_first_eight = ()
        self.advisor_gt_probability = None
        self.advisor_gt_callers = set()
        self.advisor_outgoing = None
        self._start_timeline("Advisor ready. Record each play at the real table, then use its suggestions.")
        self._push_live_undo(undo_frame)

    def reset_advisor(self) -> None:
        if self.mode != "advisor":
            raise WebError("Advisor setup is only available in advisor mode.")
        self._clear_live_undo()
        self.gs = None
        self.replay_start_state = None
        self.recorded_gt_decisions = {}
        self.recorded_passes = {}
        self.spectating = False
        self._clear_replay_navigation()
        self.waiting_for_wish = False
        self.wish_player = None
        self.pending_ai_play = None
        self.pending_phoenix_play = None
        self.advisor_stage = "first_eight"
        self.advisor_first_eight = ()
        self.advisor_gt_probability = None
        self.advisor_gt_callers = set()
        self.advisor_outgoing = None
        self.timeline = []
        self.last_event = "Enter your first eight cards."

    def advisor_set_first_eight(self, card_text: str) -> None:
        """Record the first deal and expose the bundled GT-head recommendation."""
        if self.mode != "advisor":
            raise WebError("First-eight setup is only available in advisor mode.")
        cards = advisor.parse_cards(card_text)
        if cards is None:
            raise WebError(
                "Could not parse cards. Use 3g, m/d/w/p, '2 rgb', or '2-7 r'."
            )
        if len(cards) != 8 or len(set(cards)) != 8:
            raise WebError("Enter exactly eight distinct cards from the first deal.")
        undo_frame = self._capture_live_undo()
        self.gs = None
        self.advisor_first_eight = tuple(cards)
        self.advisor_gt_probability = advisor.suggest_gt(
            self.gt_policy, cards, self.device
        )
        self.advisor_gt_callers = set()
        self.advisor_outgoing = None
        self.advisor_stage = "remaining_six"
        self.last_event = "First eight recorded. Record Grand Tichu calls and the remaining six cards."
        self._push_live_undo(undo_frame)

    def advisor_set_remaining_six(
        self,
        card_text: str,
        gt_callers: list[int],
    ) -> None:
        """Complete the original hand and move the advisor to directed passing."""
        if self.mode != "advisor" or self.advisor_stage != "remaining_six":
            raise WebError("Record the first eight cards before the remaining six.")
        cards = advisor.parse_cards(card_text)
        if cards is None:
            raise WebError(
                "Could not parse cards. Use 3g, m/d/w/p, '2 rgb', or '2-7 r'."
            )
        full_hand = list(self.advisor_first_eight) + cards
        if len(cards) != 6 or len(set(full_hand)) != 14:
            raise WebError("Enter six new distinct cards, for fourteen cards total.")
        if len(set(gt_callers)) != len(gt_callers) or any(
            player < 0 or player >= 4 for player in gt_callers
        ):
            raise WebError("Grand Tichu callers must be distinct seats P0 through P3.")

        undo_frame = self._capture_live_undo()
        self.gs = advisor.setup_game(
            full_hand,
            self.seat,
            track_history=(self.kind == "history"),
        )
        for player in gt_callers:
            self.gs.tichu_declarations[player] = 2
        # Only the local player's real pass is entered. Opponent hands remain a
        # hidden approximation until their public plays reconcile them.
        self.gs.phase = GamePhase.PASSING
        self.gs.current_player = self.seat
        self.advisor_gt_callers = set(gt_callers)
        self.advisor_stage = "passing"
        self.advisor_outgoing = None
        self.replay_start_state = None
        self.spectating = False
        self._clear_replay_navigation()
        self.waiting_for_wish = False
        self.wish_player = None
        self.pending_ai_play = None
        self.pending_phoenix_play = None
        self._start_timeline("Original hand recorded. Choose directed passing cards.")
        for player in sorted(self.advisor_gt_callers):
            self._record_event(
                "declaration",
                f"{self._player_name(player)} called Grand Tichu.",
                player=player,
            )
        self._push_live_undo(undo_frame)

    def advisor_receive_pass(self, card_text: str) -> None:
        """Record cards received from left, partner, and right, in that order."""
        gs = self._require_game()
        if self.mode != "advisor" or self.advisor_stage != "received":
            raise WebError("Choose your three outgoing passing cards first.")
        cards = advisor.parse_cards(card_text)
        if cards is None:
            raise WebError(
                "Could not parse received cards. Enter left, partner, right order."
            )
        if len(cards) != 3:
            raise WebError("Enter exactly three cards: from left, partner, then right.")
        assert self.advisor_outgoing is not None
        undo_frame = self._capture_live_undo()
        try:
            advisor.apply_manual_passing(
                gs,
                self.seat,
                self.advisor_outgoing,
                tuple(cards),
            )
        except ValueError as exc:
            raise WebError(str(exc)) from exc
        self._record_event(
            "pass_received",
            "Received passing cards from left, partner, and right.",
            player=self.seat,
            cards=cards,
        )
        if MAHJONG_INDEX in gs.hands[self.seat]:
            self.advisor_stage = "play"
            self.last_event = "Passing complete. You hold the Mahjong and lead."
        else:
            known_holder = advisor.known_mahjong_holder_from_passing(gs, self.seat)
            if known_holder is not None:
                self.advisor_stage = "play"
                gs.current_player = known_holder
                self._record_event(
                    "mahjong_holder",
                    f"P{known_holder} holds the Mahjong you passed and leads.",
                    player=known_holder,
                )
            else:
                self.advisor_stage = "mahjong_holder"
                gs.current_player = -1
                self.last_event = "Passing complete. Record which opponent holds the Mahjong."
        self._push_live_undo(undo_frame)

    def advisor_set_mahjong_holder(self, holder: int) -> None:
        """Reconcile the approximate opponent deal with the known opening seat."""
        gs = self._require_game()
        if self.mode != "advisor" or self.advisor_stage != "mahjong_holder":
            raise WebError("The advisor is not waiting for a Mahjong holder.")
        if holder == self.seat or holder < 0 or holder >= 4:
            raise WebError("Choose the opponent who holds the Mahjong.")
        known_holder = advisor.known_mahjong_holder_from_passing(gs, self.seat)
        if known_holder is not None and holder != known_holder:
            raise WebError(
                f"You passed the Mahjong to P{known_holder}; P{known_holder} must lead."
            )
        owner = next(
            (player for player in range(4) if MAHJONG_INDEX in gs.hands[player]),
            None,
        )
        if owner is None:
            raise WebError("The simulated deal has no Mahjong holder.")
        undo_frame = self._capture_live_undo()
        if owner != holder:
            protected = set(gs._incoming_pass_source[holder].values())
            swappable = [
                card
                for card in gs.hands[holder]
                if card != MAHJONG_INDEX and card not in protected
            ]
            if not swappable:
                raise WebError("Could not reconcile the Mahjong holder.")
            swap = self.rng.choice(swappable)
            gs.hands[owner].remove(MAHJONG_INDEX)
            gs.hands[owner].add(swap)
            gs.hands[holder].remove(swap)
            gs.hands[holder].add(MAHJONG_INDEX)
            gs._invalidate_cache()
        gs.current_player = holder
        self.advisor_stage = "play"
        self._record_event(
            "mahjong_holder",
            f"P{holder} holds the Mahjong and leads.",
            player=holder,
        )
        self._push_live_undo(undo_frame)

    def _require_game(self) -> GameState:
        if self.gs is None:
            raise WebError("Start a round first.")
        return self.gs

    def _human_turn(self) -> bool:
        gs = self._require_game()
        return (
            not self.spectating
            and not gs.round_over
            and gs.phase == GamePhase.PLAY
            and gs.current_player == self.seat
            and gs.dragon_give_pending is None
            and not self.waiting_for_wish
            and self.pending_phoenix_play is None
        )

    def _queue_ai_play(self, player: int) -> None:
        """Choose an AI play now, but leave the actual game state untouched.

        This uses the same pure selection helper as ``training.play.ai_act``.
        The Mahjong head runs only after the play is revealed and recorded,
        matching the post-play state it receives during evaluation.
        """
        gs = self._require_game()
        if self.pending_ai_play is not None:
            return
        if (
            gs.phase != GamePhase.PLAY
            or player != gs.current_player
            or (player == self.seat and not self.spectating)
            or gs.dragon_give_pending is not None
            or self.waiting_for_wish
        ):
            return

        action = ai_select_action(
            self.policy, gs, player, self.device, self.rng, self.kind
        )
        cards, call_tichu = action[:2]
        phoenix_high = bool(action[2]) if len(action) == 3 else False
        self.pending_ai_play = PendingAiPlay(
            id=self._next_ai_play_id,
            player=player,
            cards=tuple(cards),
            call_tichu=call_tichu,
            phoenix_high=phoenix_high,
        )
        self._next_ai_play_id += 1

    def _pending_ai_play_is_current(self, pending: PendingAiPlay) -> bool:
        """Guard against a stale reveal applying to a changed game state."""
        gs = self._require_game()
        if (
            gs.phase != GamePhase.PLAY
            or gs.round_over
            or gs.current_player != pending.player
            or (pending.player == self.seat and not self.spectating)
            or gs.dragon_give_pending is not None
            or self.waiting_for_wish
        ):
            return False
        legal = (
            gs.get_legal_actions(pending.player)
            if self.kind == "fixed"
            else gs.get_policy_legal_actions(pending.player)
        )
        action = (
            (list(pending.cards), pending.call_tichu)
            if self.kind == "fixed"
            else (
                list(pending.cards),
                pending.call_tichu,
                pending.phoenix_high,
            )
        )
        return action in legal

    def _prepare_ai(self) -> None:
        """Resolve immediate AI-only work, then precompute one visible play.

        Grand Tichu choices, directed passes, and Dragon recipients are game
        administration rather than card reveals, so they happen immediately.
        The first ordinary play reached is selected now and held in
        ``pending_ai_play`` for the browser's half-second reveal beat.
        """
        if self.mode != "play" or self.gs is None or self.gs.round_over:
            return
        if self.pending_ai_play is not None:
            return

        gs = self.gs
        while not gs.round_over:
            if self.waiting_for_wish:
                return
            if gs.dragon_give_pending is not None:
                giver = gs.dragon_give_pending
                if giver == self.seat and not self.spectating:
                    return
                recipient = ai_select_dragon_recipient(
                    self.policy, gs, giver, self.device, self.rng, self.kind
                )
                finished_before = set(gs.finished)
                if not gs.give_dragon_trick(giver, recipient):
                    raise RuntimeError("AI could not give away a Dragon trick")
                self._record_event(
                    "dragon_gift",
                    f"{self._player_name(giver)} gave the Dragon trick to P{recipient}.",
                    player=giver,
                    recipient=recipient,
                )
                self._record_new_finishes(finished_before)
                continue

            player = gs.current_player
            if player == self.seat and not self.spectating:
                return
            if gs.phase == GamePhase.GRAND_TICHU:
                fixed_replay_decision = (
                    self.recorded_gt_decisions.get(player)
                    if self.spectating and player != self.seat
                    else None
                )
                decision = (
                    fixed_replay_decision
                    if fixed_replay_decision is not None
                    else ai_gt(
                        self.gt_policy,
                        _sort_cards(gs.hands[player])[:8],
                        self.device,
                    )
                )
                if not gs.declare_grand_tichu(player, decision):
                    raise RuntimeError(f"AI Grand Tichu decision was rejected for P{player}")
                if not self.spectating:
                    self.recorded_gt_decisions[player] = decision
                result = "called Grand Tichu" if decision else "declined Grand Tichu"
                replay_note = (
                    " (same as played round)"
                    if self.spectating and player != self.seat
                    else ""
                )
                self._record_event(
                    "declaration",
                    f"P{player} {result}{replay_note}.",
                    player=player,
                )
                continue
            if gs.phase == GamePhase.PASSING:
                fixed_replay_pass = (
                    self.recorded_passes.get(player)
                    if self.spectating and player != self.seat
                    else None
                )
                if fixed_replay_pass is None:
                    cards = list(
                        ai_pass(self.pass_policy, gs, player, self.device, self.rng)
                    )
                else:
                    cards = list(fixed_replay_pass)
                    if not gs.pass_cards(player, *cards):
                        raise RuntimeError(
                            f"Recorded pass for P{player} is not legal for this replay deal"
                        )
                self._record_pass_submission(player, cards)
                continue

            self._queue_ai_play(player)
            return

    def start_ai_replay(self) -> None:
        """Replay the completed deal with the AI controlling every seat.

        The saved source is the exact original deal before the 8-card Grand
        Tichu phase. Opponents repeat their original GT and directed passes;
        the replayed seat gets fresh model setup decisions and every play is
        model-controlled.
        """
        if self.mode != "play":
            raise WebError("AI replays are available in visual play mode only.")
        if self.gs is None or not self.gs.round_over:
            raise WebError("Finish the current round before starting its AI replay.")
        if self.replay_start_state is None:
            raise WebError("The original deal is not available for replay.")
        for player in range(4):
            self.recorded_gt_decisions.setdefault(
                player,
                self.gs.tichu_declarations[player] == 2,
            )
            targets = self.gs._outgoing_pass_target[player]
            recipients = ((player + 1) % 4, (player + 2) % 4, (player + 3) % 4)
            if player not in self.recorded_passes and all(
                recipient in targets for recipient in recipients
            ):
                self.recorded_passes[player] = tuple(
                    targets[recipient] for recipient in recipients
                )
        self._clear_live_undo()
        self.gs = copy.deepcopy(self.replay_start_state)
        self.gs._invalidate_cache()
        self.spectating = True
        self._clear_replay_navigation()
        self.waiting_for_wish = False
        self.wish_player = None
        self.pending_ai_play = None
        self.pending_phoenix_play = None
        self._start_timeline(
            f"AI replay started: same deal; P{self.seat} uses current AI setup "
            "choices while opponents repeat their Grand Tichu and directed passes."
        )
        self._prepare_ai()
        self._capture_replay_frame()

    def step_ai_replay(self, direction: int) -> bool:
        """Move one visible play backward or forward in the current replay."""
        if not self.spectating or self.gs is None or direction not in (-1, 1):
            return False
        target = self.replay_cursor + direction
        if 0 <= target < len(self.replay_frames):
            self._restore_replay_frame(target)
            return True
        if direction > 0 and self.pending_ai_play is not None:
            return self.advance_ai_step(self.pending_ai_play.id)
        return False

    def finish_ai_replay(self) -> bool:
        """Advance to the cached or newly computed final replay score."""
        if not self.spectating or self.gs is None:
            return False
        if self.replay_cursor < len(self.replay_frames) - 1:
            self._restore_replay_frame(len(self.replay_frames) - 1)
        for _ in range(2_000):
            if self.gs.round_over:
                return True
            if self.pending_ai_play is None:
                self._prepare_ai()
            if self.pending_ai_play is None:
                raise RuntimeError("AI replay could not reach another visible play")
            if not self.advance_ai_step(self.pending_ai_play.id):
                raise RuntimeError("AI replay could not advance")
        raise RuntimeError("AI replay exceeded the play limit")

    def _ai_pending(self) -> bool:
        """Whether a precomputed visible AI play is waiting for reveal."""
        return self.pending_ai_play is not None

    def advance_ai_step(self, expected_action_id: int | None = None) -> bool:
        """Reveal exactly one already-selected AI play.

        The browser sends the pending action id it observed.  An old timer or
        duplicated request therefore cannot reveal a later play by accident.
        Direct callers may omit the id for backwards-compatible local use.
        """
        if self.mode != "play" or self.gs is None:
            return False
        if self.spectating and self.replay_cursor < len(self.replay_frames) - 1:
            return False
        pending = self.pending_ai_play
        if pending is None:
            return False
        if expected_action_id is not None and expected_action_id != pending.id:
            return False
        if not self._pending_ai_play_is_current(pending):
            self.pending_ai_play = None
            self._prepare_ai()
            return False

        gs = self.gs
        # Clear before applying.  This makes duplicate requests harmless even
        # under a threaded server, and lets the post-play preparation queue
        # only the next distinct opponent action.
        self.pending_ai_play = None
        finished_before = set(gs.finished)
        previous_wish = gs.mahjong_wish
        action = (
            (list(pending.cards), pending.call_tichu)
            if self.kind == "fixed"
            else (
                list(pending.cards),
                pending.call_tichu,
                pending.phoenix_high,
            )
        )
        if not gs.play(pending.player, action):
            self._prepare_ai()
            return False
        self._record_play(
            pending.player,
            list(pending.cards),
            call_tichu=pending.call_tichu,
            previous_wish=previous_wish,
            finished_before=finished_before,
        )
        if MAHJONG_INDEX in pending.cards:
            wish = ai_select_mahjong_wish(
                self.policy, gs, pending.player, self.device, self.rng, self.kind
            )
            gs.set_mahjong_wish(wish)
            if wish is None:
                wish_text = f"{self._player_name(pending.player)} made no Mahjong wish."
            else:
                wish_text = (
                    f"{self._player_name(pending.player)} set a Mahjong wish for "
                    f"{Rank(wish).name.title()}."
                )
            self._record_event("wish", wish_text, player=pending.player)
        self._prepare_ai()
        if self.spectating:
            self._capture_replay_frame()
        return True

    def declare_grand_tichu(self, declare: bool) -> None:
        gs = self._require_game()
        if gs.phase != GamePhase.GRAND_TICHU or gs.current_player != self.seat:
            raise WebError("It is not your Grand Tichu decision.")
        undo_frame = self._capture_live_undo()
        if not gs.declare_grand_tichu(self.seat, declare):
            raise WebError("That Grand Tichu decision was not accepted.")
        self.recorded_gt_decisions[self.seat] = declare
        self._push_live_undo(undo_frame)
        result = "called Grand Tichu" if declare else "declined Grand Tichu"
        self._record_event("declaration", f"You {result}.", player=self.seat)
        self._prepare_ai()

    def submit_pass(self, cards: list[int]) -> None:
        gs = self._require_game()
        if gs.phase != GamePhase.PASSING or gs.current_player != self.seat:
            raise WebError("It is not your passing turn.")
        if len(cards) != 3 or len(set(cards)) != 3:
            raise WebError("Choose exactly three distinct cards, in left / partner / right order.")
        undo_frame = self._capture_live_undo()
        if self.mode == "advisor":
            if self.advisor_stage != "passing":
                raise WebError("The advisor is not waiting for outgoing passing cards.")
            if not set(cards).issubset(gs.hands[self.seat]):
                raise WebError("Those passing cards are not in your original hand.")
            self.advisor_outgoing = tuple(cards)
            self.advisor_stage = "received"
            self._record_pass_submission(self.seat, cards)
            self.last_event = "Enter cards received from left, partner, and right."
            self._push_live_undo(undo_frame)
            return
        if not gs.pass_cards(self.seat, *cards):
            raise WebError("Those passing cards are not valid for your current hand.")
        self._push_live_undo(undo_frame)
        self._record_pass_submission(self.seat, cards)
        self._prepare_ai()

    def _valid_phoenix_choices(
        self,
        cards: list[int],
        call_tichu: bool,
        *,
        player: int | None = None,
        advisor_opponent: bool = False,
    ) -> list[Combination]:
        """Filter physical-card interpretations through authoritative play."""

        gs = self._require_game()
        valid: list[Combination] = []
        acting_player = self.seat if player is None else player
        random_state = random.getstate()
        try:
            for combo in phoenix_interpretations(cards):
                trial = copy.deepcopy(gs)
                trial._invalidate_cache()
                if advisor_opponent:
                    accepted = advisor.force_play(
                        trial,
                        acting_player,
                        list(cards),
                        self.seat,
                        call_tichu,
                        phoenix_as=combo.phoenix_as,
                    )
                else:
                    accepted = trial.play(
                        acting_player,
                        (list(cards), call_tichu),
                        phoenix_as=combo.phoenix_as,
                    )
                if accepted:
                    valid.append(combo)
        finally:
            random.setstate(random_state)
        return valid

    def play_cards(
        self,
        cards: list[int],
        call_tichu: bool = False,
        *,
        phoenix_as: int | None = None,
        _undo_frame: LiveUndoFrame | None = None,
    ) -> None:
        gs = self._require_game()
        if not self._human_turn():
            raise WebError("It is not your turn to play.")
        wanted = _canonical_play_action(
            cards,
            bool(call_tichu),
            gs.get_legal_actions(self.seat),
        )
        if wanted is None:
            raise WebError("That card combination is not a legal play here.")
        canonical_cards = wanted[0]
        selected_phoenix_as = None
        if PHOENIX_INDEX in canonical_cards:
            choices = self._valid_phoenix_choices(canonical_cards, call_tichu)
            if not choices:
                raise WebError("That Phoenix combination is not legal here.")
            if phoenix_as is None and len(choices) > 1:
                self.pending_phoenix_undo = (
                    _undo_frame if _undo_frame is not None else self._capture_live_undo()
                )
                self.pending_phoenix_play = PendingPhoenixPlay(
                    cards=tuple(canonical_cards),
                    call_tichu=call_tichu,
                    choices=tuple(choices),
                    player=self.seat,
                )
                self.last_event = "Choose what rank the Phoenix represents."
                return
            selected = [combo for combo in choices if combo.phoenix_as == phoenix_as]
            if phoenix_as is not None and not selected:
                raise WebError("That Phoenix value is not legal for this play.")
            selected_phoenix_as = (
                phoenix_as if phoenix_as is not None else choices[0].phoenix_as
            )
        elif phoenix_as is not None:
            raise WebError("A Phoenix value was supplied without the Phoenix.")
        undo_frame = _undo_frame if _undo_frame is not None else self._capture_live_undo()
        finished_before = set(gs.finished)
        previous_wish = gs.mahjong_wish
        if not gs.play(self.seat, wanted, phoenix_as=selected_phoenix_as):
            raise WebError("The play could not be applied.")
        self.pending_phoenix_undo = None
        self._push_live_undo(undo_frame)
        self._record_play(
            self.seat,
            canonical_cards,
            call_tichu=call_tichu,
            previous_wish=previous_wish,
            finished_before=finished_before,
        )
        if MAHJONG_INDEX in canonical_cards and not gs.round_over:
            self.waiting_for_wish = True
            self.wish_player = self.seat
            self.last_event = "Choose a Mahjong wish, or choose no wish."
        else:
            self._prepare_ai()

    def choose_phoenix(self, phoenix_as: int) -> None:
        pending = self.pending_phoenix_play
        if pending is None:
            raise WebError("There is no Phoenix interpretation waiting for you.")
        if phoenix_as not in {combo.phoenix_as for combo in pending.choices}:
            raise WebError("That Phoenix value is not available for this play.")
        undo_frame = self.pending_phoenix_undo
        if undo_frame is None:
            raise WebError("The Phoenix play has no undo baseline.")
        self.pending_phoenix_play = None
        self.pending_phoenix_undo = None
        try:
            if pending.advisor_opponent:
                self._apply_advisor_opponent_play(
                    pending.player,
                    list(pending.cards),
                    pending.call_tichu,
                    phoenix_as=phoenix_as,
                    _undo_frame=undo_frame,
                )
            else:
                self.play_cards(
                    list(pending.cards),
                    pending.call_tichu,
                    phoenix_as=phoenix_as,
                    _undo_frame=undo_frame,
                )
        except Exception:
            self.pending_phoenix_play = pending
            self.pending_phoenix_undo = undo_frame
            raise

    def cancel_phoenix(self) -> None:
        if self.pending_phoenix_play is None:
            raise WebError("There is no Phoenix interpretation waiting for you.")
        self.pending_phoenix_play = None
        self.pending_phoenix_undo = None
        self.last_event = "Phoenix play cancelled. Choose cards to play."

    def set_wish(self, wish: int | None) -> None:
        gs = self._require_game()
        if not self.waiting_for_wish:
            raise WebError("There is no Mahjong wish waiting for you.")
        if wish is not None and not 2 <= wish <= 14:
            raise WebError("A Mahjong wish must be a rank from 2 through Ace.")
        gs.set_mahjong_wish(wish)
        self.waiting_for_wish = False
        player = self.wish_player if self.wish_player is not None else self.seat
        self.wish_player = None
        text = (
            f"{self._player_name(player)} chose no Mahjong wish."
            if wish is None
            else f"{self._player_name(player)} set a Mahjong wish for {Rank(wish).name.title()}."
        )
        self._record_event("wish", text, player=player)
        self._prepare_ai()

    def give_dragon(self, recipient: int) -> None:
        gs = self._require_game()
        giver = gs.dragon_give_pending
        # In advisor mode an opponent's real-world Dragon gift is recorded here.
        recording = (
            self.mode == "advisor" and giver is not None and giver != self.seat
        )
        if giver != self.seat and not recording:
            raise WebError("You do not have a Dragon trick to give away.")
        if recipient not in gs.get_dragon_give_options(giver):
            raise WebError("The Dragon trick must go to an opponent.")
        undo_frame = self._capture_live_undo()
        finished_before = set(gs.finished)
        if not gs.give_dragon_trick(giver, recipient):
            raise WebError("The Dragon recipient was not accepted.")
        self._push_live_undo(undo_frame)
        self._record_event(
            "dragon_gift",
            f"{self._player_name(giver)} gave the Dragon trick to P{recipient}.",
            player=giver,
            recipient=recipient,
        )
        self._record_new_finishes(finished_before)
        self._prepare_ai()

    def advisor_opponent_play(self, player: int, card_text: str, call_tichu: bool = False) -> None:
        """Record a real-world opponent move through advisor.py's force helper."""
        cards = advisor.parse_cards(card_text)
        if cards is None or len(cards) != len(set(cards)):
            raise WebError("Could not parse distinct cards for that play.")
        self._apply_advisor_opponent_play(player, cards, call_tichu)

    def _apply_advisor_opponent_play(
        self,
        player: int,
        cards: list[int],
        call_tichu: bool,
        *,
        phoenix_as: int | None = None,
        _undo_frame: LiveUndoFrame | None = None,
    ) -> None:
        """Validate and apply already-parsed public opponent cards."""
        if self.mode != "advisor":
            raise WebError("Opponent recording is only available in advisor mode.")
        gs = self._require_game()
        if self.waiting_for_wish:
            raise WebError("Record the pending Mahjong wish before another play.")
        if self.pending_phoenix_play is not None:
            raise WebError(
                "Resolve or cancel the pending Phoenix interpretation before another play."
            )
        if player == self.seat:
            raise WebError("Use the card controls for your own play.")
        if player != gs.current_player:
            raise WebError(f"It is P{gs.current_player}'s turn, not P{player}'s.")
        selected_phoenix_as = None
        if PHOENIX_INDEX in cards:
            choices = self._valid_phoenix_choices(
                cards,
                call_tichu,
                player=player,
                advisor_opponent=True,
            )
            if not choices:
                raise WebError("That Phoenix combination is not legal here.")
            if phoenix_as is None and len(choices) > 1:
                self.pending_phoenix_undo = (
                    _undo_frame if _undo_frame is not None else self._capture_live_undo()
                )
                self.pending_phoenix_play = PendingPhoenixPlay(
                    cards=tuple(cards),
                    call_tichu=call_tichu,
                    choices=tuple(choices),
                    player=player,
                    advisor_opponent=True,
                )
                self.last_event = "Choose what rank the opponent's Phoenix represents."
                return
            selected = [combo for combo in choices if combo.phoenix_as == phoenix_as]
            if phoenix_as is not None and not selected:
                raise WebError("That Phoenix value is not legal for this play.")
            selected_phoenix_as = (
                phoenix_as if phoenix_as is not None else choices[0].phoenix_as
            )
        elif phoenix_as is not None:
            raise WebError("A Phoenix value was supplied without the Phoenix.")
        undo_frame = _undo_frame if _undo_frame is not None else self._capture_live_undo()
        # force_play contains advisor's existing unknown-hand reconciliation.
        finished_before = set(gs.finished)
        previous_wish = gs.mahjong_wish
        if not advisor.force_play(
            gs,
            player,
            cards,
            self.seat,
            call_tichu,
            phoenix_as=selected_phoenix_as,
        ):
            raise WebError("That is not a legal play at the current table state.")
        self.pending_phoenix_undo = None
        self._push_live_undo(undo_frame)
        self._record_play(
            player,
            cards,
            call_tichu=call_tichu,
            previous_wish=previous_wish,
            finished_before=finished_before,
        )
        if MAHJONG_INDEX in cards and not gs.round_over:
            self.waiting_for_wish = True
            self.wish_player = player
            self.last_event = f"Recorded P{player}'s Mahjong. Record their wish, if any."

    def _suggestions(self) -> list[dict[str, Any]]:
        gs = self._require_game()
        if self.policy is None or not self._human_turn():
            return []
        suggestions = advisor.suggest(self.policy, gs, self.seat, self.device, kind=self.kind)
        return [
            {
                "cards": _cards_payload(cards),
                "indices": list(cards),
                "call_tichu": call_tichu,
                "phoenix_high": phoenix_high,
                "probability": probability,
                "label": (
                    f"{_combo_name(cards) or 'Play'} (higher Phoenix)"
                    if phoenix_high
                    else (_combo_name(cards) or "Play")
                ),
            }
            for cards, call_tichu, phoenix_high, probability in suggestions
        ]

    def _pass_suggestion(self) -> list[dict[str, Any]]:
        gs = self._require_game()
        if self.pass_policy is None or gs.phase != GamePhase.PASSING or gs.current_player != self.seat:
            return []
        suggestion = advisor.suggest_pass(self.pass_policy, gs, self.seat, self.device)
        return _cards_payload(list(suggestion)) if suggestion is not None else []

    def _value_is_in_distribution(self) -> bool:
        """Whether the critic received this observer/state type in training.

        Main-policy rollouts train a value row from the active player's private
        view, immediately before a normal play.  Asking the current checkpoint
        for a human-seat value while another player acts is therefore an
        out-of-distribution private-information query, and it can fluctuate
        sharply after the human commits an otherwise unsurprising action.
        """
        gs = self._require_game()
        return (
            self.policy is not None
            and not self.spectating
            and not gs.round_over
            and gs.phase == GamePhase.PLAY
            and gs.current_player == self.seat
            and not self.waiting_for_wish
            and gs.dragon_give_pending is None
        )

    @torch.no_grad()
    def _predicted_value(self) -> float | None:
        """Return the critic's current estimate from the local player's seat.

        Values are trained in the observer's team-relative frame, so encoding
        from ``self.seat`` is already the requested human-team perspective.
        The critic is deliberately queried without choosing or applying an
        action; it is only a read-only visual diagnostic.
        """
        if self.policy is None or self.gs is None:
            return None
        gs = self.gs
        obs = torch.as_tensor(
            encode_policy_static(self.policy, gs, self.seat),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        pass_src = torch.as_tensor(
            encode_pass_source_ids(gs, self.seat), dtype=torch.int64, device=self.device
        ).unsqueeze(0)
        pass_tgt = torch.as_tensor(
            encode_pass_target_ids(gs, self.seat), dtype=torch.int64, device=self.device
        ).unsqueeze(0)
        # A single dummy action is enough because the value head is computed
        # before action logits.  This keeps the exact model path for both
        # action-embedding variants without enumerating legal combinations.
        dummy_action = torch.zeros((1, 1, ACTION_DIM), dtype=torch.float32, device=self.device)
        if self.kind == "history":
            hist, hist_len = encode_history(gs, self.seat)
            max_history = max(int(hist_len), 1)
            history = torch.as_tensor(
                hist[:max_history][None, :], dtype=torch.float32, device=self.device
            )
            _, value = self.policy._forward_scores(
                obs,
                dummy_action,
                history,
                torch.tensor([hist_len], dtype=torch.int32, device=self.device),
                pass_src,
                pass_tgt,
            )
        elif self.kind == "fixed":
            # The fixed-action policy does not need an action mask to produce
            # a value; bypassing its masked policy head avoids making a dummy
            # 2,000-action mask just for a read-only number.
            value = self.policy.value_head(self.policy.trunk(obs)).squeeze(-1)
        else:
            _, value = self.policy._forward_scores(obs, dummy_action, pass_src, pass_tgt)
        return float(value.item())

    def state(self) -> dict[str, Any]:
        """Serialize only information appropriate for the local player."""
        if self.gs is None:
            return {
                "mode": self.mode,
                "show_mode_menu": self.show_mode_menu,
                "seat": self.seat,
                "ready": False,
                "advisor_stage": self.advisor_stage if self.mode == "advisor" else None,
                "first_eight": _cards_payload(list(self.advisor_first_eight)),
                "gt_probability": self.advisor_gt_probability,
                "has_gt_model": self.gt_policy is not None,
                "can_undo": self._can_undo(),
                "message": (
                    self.last_event or "Enter your first eight cards."
                    if self.mode == "advisor"
                    else "Start a visual round."
                ),
            }
        gs = self.gs
        table_combo = _combo_name(gs.trick.cards_on_table)
        rank_names = {rank.value: rank.name.title() for rank in Rank}
        seats = []
        for player in range(4):
            relation = ("You" if player == self.seat else
                        "Left" if player == (self.seat + 1) % 4 else
                        "Partner" if player == (self.seat + 2) % 4 else "Right")
            seats.append(
                {
                    "player": player,
                    "relation": relation,
                    "count": len(gs.hands[player]),
                    "current": player == gs.current_player and not gs.round_over,
                    "finished": player in gs.finish_order,
                    "finish_position": gs.finish_order.index(player) + 1 if player in gs.finish_order else None,
                    "tichu": gs.tichu_declarations[player],
                }
            )
        return {
            "mode": self.mode,
            "show_mode_menu": self.show_mode_menu,
            "seat": self.seat,
            "ready": True,
            "advisor_stage": self.advisor_stage if self.mode == "advisor" else None,
            "first_eight": _cards_payload(list(self.advisor_first_eight)),
            "gt_probability": self.advisor_gt_probability,
            "has_gt_model": self.gt_policy is not None,
            "phase": gs.phase.name.lower(),
            "current_player": gs.current_player,
            "human_turn": self._human_turn(),
            "can_undo": self._can_undo(),
            "spectating": self.spectating,
            "replay_cursor": self.replay_cursor if self.spectating else None,
            "replay_frame_count": len(self.replay_frames) if self.spectating else 0,
            "replay_can_back": self.spectating and self.replay_cursor > 0,
            "replay_can_forward": (
                self.spectating
                and (
                    self.replay_cursor < len(self.replay_frames) - 1
                    or self.pending_ai_play is not None
                )
            ),
            "can_replay": (
                self.mode == "play"
                and gs.round_over
                and self.replay_start_state is not None
            ),
            "waiting_for_wish": self.waiting_for_wish,
            "phoenix_choices": [
                {
                    "value": combo.phoenix_as,
                    "label": f"Phoenix as {_rank_name(combo.phoenix_as)}",
                    "combo": COMBO_NAMES.get(combo.type, "Play"),
                    "rank": combo.rank,
                }
                for combo in (
                    self.pending_phoenix_play.choices
                    if self.pending_phoenix_play is not None
                    else ()
                )
                if combo.phoenix_as is not None
            ],
            "dragon_give_pending": gs.dragon_give_pending,
            "round_over": gs.round_over,
            "scores": list(gs.round_scores),
            "finish_order": list(gs.finish_order),
            "seats": seats,
            "hand": _hand_cards_payload(gs, self.seat),
            "table": _cards_payload(gs.trick.cards_on_table),
            "table_combo": table_combo,
            "table_leader": gs.trick.leading_player,
            "passes": gs.trick.passes,
            "wish": rank_names.get(gs.mahjong_wish) if gs.mahjong_wish else None,
            "message": self.last_event,
            "history": list(self.timeline),
            "ai_pending": self._ai_pending(),
            # The client delays only the reveal of this already-computed play;
            # it never delays the model call itself.
            "ai_delay_ms": 500 if self.pending_ai_play is not None else 0,
            "ai_action_id": self.pending_ai_play.id if self.pending_ai_play else None,
            "ai_player": self.pending_ai_play.player if self.pending_ai_play else None,
            # The current checkpoint was trained only on active-player
            # normal-play states.  Do not present an out-of-distribution
            # private-information value as a stable human-team estimate.
            "has_model_value": self.policy is not None,
            "value_available": self._value_is_in_distribution(),
            "predicted_value": (
                self._predicted_value() if self._value_is_in_distribution() else None
            ),
            "suggestions": self._suggestions(),
            "pass_suggestion": self._pass_suggestion(),
        }


class TichuWebServer(ThreadingHTTPServer):
    """HTTP server with one lock-protected local game state."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], game: LocalGame):
        self.game = game
        self.game_lock = threading.RLock()
        super().__init__(address, TichuWebHandler)


class TichuWebHandler(SimpleHTTPRequestHandler):
    """Serve static files and a deliberately small JSON API."""

    server: TichuWebServer

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the server useful without printing every state poll.
        if not self.path.startswith(("/api/state", "/api/advance-ai")):
            super().log_message(format, *args)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise WebError("Invalid request body length.") from exc
        if length < 0:
            raise WebError("Invalid request body length.")
        if length > 32_768:
            raise WebError("Request body is too large.")
        try:
            value = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            raise WebError("Request body must be JSON.") from exc
        if not isinstance(value, dict):
            raise WebError("Request body must be a JSON object.")
        return value

    @staticmethod
    def _int_list(value: Any) -> list[int]:
        if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise WebError("Cards must be a list of card numbers.")
        if any(card < 0 or card >= NUM_CARDS for card in value):
            raise WebError("A card number was outside the Tichu deck.")
        return value

    def _dispatch(self, path: str, body: dict[str, Any]) -> None:
        game = self.server.game
        if path == "/api/mode":
            mode = body.get("mode")
            if mode not in ("play", "advisor"):
                raise WebError("Mode must be play or advisor.")
            game.choose_mode(mode)
        elif path == "/api/mode-menu":
            game.open_mode_menu()
        elif path == "/api/new":
            seed = body.get("seed")
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
                raise WebError("Seed must be a number.")
            game.new_round(seed)
        elif path == "/api/advisor/setup":
            cards = body.get("cards")
            if not isinstance(cards, str):
                raise WebError("Advisor cards must be text.")
            game.setup_advisor(cards)
        elif path == "/api/advisor/reset":
            game.reset_advisor()
        elif path == "/api/advisor/first-eight":
            cards = body.get("cards")
            if not isinstance(cards, str):
                raise WebError("First-eight cards must be text.")
            game.advisor_set_first_eight(cards)
        elif path == "/api/advisor/remaining":
            cards = body.get("cards")
            if not isinstance(cards, str):
                raise WebError("Remaining cards must be text.")
            game.advisor_set_remaining_six(
                cards,
                self._int_list(body.get("gt_callers")),
            )
        elif path == "/api/advisor/received":
            cards = body.get("cards")
            if not isinstance(cards, str):
                raise WebError("Received cards must be text.")
            game.advisor_receive_pass(cards)
        elif path == "/api/advisor/mahjong-holder":
            holder = body.get("holder")
            if isinstance(holder, bool) or not isinstance(holder, int):
                raise WebError("Mahjong holder must be a player number.")
            game.advisor_set_mahjong_holder(holder)
        elif path == "/api/grand-tichu":
            declare = body.get("declare")
            if not isinstance(declare, bool):
                raise WebError("Grand Tichu decision must be true or false.")
            game.declare_grand_tichu(declare)
        elif path == "/api/pass":
            game.submit_pass(self._int_list(body.get("cards")))
        elif path == "/api/play":
            call_tichu = body.get("call_tichu", False)
            if not isinstance(call_tichu, bool):
                raise WebError("call_tichu must be true or false.")
            game.play_cards(self._int_list(body.get("cards")), call_tichu)
        elif path == "/api/undo":
            game.undo_live()
        elif path == "/api/phoenix":
            phoenix_as = body.get("phoenix_as")
            if (
                isinstance(phoenix_as, bool)
                or not isinstance(phoenix_as, int)
                or not 1 <= phoenix_as <= 14
            ):
                raise WebError("Phoenix value must be a rank from 1 through Ace.")
            game.choose_phoenix(phoenix_as)
        elif path == "/api/phoenix/cancel":
            game.cancel_phoenix()
        elif path == "/api/wish":
            wish = body.get("wish")
            if wish is not None and (isinstance(wish, bool) or not isinstance(wish, int)):
                raise WebError("Wish must be a rank or null.")
            game.set_wish(wish)
        elif path == "/api/dragon":
            recipient = body.get("recipient")
            if isinstance(recipient, bool) or not isinstance(recipient, int):
                raise WebError("Dragon recipient must be a player number.")
            game.give_dragon(recipient)
        elif path == "/api/advisor/opponent":
            player = body.get("player")
            cards = body.get("cards")
            call_tichu = body.get("call_tichu", False)
            if isinstance(player, bool) or not isinstance(player, int) or not 0 <= player < 4:
                raise WebError("Player must be between P0 and P3.")
            if not isinstance(cards, str) or not isinstance(call_tichu, bool):
                raise WebError("Opponent card text or Tichu flag was invalid.")
            game.advisor_opponent_play(player, cards, call_tichu)
        elif path == "/api/advance-ai":
            action_id = body.get("action_id")
            if isinstance(action_id, bool) or not isinstance(action_id, int):
                raise WebError("AI reveal action id must be a number.")
            game.advance_ai_step(action_id)
        elif path == "/api/replay":
            game.start_ai_replay()
        elif path == "/api/replay/back":
            game.step_ai_replay(-1)
        elif path == "/api/replay/forward":
            game.step_ai_replay(1)
        elif path == "/api/replay/end":
            game.finish_ai_replay()
        elif path == "/api/shutdown":
            # shutdown() must run outside the request thread; it waits for
            # serve_forever() to finish before returning.
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            raise WebError("Unknown action.")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            with self.server.game_lock:
                self._send_json({"ok": True, "state": self.server.game.state()})
            return
        if path == "/":
            path = "/index.html"
        # Resolve the requested file below the static root, so /../ cannot
        # read files from the checkout.
        target = (WEB_ROOT / path.lstrip("/")).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # This is a localhost development UI.  Avoid pinning stale CSS or
        # card art in the browser while the visual table is being refined.
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            with self.server.game_lock:
                self._dispatch(path, body)
                state = self.server.game.state()
            self._send_json({"ok": True, "state": state})
        except WebError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local visual Tichu play and advisor UI")
    parser.add_argument("--mode", choices=("play", "advisor"), default="play")
    parser.add_argument("--menu", action="store_true",
                        help="Open with a visual Play/Advisor mode menu.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Play-model checkpoint. Omit with --random-ai.")
    parser.add_argument("--pass-checkpoint", default=None)
    parser.add_argument("--seat", choices=(0, 1, 2, 3), type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Local address to listen on (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--random-ai", action="store_true",
                        help="Run visual play without loading a model checkpoint.")
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("For safety, training.web only serves localhost. Use a local browser.")

    device = torch.device(args.device)
    policy = pass_policy = gt_policy = None
    kind = "history"
    if not args.random_ai:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file():
            raise SystemExit(
                f"Checkpoint not found: {checkpoint}\n"
                "Pass --checkpoint PATH, or use --random-ai for the visual demo."
            )
        policy, pass_policy, gt_policy, kind, info = load_play_bundle(
            str(checkpoint), args.pass_checkpoint, device
        )
        print(f"Loaded {checkpoint} ({kind}; {info['games_played']} games)")
    else:
        print("Running with random opponents; model recommendations are unavailable.")

    game = LocalGame(
        mode=args.mode,
        seat=args.seat,
        show_mode_menu=args.menu,
        policy=policy,
        pass_policy=pass_policy,
        gt_policy=gt_policy,
        kind=kind,
        device=device,
    )
    server = TichuWebServer((args.host, args.port), game)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(f"Tichu visual UI: {url}")
    print("Press Ctrl-C to stop the local server.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Tichu visual UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
