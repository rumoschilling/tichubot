"""Core Tichu game state and trick logic.

Manages:
  - 4 player hands, turn order, pass tracking
  - Trick play and comparison
  - Special card effects (Dragon, Dog, Mahjong wish)
  - Tichu/Grand Tichu declarations
  - Finish order and scoring

Players are indexed 0-3. Teams: (0, 2) vs (1, 3).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

from tichu_env.cards import (
    CARD_POINTS,
    DRAGON_INDEX,
    DOG_INDEX,
    MAHJONG_INDEX,
    NUM_CARDS,
    PHOENIX_INDEX,
    Deck,
    Rank,
    card_name,
    card_rank,
    card_suit,
)
from itertools import combinations as _combinations, product as _product
from collections import defaultdict as _defaultdict

from tichu_env.combinations import (
    PASS,
    Combination,
    CombinationType,
    beats,
    binary_phoenix_interpretations,
    classify_cards,
)


class GamePhase(IntEnum):
    """Phases of a Tichu round."""
    GRAND_TICHU = 0  # Players see 8 cards, decide on grand tichu
    PASSING = 1      # Each player picks 3 cards to pass (one per opponent)
    PLAY = 2         # Normal card play


def _action_sort_key(action: list[int]) -> tuple:
    """Canonical sort key for a legal action.

    Orders by (combo_size, max_rank, card_tuple) so that:
      - pass is always first (size 0)
      - singles before pairs before triples etc.
      - within same size, lower rank combos first
      - within same rank, deterministic tiebreak on card indices
    """
    if not action:
        return (0, 0, ())
    max_rank = 0
    for c in action:
        if c < 52:
            r = c % 13 + 2
        elif c == MAHJONG_INDEX:
            r = 1
        elif c == DRAGON_INDEX:
            r = 15
        elif c == DOG_INDEX:
            r = 0
        else:
            # Phoenix: doesn't define the combo rank
            continue
        if r > max_rank:
            max_rank = r
    return (len(action), max_rank, tuple(sorted(action)))


TEAM_A = (0, 2)
TEAM_B = (1, 3)


def teammate(player: int) -> int:
    return (player + 2) % 4


def team_of(player: int) -> tuple[int, int]:
    if player in TEAM_A:
        return TEAM_A
    return TEAM_B


def opponent_team(player: int) -> tuple[int, int]:
    if player in TEAM_A:
        return TEAM_B
    return TEAM_A


@dataclass
class TrickState:
    """State of the current active trick. 
    
    A trick ends when all other active players pass consecutively.
    """
    leading_combo: Combination | None = None  # The combo that others must beat
    leading_player: int = -1                 # Player who played the leading combo
    cards_on_table: list[int] = field(default_factory=list) # All cards played in this trick
    passes: int = 0                          # Number of consecutive passes
    required_passes: int = 3                 # Stale field, maintained for compat


@dataclass
class GameState:
    """Full state of one Tichu round. 
    
    Handles rules, turn advancement, special cards, and scoring.
    """

    # Hands: cards each player currently holds (sets for O(1) lookup)
    hands: list[set[int]] = field(default_factory=lambda: [set() for _ in range(4)])

    # Whose turn it is (0-3)
    current_player: int = 0

    # Players who have gone out (in order)
    finish_order: list[int] = field(default_factory=list)

    # Set mirror of finish_order for O(1) membership tests
    finished: set[int] = field(default_factory=set)

    # Trick collection: cards won by each player
    won_cards: list[list[int]] = field(default_factory=lambda: [[] for _ in range(4)])

    # Current trick
    trick: TrickState = field(default_factory=TrickState)

    # Tichu declarations: 0=none, 1=Tichu, 2=Grand Tichu
    tichu_declarations: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    # Which "brain" controls each seat — an annotation for observation
    # encoding, not a game rule. Seats with equal ids run the same policy
    # weights (default: pure self-play, all the same). Set e.g. [0, 1, 0, 1]
    # for team-vs-team evals or [0, 1, 2, 3] when seats are humans/foreign
    # models. reset() leaves this untouched: identity persists across rounds.
    player_model_ids: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    # Mahjong wish: if set, the rank value (2-14) that must be played if possible
    mahjong_wish: int | None = None

    # Dragon: when a trick is won with Dragon, winner must give trick to an opponent.
    # This is set to the player who must choose which opponent receives the trick.
    dragon_give_pending: int | None = None

    # Cards played so far (for observation: which cards are "out")
    played_cards: set[int] = field(default_factory=set)

    # History of completed tricks: list of (winner, cards, combo_type)
    trick_history: list[tuple[int, list[int], CombinationType]] = field(
        default_factory=list
    )

    # Whether to record per-play history (for sequence model encoding).
    # When False, only played_cards is tracked (cheaper static-only mode).
    track_history: bool = False

    # Chronological play log: (player, card_indices, trick_num, is_trick_opener)
    # Only populated when track_history=True.
    play_log: list[tuple[int, list[int], int, bool]] = field(default_factory=list)

    # Cached encoded tokens for chronological history (only objective bits 4 to 62).
    # Shape: (MAX_HISTORY_LEN, 59). Only used when track_history=True.
    # bits: 0-55 cards, 56 is_pass, 57 trick_idx, 58 is_opener.
    history_tokens: np.ndarray | None = None

    # Cached player indices for each play in the log.
    # Shape: (MAX_HISTORY_LEN,). Used for fast relative seat encoding.
    player_indices: np.ndarray | None = None

    # Current trick number (0-indexed, increments each new trick)
    trick_num: int = 0

    # Card passing phase: pending submissions {player: {recipient: card_index}}
    _pass_pending: dict[int, dict[int, int]] = field(default_factory=dict)
    # For each player, track which incoming passed card came from which source seat.
    _incoming_pass_source: list[dict[int, int]] = field(
        default_factory=lambda: [dict() for _ in range(4)]
    )
    # For each player, track which outgoing passed card went to which target seat.
    _outgoing_pass_target: list[dict[int, int]] = field(
        default_factory=lambda: [dict() for _ in range(4)]
    )
    _with_passing: bool = False

    # Is the round over?
    round_over: bool = False

    # Scores for this round (set when round ends)
    round_scores: tuple[int, int] = (0, 0)  # (team_A, team_B)

    # Game phase (calling vs play)
    phase: GamePhase = GamePhase.PLAY

    # Calling phase tracking
    _gt_decided: set[int] = field(default_factory=set)     # who decided on grand tichu
    _remaining_cards: list[list[int]] | None = None          # 6 held-back cards per player

    # Tichu calling during play: players who have played cards can no longer call
    _has_played: set[int] = field(default_factory=set)

    # RNG for reproducibility
    rng: random.Random = field(default_factory=random.Random)

    # Internal cache for sorted hands
    _sorted_hands_cache: list[list[int] | None] = field(default_factory=lambda: [None] * 4)

    @property
    def sorted_hands(self) -> list[list[int]]:
        """Return a list of sorted hand card indices for each player."""
        for i in range(4):
            if self._sorted_hands_cache[i] is None:
                self._sorted_hands_cache[i] = sorted(list(self.hands[i]))
        return self._sorted_hands_cache
    
    def _invalidate_cache(self, player: int | None = None):
        """Invalidate the sorted hand cache for one or all players."""
        if player is None:
            self._sorted_hands_cache = [None] * 4
        else:
            self._sorted_hands_cache[player] = None

    def reset(self, seed: int | None = None, with_calling: bool = False,
              with_passing: bool = False):
        """Reset and deal a new round.

        Args:
            seed: RNG seed for reproducibility.
            with_calling: If True, enable grand-tichu calling phase.
                Players receive 8 cards first (grand tichu decision), then
                the remaining 6, then passing/play begins.
            with_passing: If True, enable card passing phase.
                Each player passes one card to each opponent before play.
                Comes after grand tichu (if enabled), before play.
            Both default to False — backward-compatible with existing code.
        """
        if seed is not None:
            self.rng = random.Random(seed)

        deck = Deck.full()
        deck.shuffle(self.rng)

        self.finish_order = []
        self.finished = set()
        self.won_cards = [[] for _ in range(4)]
        self.trick = TrickState()
        self.tichu_declarations = [0, 0, 0, 0]
        self.mahjong_wish = None
        self.dragon_give_pending = None
        self.played_cards = set()
        self.trick_history = []
        self.play_log = []
        if self.track_history:
            from tichu_env.encoding import MAX_HISTORY_LEN
            self.history_tokens = np.zeros((MAX_HISTORY_LEN, 59), dtype=np.float32)
            self.player_indices = np.zeros(MAX_HISTORY_LEN, dtype=np.int8)
        else:
            self.history_tokens = None
            self.player_indices = None
        self.trick_num = 0
        self._pass_pending = {}
        self._incoming_pass_source = [dict() for _ in range(4)]
        self._outgoing_pass_target = [dict() for _ in range(4)]
        self._with_passing = with_passing
        self.round_over = False
        self.round_scores = (0, 0)

        self._has_played = set()

        if with_calling:
            # Deal 8 cards now, hold back 6 per player
            cards = deck.cards
            self.hands = [
                set(cards[i * 14 : i * 14 + 8]) for i in range(4)
            ]
            self._invalidate_cache()
            self._remaining_cards = [
                list(cards[i * 14 + 8 : i * 14 + 14]) for i in range(4)
            ]
            self._gt_decided = set()
            self.phase = GamePhase.GRAND_TICHU
            self.current_player = 0
        else:
            h0, h1, h2, h3 = deck.deal()
            self.hands = [set(h0), set(h1), set(h2), set(h3)]
            self._invalidate_cache()
            self._remaining_cards = None
            self._gt_decided = set()
            if with_passing:
                self.phase = GamePhase.PASSING
                self.current_player = 0
            else:
                self.phase = GamePhase.PLAY
                # Player with Mahjong starts
                for i in range(4):
                    if MAHJONG_INDEX in self.hands[i]:
                        self.current_player = i
                        break

    # ── Calling phase methods ────────────────────────────────────────────

    def declare_grand_tichu(self, player: int, declare: bool) -> bool:
        """Decide on grand tichu during GRAND_TICHU phase.

        Each player calls this once (in turn order 0→1→2→3).
        After all four decide, the remaining 6 cards are dealt and
        play begins.

        Returns True if the declaration was accepted.
        """
        if self.phase != GamePhase.GRAND_TICHU:
            return False
        if player != self.current_player:
            return False
        if player in self._gt_decided:
            return False

        if declare:
            self.tichu_declarations[player] = 2

        self._gt_decided.add(player)

        if len(self._gt_decided) == 4:
            # Deal remaining 6 cards to each player
            for p in range(4):
                self.hands[p].update(self._remaining_cards[p])
            self._invalidate_cache()
            self._remaining_cards = None
            if self._with_passing:
                self.phase = GamePhase.PASSING
                self.current_player = 0
            else:
                self._start_play()
        else:
            self.current_player = (player + 1) % 4

        return True

    def can_call_tichu(self, player: int) -> bool:
        """Check if a player is still eligible to call tichu."""
        return (
            self.phase == GamePhase.PLAY
            and self.tichu_declarations[player] == 0
            and player not in self._has_played
        )

    # ── Passing phase methods ─────────────────────────────────────────

    def get_passing_hand(self, player: int) -> list[int]:
        """Return the player's hand as a sorted list (for the passing network).

        Only valid during PASSING phase for the current player.
        """
        assert self.phase == GamePhase.PASSING
        return list(self.sorted_hands[player])

    def pass_cards(self, player: int, left: int, partner: int, right: int) -> bool:
        """Submit cards to pass to opponents.

        Args:
            player: The player submitting.
            left: Card index to pass to the player on the left ((player+1)%4).
            partner: Card index to pass to partner ((player+2)%4).
            right: Card index to pass to the player on the right ((player+3)%4).

        Sequential: players 0→1→2→3 submit. After all 4, cards are
        exchanged simultaneously and phase transitions to PLAY.

        Returns True if accepted.
        """
        if self.phase != GamePhase.PASSING:
            return False
        if player != self.current_player:
            return False
        if player in self._pass_pending:
            return False

        cards = {left, partner, right}
        if len(cards) != 3:
            return False  # must be 3 distinct cards
        if not cards.issubset(self.hands[player]):
            return False

        left_p = (player + 1) % 4
        partner_p = (player + 2) % 4
        right_p = (player + 3) % 4
        self._pass_pending[player] = {left_p: left, partner_p: partner, right_p: right}

        if len(self._pass_pending) == 4:
            self._execute_passing()
        else:
            self.current_player = (player + 1) % 4

        return True

    def _execute_passing(self):
        """Swap all pending cards simultaneously, then start play."""
        # Collect incoming cards per player
        incoming: dict[int, list[int]] = {p: [] for p in range(4)}
        incoming_source: dict[int, dict[int, int]] = {p: {} for p in range(4)}
        outgoing_target: dict[int, dict[int, int]] = {p: {} for p in range(4)}
        for giver, recipients in self._pass_pending.items():
            for recipient, card in recipients.items():
                self.hands[giver].remove(card)
                incoming[recipient].append(card)
                incoming_source[recipient][giver] = card
                outgoing_target[giver][recipient] = card

        for p, cards in incoming.items():
            self.hands[p].update(cards)
        
        self._invalidate_cache()
        for p in range(4):
            self._incoming_pass_source[p] = incoming_source[p]
            self._outgoing_pass_target[p] = outgoing_target[p]

        self._pass_pending = {}
        self._start_play()

    def _start_play(self):
        """Transition from calling/passing phases to play."""
        self.phase = GamePhase.PLAY
        for i in range(4):
            if MAHJONG_INDEX in self.hands[i]:
                self.current_player = i
                return

    @property
    def active_players(self) -> list[int]:
        """Players still in the round (have cards or haven't finished)."""
        return [p for p in range(4) if p not in self.finished]

    def is_player_active(self, player: int) -> bool:
        return player not in self.finished

    def _next_active_player(self, after: int) -> int:
        """Next player (clockwise) who is still active."""
        p = (after + 1) % 4
        while p in self.finished:
            p = (p + 1) % 4
            if p == after:
                break  # shouldn't happen in normal play
        return p

    def _record_play(self, player: int, card_indices: list[int], trick_num: int, is_opener: bool):
        """Internal helper to record a play and update its cached token."""
        idx = len(self.play_log)
        self.play_log.append((player, list(card_indices), trick_num, is_opener))
        
        if self.track_history and self.history_tokens is not None:
            if idx < len(self.history_tokens):
                self.player_indices[idx] = player
                token = self.history_tokens[idx]
                token.fill(0)  # CLEAR before filling to prevent bit leakage
                # token[0:4] is for relative player, set during encode_history
                # token[4:60] -> objective bits (offset by -4 here)
                
                # Cards (56 binary)
                for c in card_indices:
                    token[c] = 1.0
                
                # Is pass (bit 60 in full token, so 56 here)
                if not card_indices:
                    token[56] = 1.0
                
                # Trick index (bit 61 in full token, so 57 here)
                token[57] = trick_num / 25.0
                
                # Is opener (bit 62 in full token, so 58 here)
                if is_opener:
                    token[58] = 1.0

    def _award_trick(self, winner: int):
        """Award current trick's cards to the winner."""
        self.won_cards[winner].extend(self.trick.cards_on_table)
        combo_type = (
            self.trick.leading_combo.type
            if self.trick.leading_combo
            else CombinationType.PASS
        )
        self.trick_history.append((winner, list(self.trick.cards_on_table), combo_type))
        self.trick = TrickState()
        if self.track_history:
            self.trick_num += 1

    def play(
        self,
        player: int,
        action,
        call_tichu: bool = False,
        *,
        phoenix_as: int | None = None,
    ) -> bool:
        """Execute a play, optionally calling tichu at the same time.

        action: list of card indices (or [] for pass), a tuple
            (card_indices, call_tichu) from get_legal_actions(), or a tuple
            (card_indices, call_tichu, phoenix_high) from
            get_policy_legal_actions().
        call_tichu: declare tichu alongside this action (ignored when
            action is a tuple — the bool from the tuple is used instead).
        """
        if self.phase != GamePhase.PLAY:
            return False

        # Unpack tuple forms from get_legal_actions/get_policy_legal_actions.
        phoenix_high: bool | None = None
        if isinstance(action, tuple):
            if len(action) not in (2, 3):
                return False
            if len(action) == 3:
                card_indices, call_tichu, phoenix_high = action
            else:
                card_indices, call_tichu = action
        else:
            card_indices = action

        if self.round_over:
            return False
        if player != self.current_player:
            return False
        if self.dragon_give_pending is not None:
            return False

        if not isinstance(call_tichu, (bool, np.bool_)):
            return False
        call_tichu = bool(call_tichu)
        if call_tichu and not self.can_call_tichu(player):
            return False
        if phoenix_high is not None and not isinstance(
            phoenix_high, (bool, np.bool_)
        ):
            return False

        if not isinstance(card_indices, (list, tuple)):
            return False
        if any(
            isinstance(card, (bool, np.bool_))
            or not isinstance(card, (int, np.integer))
            or not 0 <= int(card) < NUM_CARDS
            for card in card_indices
        ):
            return False
        card_indices = [int(card) for card in card_indices]
        if len(card_indices) != len(set(card_indices)):
            return False

        # Handle pass only after checking whether an active Mahjong wish
        # requires this player to make a card play.
        if len(card_indices) == 0:
            if self.trick.leading_combo is None:
                return False
            if (
                self.mahjong_wish is not None
                and not self._check_wish_compliance(player, PASS)
            ):
                return False
            if call_tichu:
                self.tichu_declarations[player] = 1
            return self._do_pass(player)

        # Validate player has these cards
        card_set = set(card_indices)
        if not card_set.issubset(self.hands[player]):
            return False

        if phoenix_as is None and phoenix_high is not None:
            interpretations = binary_phoenix_interpretations(card_indices)
            if interpretations is not None:
                chosen = interpretations[1 if phoenix_high else 0]
                phoenix_as = chosen.phoenix_as

        # Classify the combination
        combo = classify_cards(card_indices, phoenix_as=phoenix_as)
        if combo is None:
            return False

        # Dog: can only be played when leading, and cannot bypass an
        # active Mahjong wish that the player is able to fulfil.
        if combo.type == CombinationType.DOG:
            if self.trick.leading_combo is not None:
                return False
            if (
                self.mahjong_wish is not None
                and not self._check_wish_compliance(player, combo)
            ):
                return False
            if call_tichu:
                self.tichu_declarations[player] = 1
            return self._play_dog(player)

        is_leading = self.trick.leading_combo is None

        # Phoenix as single: adjust rank to be 0.5 above current
        # (must happen before beats check so Phoenix gets its actual rank)
        if (
            combo.type == CombinationType.SINGLE
            and PHOENIX_INDEX in card_set
            and not is_leading
        ):
            # Phoenix cannot beat Dragon
            if self.trick.leading_combo.rank >= 15:
                return False
            combo = Combination(
                type=CombinationType.SINGLE,
                cards=combo.cards,
                rank=self.trick.leading_combo.rank + 0.5,
                length=1,
            )

        # Check if this beats the current trick
        if not is_leading:
            if not beats(combo, self.trick.leading_combo):
                return False

        # Check Mahjong wish compliance
        if self.mahjong_wish is not None:
            if not self._check_wish_compliance(player, combo):
                return False

        # Execute the play
        if call_tichu:
            self.tichu_declarations[player] = 1
        self._has_played.add(player)
        self._remove_cards(player, card_set)
        self.trick.leading_combo = combo
        self.trick.leading_player = player
        self.trick.cards_on_table.extend(card_indices)
        self.trick.passes = 0
        
        if is_leading:
            # Set required passes once at the start of the trick.
            # Everyone else who is currently active must pass.
            self.trick.required_passes = 4 - len(self.finish_order) - 1
        
        if self.track_history:
            self._record_play(player, list(card_indices), self.trick_num, is_leading)

        # Check if player went out
        if len(self.hands[player]) == 0:
            self.finish_order.append(player)
            self.finished.add(player)

        if self._check_round_over():
            return True

        # Advance player (skip those who are finished)
        self.current_player = self._next_active_player(player)

        # If all other active players have passed, trick is won.
        # Note: _check_trick_won might change current_player again if the trick ends.
        self._check_trick_won()

        return True

    def _do_pass(self, player: int) -> bool:
        """Handle a pass."""
        # Can't pass when leading (except if no cards, but that shouldn't happen)
        if self.trick.leading_combo is None:
            return False

        self.trick.passes += 1
        if self.track_history:
            self._record_play(player, [], self.trick_num, False)

        # Update player BEFORE checking if trick is won
        self.current_player = self._next_active_player(player)
        self._check_trick_won()
        return True

    def _play_dog(self, player: int):
        """Dog passes the lead to the player's teammate."""
        self._has_played.add(player)
        if self.track_history:
            self._record_play(player, [DOG_INDEX], self.trick_num, True)
        self._remove_cards(player, {DOG_INDEX})

        if len(self.hands[player]) == 0:
            self.finish_order.append(player)
            self.finished.add(player)

        # Dog has 0 points, but assign the physical card before round scoring
        # so a final Dog remains accounted for even when the round ends here.
        tm = teammate(player)
        self.won_cards[tm].append(DOG_INDEX)

        if self._check_round_over():
            return True

        # Lead passes to teammate
        if self.is_player_active(tm):
            self.current_player = tm
        else:
            # Teammate already out, lead goes to next active player after teammate
            self.current_player = self._next_active_player(tm)

        self.trick = TrickState()
        self.trick_history.append((tm, [DOG_INDEX], CombinationType.DOG))
        if self.track_history:
            self.trick_num += 1
        return True

    def _check_trick_won(self):
        """Check if the current trick has been won (all others passed)."""
        if self.trick.leading_combo is None:
            return

        # num_active is the number of players who still have cards
        num_active = 4 - len(self.finished)

        # Trick is won when all other active players have passed. If the
        # leading player went out with their play, they are no longer active,
        # so ALL num_active remaining players must pass (not num_active - 1).
        if self.trick.leading_player in self.finished:
            passes_needed = num_active
        else:
            passes_needed = num_active - 1

        if self.trick.passes >= passes_needed:
            winner = self.trick.leading_player

            # If winner already went out, they still win the trick
            # Only a winning Dragon requires a gift. A bomb that captured
            # the Dragon keeps the trick and its points.
            if self.trick.leading_combo.cards == (DRAGON_INDEX,):
                self.dragon_give_pending = winner
                # We still need to advance current_player so the system knows
                # WHO is supposed to make the give decision.
                self.current_player = winner
                return

            self._award_trick(winner)

            # Next lead
            if self.is_player_active(winner):
                self.current_player = winner
            else:
                self.current_player = self._next_active_player(winner)

            self._check_round_over()

    def give_dragon_trick(self, player: int, recipient: int) -> bool:
        """After winning a trick with Dragon, give it to an opponent.

        player: the player who won the trick (must match dragon_give_pending)
        recipient: the opponent to give the trick to (must be on opposing team)
        """
        if self.dragon_give_pending != player:
            return False
        if (
            isinstance(recipient, (bool, np.bool_))
            or not isinstance(recipient, (int, np.integer))
        ):
            return False
        recipient = int(recipient)
        if recipient not in opponent_team(player):
            return False

        self.won_cards[recipient].extend(self.trick.cards_on_table)
        combo_type = (
            self.trick.leading_combo.type
            if self.trick.leading_combo
            else CombinationType.SINGLE
        )
        self.trick_history.append(
            (recipient, list(self.trick.cards_on_table), combo_type)
        )
        self.trick = TrickState()
        self.dragon_give_pending = None
        if self.track_history:
            self.trick_num += 1

        # Next lead goes to the winner (or next active)
        if self.is_player_active(player):
            self.current_player = player
        else:
            self.current_player = self._next_active_player(player)

        self._check_round_over()
        return True

    def set_mahjong_wish(self, rank: int | None):
        """Set the Mahjong wish (rank 2-14, or None for no wish)."""
        if rank is not None and (rank < 2 or rank > 14):
            return
        self.mahjong_wish = rank

    def _check_wish_compliance(self, player: int, combo: Combination) -> bool:
        """Check Mahjong wish: if player can play the wished rank, they should.

        The wish only requires you to play the wished rank if you CAN play it
        in a legal combination that beats the current trick. If you can't, you
        may play anything.

        Returns True if the play is legal w.r.t. the wish.
        """
        if self.mahjong_wish is None:
            return True

        wished_rank = self.mahjong_wish

        # If the played combo contains the wished rank, wish is satisfied
        for c in combo.cards:
            if c < 52 and c % 13 + 2 == wished_rank:
                self._fulfill_wish()
                return True

        # If player doesn't have any card of wished rank, any play is fine
        has_wished = any(c < 52 and c % 13 + 2 == wished_rank for c in self.hands[player])

        if not has_wished:
            return True

        # Player has the wished rank but didn't play it — check if they COULD
        # play it in a legal combo that beats the current trick
        if self._can_play_wished_rank(player, wished_rank):
            return False  # Must play the wished rank

        return True  # Can't incorporate wished rank into a beating combo

    def _can_play_wished_rank(self, player: int, wished_rank: int) -> bool:
        """Check if player can make any legal combo containing the wished rank."""
        hand = sorted(self.hands[player])
        if self.trick.leading_combo is None:
            combos = self._enumerate_leading_combos(hand)
        else:
            combos = self._enumerate_beating_combos(hand, player)
        return any(
            any(c < 52 and c % 13 + 2 == wished_rank for c in combo)
            for combo in combos
        )

    def _fulfill_wish(self):
        """Clear the Mahjong wish once fulfilled."""
        self.mahjong_wish = None

    def _remove_cards(self, player: int, cards: set[int]):
        self.hands[player] -= cards
        self.played_cards |= cards
        self._invalidate_cache(player)

    def _check_round_over(self) -> bool:
        """Check if the round has ended according to Tichu rules.
        
        Round ends if:
          - Teammates finish 1st and 2nd (Double Win: 200 pts)
          - 3 players have finished (Normal finish: scoring based on won cards)
        """
        if self.round_over:
            return True

        n_finished = len(self.finished)

        # 1. 1-2 finish: teammates finish first and second
        if len(self.finish_order) >= 2:
            first, second = self.finish_order[0], self.finish_order[1]
            if second in team_of(first) and second != first:
                self.round_over = True
                self._score_double_win(first)
                return True

        # 2. 3 players out: round over, final player cleanup
        if n_finished >= 3:
            self.round_over = True
            # The one remaining player is the 'last' player
            last = next(p for p in range(4) if p not in self.finished)
            first_out = self.finish_order[0]
            opps = opponent_team(last)

            # Cleanup logic per Tichu rules:
            # 1. Last player's HAND cards go to the OPPOSING TEAM
            remaining_hand = list(self.hands[last])
            self.won_cards[opps[0]].extend(remaining_hand)
            self.played_cards |= self.hands[last]
            self.hands[last] = set()
            self._invalidate_cache(last)

            # 2. Last player's WON cards go to the FIRST PLAYER OUT
            self.won_cards[first_out].extend(self.won_cards[last])
            self.won_cards[last] = []

            # 3. Any pending trick cards on the table go to the LAST TRICK WINNER
            if self.trick.cards_on_table:
                winner = self.trick.leading_player
                if winner != -1:
                    # A winning Dragon is given away; a Dragon beaten by
                    # the final bomb belongs to the bomb winner.
                    if (
                        self.trick.leading_combo is not None
                        and self.trick.leading_combo.cards == (DRAGON_INDEX,)
                    ):
                        self.won_cards[opponent_team(winner)[0]].extend(self.trick.cards_on_table)
                    else:
                        self.won_cards[winner].extend(self.trick.cards_on_table)
                self.trick = TrickState()

            # Mark last player as finished too
            self.finish_order.append(last)
            self.finished.add(last)

            self._score_normal()
            return True

        return False

    def _tichu_bonuses(self) -> tuple[int, int]:
        """Compute tichu/grand tichu bonus for each team.

        +100/-100 for tichu, +200/-200 for grand tichu,
        based on whether the declaring player finished first.
        """
        a_bonus, b_bonus = 0, 0
        first = self.finish_order[0]
        for p in range(4):
            if self.tichu_declarations[p] == 1:
                bonus = 100 if first == p else -100
            elif self.tichu_declarations[p] == 2:
                bonus = 200 if first == p else -200
            else:
                continue
            if p in TEAM_A:
                a_bonus += bonus
            else:
                b_bonus += bonus
        return a_bonus, b_bonus

    def _score_double_win(self, first_player: int):
        """Score a 1-2 finish: winning team gets 200, losing team 0, plus tichu bonuses."""
        if first_player in TEAM_A:
            base_a, base_b = 200, 0
        else:
            base_a, base_b = 0, 200
        bonus_a, bonus_b = self._tichu_bonuses()
        self.round_scores = (base_a + bonus_a, base_b + bonus_b)

    def _score_normal(self):
        """Normal scoring: count card points for each team, plus tichu bonuses."""
        team_a_points = sum(
            sum(CARD_POINTS.get(c, 0) for c in self.won_cards[p])
            for p in TEAM_A
        )
        team_b_points = sum(
            sum(CARD_POINTS.get(c, 0) for c in self.won_cards[p])
            for p in TEAM_B
        )

        # Sanity check: total points should always be 100 (if no errors in card tracking)
        # 5s(5*4=20), 10s(10*4=40), Ks(10*4=40), Phoenix(-25), Dragon(25) = 100
        total = team_a_points + team_b_points
        if total != 100:
            # This shouldn't happen with correct logic, but we log it if it does
            pass

        bonus_a, bonus_b = self._tichu_bonuses()
        self.round_scores = (int(team_a_points + bonus_a), int(team_b_points + bonus_b))

    def get_legal_actions(self, player: int) -> list[tuple[list[int], bool]]:
        """Return all legal plays for a player.

        Each action is a (card_indices, call_tichu) tuple that can be
        passed directly to play().  When tichu is callable, every base
        action appears twice — once without and once with tichu.

        The list is sorted by (combo_size, rank, cards, call_tichu).
        """
        if (
            self.phase != GamePhase.PLAY
            or player != self.current_player
            or self.round_over
        ):
            return []

        if self.dragon_give_pending is not None:
            return []  # Must call give_dragon_trick instead

        hand = list(self.sorted_hands[player])
        is_leading = self.trick.leading_combo is None
        base: list[list[int]] = []

        if is_leading:
            base = self._enumerate_leading_combos(hand)

            if self.mahjong_wish is not None and base:
                wished = self.mahjong_wish
                with_wish = [
                    combo for combo in base
                    if any(c < 52 and c % 13 + 2 == wished for c in combo)
                ]
                if with_wish:
                    base = with_wish
        else:
            beating = self._enumerate_beating_combos(hand, player)

            must_play_wish = False
            if self.mahjong_wish is not None and beating:
                wished = self.mahjong_wish
                # A player is ONLY forced to fulfill the wish if they have a
                # combo that satisfies the wish AND beats the current trick.
                with_wish = [
                    combo for combo in beating
                    if any(c < 52 and c % 13 + 2 == wished for c in combo)
                ]
                if with_wish:
                    beating = with_wish
                    must_play_wish = True

            if not must_play_wish:
                base.append([])

            base.extend(beating)

        base.sort(key=_action_sort_key)

        # Build final action list with tichu variants
        actions: list[tuple[list[int], bool]] = [(a, False) for a in base]
        if self.can_call_tichu(player):
            actions.extend((a, True) for a in base)

        return actions

    def get_policy_legal_actions(
        self, player: int
    ) -> list[tuple[list[int], bool, bool]]:
        """Return learned-policy actions with an explicit Phoenix choice bit.

        The third item is ``phoenix_high``. It is false for ordinary and
        unambiguous actions. Ambiguous straights/full houses appear twice:
        lower first, higher second. Physical duplicates from the combination
        enumerator are collapsed before adding these semantic variants.
        """
        current = self.trick.leading_combo
        actions: list[tuple[list[int], bool, bool]] = []
        seen: set[tuple[tuple[int, ...], bool]] = set()
        for cards, call_tichu in self.get_legal_actions(player):
            canonical_cards = tuple(sorted(cards))
            key = (canonical_cards, call_tichu)
            if key in seen:
                continue
            seen.add(key)

            interpretations = binary_phoenix_interpretations(canonical_cards)
            if interpretations is None:
                actions.append((list(canonical_cards), call_tichu, False))
                continue

            for phoenix_high, combo in enumerate(interpretations):
                if current is None or beats(combo, current):
                    actions.append(
                        (list(canonical_cards), call_tichu, bool(phoenix_high))
                    )
        return actions

    def _enumerate_leading_combos(self, hand: list[int]) -> list[list[int]]:
        """Enumerate all valid combinations from hand (for leading)."""
        return _enumerate_combos(hand)

    def _enumerate_beating_combos(
        self, hand: list[int], player: int
    ) -> list[list[int]]:
        """Enumerate combos from hand that beat the current trick."""
        trick = self.trick.leading_combo
        if trick is None:
            return []

        combos = []
        target_rank = trick.rank
        ha = _hand_analysis(hand)

        # Always collect bombs (they beat anything)
        bombs = _enum_four_bombs(ha) + _enum_straight_bombs(ha)

        if trick.is_bomb:
            # Only higher bombs beat a bomb — compare inline without classify_cards.
            # Four bombs are exactly 4 cards; straight bombs are 5+.
            trick_is_straight_bomb = trick.type == CombinationType.STRAIGHT_BOMB
            for b in bombs:
                if len(b) == 4:
                    # Four bomb: beats only a weaker four bomb
                    b_rank = b[0] % 13 + 2
                    if not trick_is_straight_bomb and b_rank > trick.rank:
                        combos.append(b)
                else:
                    # Straight bomb: beats any four bomb; beats shorter/lower straight bomb
                    b_rank = b[-1] % 13 + 2
                    b_len = len(b)
                    if trick_is_straight_bomb:
                        if b_len > trick.length or (b_len == trick.length and b_rank > trick.rank):
                            combos.append(b)
                    else:
                        combos.append(b)  # straight bomb always beats four bomb
            return combos

        # Non-bomb trick: same type + higher rank, OR any bomb
        combos.extend(bombs)

        if trick.type == CombinationType.SINGLE:
            # Singles above target rank
            for c in hand:
                if c >= 52:
                    if c == DRAGON_INDEX:
                        combos.append([c])
                    elif c == PHOENIX_INDEX and target_rank < 15:
                        combos.append([c])
                    # DOG and MAHJONG (rank 1) can't beat anything
                    continue
                if c % 13 + 2 > target_rank:
                    combos.append([c])

        elif trick.type == CombinationType.PAIR:
            for rank, cards in ha.rank_cards.items():
                if rank <= target_rank:
                    continue
                for pair in _combinations(cards, 2):
                    combos.append(list(pair))
                if ha.has_phoenix:
                    for card in cards:
                        combos.append([card, PHOENIX_INDEX])
            # Phoenix can also pair with a card AT a rank above target
            # (already handled above since we iterate rank > target_rank)

        elif trick.type == CombinationType.TRIPLE:
            for rank, cards in ha.rank_cards.items():
                if rank <= target_rank:
                    continue
                for trip in _combinations(cards, 3):
                    combos.append(list(trip))
                if ha.has_phoenix and len(cards) >= 2:
                    for pair in _combinations(cards, 2):
                        combos.append(list(pair) + [PHOENIX_INDEX])

        elif trick.type == CombinationType.FULL_HOUSE:
            # FH rank = triple rank; pass min_rank to skip low-rank combos
            combos.extend(_enum_full_houses(ha, min_rank=int(target_rank)))

        elif trick.type == CombinationType.STRAIGHT:
            # Straight rank = highest card = start + length - 1; filter in enumerator
            combos.extend(_enum_straights(ha, trick.length, min_rank=int(target_rank)))

        elif trick.type == CombinationType.STEPS:
            # Steps rank = highest pair rank; filter in enumerator
            combos.extend(_enum_steps(ha, trick.length, min_rank=int(target_rank)))

        return combos

    def get_dragon_give_options(self, player: int) -> list[int]:
        """Return which opponents the player can give the Dragon trick to."""
        if self.dragon_give_pending != player:
            return []
        opps = opponent_team(player)
        return list(opps)


# ── Fast combo enumeration ──────────────────────────────────────────


class _HandInfo:
    """Pre-analysed hand for fast enumeration."""
    __slots__ = ('rank_cards', 'suit_ranks', 'has_phoenix', 'has_dragon',
                 'has_dog', 'has_mahjong', 'hand')

    def __init__(self):
        self.rank_cards: dict[int, list[int]] = {}   # rank -> [card indices]
        self.suit_ranks: dict[int, list[tuple[int, int]]] = {}  # suit -> [(rank, card)]
        self.has_phoenix = False
        self.has_dragon = False
        self.has_dog = False
        self.has_mahjong = False
        self.hand: list[int] = []


def _hand_analysis(hand: list[int]) -> _HandInfo:
    ha = _HandInfo()
    ha.hand = hand
    rc: dict[int, list[int]] = _defaultdict(list)
    sr: dict[int, list[tuple[int, int]]] = _defaultdict(list)

    for c in hand:
        if c >= 52:
            if c == PHOENIX_INDEX:
                ha.has_phoenix = True
            elif c == DRAGON_INDEX:
                ha.has_dragon = True
            elif c == DOG_INDEX:
                ha.has_dog = True
            else:  # MAHJONG
                ha.has_mahjong = True
                rc[1].append(c)
        else:
            r = c % 13 + 2
            s = c // 13
            rc[r].append(c)
            sr[s].append((r, c))

    ha.rank_cards = dict(rc)
    ha.suit_ranks = {s: sorted(v) for s, v in sr.items()}
    return ha


def _enumerate_combos(hand: list[int]) -> list[list[int]]:
    """Enumerate all valid combinations from a hand (for leading)."""
    ha = _hand_analysis(hand)
    combos: list[list[int]] = []

    # Singles
    for c in hand:
        if c == DOG_INDEX:
            combos.append([DOG_INDEX])
        else:
            combos.append([c])

    # Pairs
    for rank, cards in ha.rank_cards.items():
        if rank == 1:
            continue  # Mahjong can't pair
        for pair in _combinations(cards, 2):
            combos.append(list(pair))
        if ha.has_phoenix:
            for card in cards:
                combos.append([card, PHOENIX_INDEX])

    # Triples
    for rank, cards in ha.rank_cards.items():
        if rank == 1:
            continue
        for trip in _combinations(cards, 3):
            combos.append(list(trip))
        if ha.has_phoenix and len(cards) >= 2:
            for pair in _combinations(cards, 2):
                combos.append(list(pair) + [PHOENIX_INDEX])

    # Four-of-a-kind bombs
    combos.extend(_enum_four_bombs(ha))

    # Full houses
    combos.extend(_enum_full_houses(ha))

    # Straights (length 5+)
    combos.extend(_enum_straights(ha))

    # Steps (consecutive pairs, 4+ cards)
    combos.extend(_enum_steps(ha))

    # Straight flush bombs
    combos.extend(_enum_straight_bombs(ha))

    return combos


def _enum_four_bombs(ha: _HandInfo) -> list[list[int]]:
    return [cards for cards in ha.rank_cards.values() if len(cards) == 4]


def _enum_straight_bombs(ha: _HandInfo) -> list[list[int]]:
    combos: list[list[int]] = []
    for suit_cards in ha.suit_ranks.values():
        # suit_cards already sorted by rank from _hand_analysis
        ranks = [r for r, _ in suit_cards]
        card_at = {r: c for r, c in suit_cards}

        # Find all consecutive runs of length >= 5
        for start_i in range(len(ranks)):
            run = [ranks[start_i]]
            for j in range(start_i + 1, len(ranks)):
                if ranks[j] == run[-1] + 1:
                    run.append(ranks[j])
                    if len(run) >= 5:
                        combos.append([card_at[r] for r in run])
                else:
                    break
    return combos


def _enum_full_houses(ha: _HandInfo, min_rank: int = 0) -> list[list[int]]:
    """Enumerate full houses. min_rank skips triples at or below that rank."""
    combos: list[list[int]] = []

    # Without phoenix: triple + pair
    for t_rank, t_cards in ha.rank_cards.items():
        if t_rank == 1 or t_rank <= min_rank or len(t_cards) < 3:
            continue
        for triple in _combinations(t_cards, 3):
            for p_rank, p_cards in ha.rank_cards.items():
                if p_rank == t_rank or p_rank == 1 or len(p_cards) < 2:
                    continue
                for pair in _combinations(p_cards, 2):
                    combos.append(list(triple) + list(pair))

    if ha.has_phoenix:
        # Phoenix completes pair→triple
        for t_rank, t_cards in ha.rank_cards.items():
            if t_rank == 1 or t_rank <= min_rank or len(t_cards) < 2:
                continue
            for pair_as_triple in _combinations(t_cards, 2):
                for p_rank, p_cards in ha.rank_cards.items():
                    if p_rank == t_rank or p_rank == 1 or len(p_cards) < 2:
                        continue
                    for pair in _combinations(p_cards, 2):
                        combos.append(list(pair_as_triple) + [PHOENIX_INDEX] + list(pair))

        # Phoenix completes single→pair
        for t_rank, t_cards in ha.rank_cards.items():
            if t_rank == 1 or t_rank <= min_rank or len(t_cards) < 3:
                continue
            for triple in _combinations(t_cards, 3):
                for p_rank, p_cards in ha.rank_cards.items():
                    if p_rank == t_rank or p_rank == 1:
                        continue
                    for single in p_cards:
                        combos.append(list(triple) + [single, PHOENIX_INDEX])

    return combos


def _enum_straights(
    ha: _HandInfo,
    required_length: int | None = None,
    min_rank: int = 0,
) -> list[list[int]]:
    """Enumerate straights. If required_length is set, only that length.
    min_rank skips straights whose highest card rank is at or below min_rank."""
    combos: list[list[int]] = []
    start_rank = 1 if ha.has_mahjong else 2
    max_rank = 14

    lengths = [required_length] if required_length else range(5, 15)

    for length in lengths:
        for start in range(start_rank, max_rank - length + 2):
            end = start + length - 1
            if end > max_rank:
                break
            if end <= min_rank:
                continue

            ranks_needed = list(range(start, end + 1))

            if all(r in ha.rank_cards for r in ranks_needed):
                # Natural straight: pick one card per rank.
                choices = [ha.rank_cards[r] for r in ranks_needed]
                for combo in _product(*choices):
                    combos.append(list(combo))

            if not ha.has_phoenix:
                continue

            # Phoenix may replace any selected normal rank, including one for
            # which another physical card remains in hand.  It cannot replace
            # rank 1: a straight starting at 1 requires the actual Mahjong.
            for phoenix_rank in ranks_needed:
                if phoenix_rank == 1:
                    continue
                physical_ranks = [r for r in ranks_needed if r != phoenix_rank]
                if not all(r in ha.rank_cards for r in physical_ranks):
                    continue
                choices = [ha.rank_cards[r] for r in physical_ranks]
                for combo in _product(*choices):
                    combos.append(list(combo) + [PHOENIX_INDEX])

    return combos


def _enum_steps(
    ha: _HandInfo,
    required_length: int | None = None,
    min_rank: int = 0,
) -> list[list[int]]:
    """Enumerate steps (consecutive pairs). required_length = total cards (even).
    min_rank skips steps whose highest pair rank is at or below min_rank."""
    combos: list[list[int]] = []
    if required_length is not None:
        num_pairs_list = [required_length // 2]
    else:
        max_pairs = min(13, (len(ha.hand) + 1) // 2)
        num_pairs_list = range(2, max_pairs + 1)

    for num_pairs in num_pairs_list:
        # Slide a window of consecutive normal ranks.  At least one physical
        # card is required at every rank because one Phoenix cannot make a pair
        # by itself.
        for start in range(2, 14 - num_pairs + 2):
            selected = list(range(start, start + num_pairs))
            if selected[-1] <= min_rank:
                continue
            if not all(rk in ha.rank_cards for rk in selected):
                continue

            if all(len(ha.rank_cards[rk]) >= 2 for rk in selected):
                choices = [
                    list(_combinations(ha.rank_cards[rk], 2))
                    for rk in selected
                ]
                for parts in _product(*choices):
                    combos.append([card for pair in parts for card in pair])

            if not ha.has_phoenix:
                continue

            # Phoenix may complete the selected pair even when another card of
            # that rank remains in hand.
            for phoenix_rank in selected:
                other_ranks = [rk for rk in selected if rk != phoenix_rank]
                if not all(len(ha.rank_cards[rk]) >= 2 for rk in other_ranks):
                    continue
                choices = []
                for rk in selected:
                    if rk == phoenix_rank:
                        choices.append(
                            [(card, PHOENIX_INDEX) for card in ha.rank_cards[rk]]
                        )
                    else:
                        choices.append(
                            list(_combinations(ha.rank_cards[rk], 2))
                        )
                for parts in _product(*choices):
                    combos.append([card for pair in parts for card in pair])

    return combos
