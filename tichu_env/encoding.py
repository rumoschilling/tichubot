"""Observation encoding for the Tichu environment.

Two modes controlled by track_history on the GameState:

=== Static-only mode (track_history=False) ===

  Cheap flat vector. Played cards are tracked as a binary mask but
  no per-play history is recorded. Static size: STATIC_OBS_SIZE = 196.

=== Full mode (track_history=True) ===

  Static features + a variable-length sequence of play tokens.
  The network should process these with separate encoders:

    static → MLP → static_embed
    history → Transformer/LSTM → history_embed
    [static_embed ; history_embed] → policy/value heads

All per-player features use RELATIVE indexing:
  index 0 = self, 1 = next clockwise, 2 = partner, 3 = previous clockwise

---

Static features (STATIC_OBS_SIZE = 196):
  [0:56]      own hand (56 binary)
  [56:112]    all played cards (56 binary)
  [112:168]   current trick cards on table (56 binary)
  [168:172]   tichu declarations (4 binary, relative)
  [172:176]   grand tichu declarations (4 binary, relative)
  [176:180]   player finished (4 binary, relative)
  [180:184]   cards remaining per player (4 float /14, relative)
  [184]       is leading (binary)
  [185]       mahjong wish active (binary)
  [186]       mahjong wish rank (float, (rank-2)/12, 0 if none)
  [187]       current trick combo type (float, type/9)
  [188]       current trick combo rank (float, rank/15)
  [189]       current trick combo length (float, length/14)
  [190]       current trick leading player (float, relative/3)
  [191]       number of consecutive passes in current trick (float, /3)
  [192]       dragon give pending (binary)
  [193]       round progress (float, cards_played / 56)
  [194]       position in finish order of partner (float, 0 if not finished, else pos/4)
  [195]       partner runs the same policy weights as me (binary, from
              game_state.player_model_ids; 1 in pure self-play — enables
              learned implicit signaling conventions to switch off when
              partnered with a foreign model or a human)

Additional provenance features are provided separately as:
  pass_source_ids:
  (left_card_id, partner_card_id, right_card_id)
  Each is a persistent card id for the received card from that source seat,
  or PASS_SOURCE_NONE before passing has resolved.
  pass_target_ids:
  (left_card_id, partner_card_id, right_card_id)
  Each is a persistent card id for the card this player passed to that target
  seat, or PASS_SOURCE_NONE before passing has resolved.

---

Play history tokens (PLAY_TOKEN_SIZE = 63, only when track_history=True):
  Each token represents one play (or pass) in chronological order.

  [0:4]       player who played (4 one-hot, relative to observer)
  [4:60]      cards played (56 binary, all zeros for pass)
  [60]        is pass (binary)
  [61]        trick index (float, normalized: trick_num / max_tricks)
  [62]        is trick opener (binary — first play of a new trick)

  Note: tichu/grand tichu calls are NOT encoded in history tokens.
  They are already visible in the static features ([168:176]).

  Max sequence length: MAX_HISTORY_LEN = 200 (padded with zeros)
  A separate history_len int tells the model how many tokens are real.
"""

from __future__ import annotations

import numpy as np

from tichu_env.cards import CARD_POINTS, NUM_CARDS, card_rank
from tichu_env.combinations import CombinationType

STATIC_OBS_SIZE = 196
COUNTED_TRICK_POINTS_SIZE = 4
COUNTED_TRICK_POINTS_OBS_SIZE = STATIC_OBS_SIZE + COUNTED_TRICK_POINTS_SIZE
COUNTED_TRICK_POINTS_SCALE = 100.0
PLAY_TOKEN_SIZE = 63
MAX_HISTORY_LEN = 200
PASS_SOURCE_COUNT = 3
PASS_SOURCE_NONE = NUM_CARDS
PASS_OBS_SIZE = NUM_CARDS + 4  # own hand + grand tichu declarations


def _relative_player(observer: int, player: int) -> int:
    """Map absolute player index to relative coordinates.
    
    Returns:
      0: The observing player (self)
      1: Next player (clockwise)
      2: Teammate (partner)
      3: Previous player (counter-clockwise)
    """
    return (player - observer) % 4


def encode_pass_obs(game_state, observer: int, out: np.ndarray | None = None) -> np.ndarray:
    """Passing-network input (PASS_OBS_SIZE = 60).

      [0:56]   own pre-pass hand (56 binary)
      [56:60]  grand tichu declarations (4 binary, relative, [56] = self)

    The declaration bits let the net pass to support a declaring partner
    (or starve a declaring opponent) instead of passing blind.
    """
    if out is not None:
        obs = out
        obs.fill(0)
    else:
        obs = np.zeros(PASS_OBS_SIZE, dtype=np.float32)

    for c in game_state.get_passing_hand(observer):
        obs[c] = 1.0
    for i in range(4):
        if game_state.tichu_declarations[(observer + i) % 4] == 2:
            obs[NUM_CARDS + i] = 1.0
    return obs


def encode_static(game_state, observer: int, out: np.ndarray | None = None) -> np.ndarray:
    """Encode fixed-size features for a player's observation.

    Format (196-dim):
      [0:56]    - Self hand (binary mask)
      [56:112]  - All cards played in round so far (binary mask)
      [112:168] - Active cards in current trick (binary mask)
      [168:172] - Tichu declarations (binary, relative)
      [172:176] - Grand Tichu declarations (binary, relative)
      [176:180] - Finished status (binary, relative)
      ... other game metadata (wish, trick info, round progress)
    """
    if out is not None:
        obs = out
        obs.fill(0)
    else:
        obs = np.zeros(STATIC_OBS_SIZE, dtype=np.float32)

    # Own hand (56 binary)
    # Optimization: Use sorted_hands if available
    hand = getattr(game_state, 'sorted_hands', [None]*4)[observer]
    if hand is None:
        hand = sorted(game_state.hands[observer])

    for c in hand:
        obs[c] = 1.0


    # All played cards (56 binary)
    for c in game_state.played_cards:
        obs[56 + c] = 1.0

    # Current trick cards (56 binary)
    for c in game_state.trick.cards_on_table:
        obs[112 + c] = 1.0

    # Tichu declarations (relative)
    for i in range(4):
        abs_p = (observer + i) % 4
        if game_state.tichu_declarations[abs_p] == 1:
            obs[168 + i] = 1.0
        if game_state.tichu_declarations[abs_p] == 2:
            obs[172 + i] = 1.0

    # Finished (relative)
    for i in range(4):
        abs_p = (observer + i) % 4
        if abs_p in game_state.finish_order:
            obs[176 + i] = 1.0

    # Cards remaining (relative, normalized)
    for i in range(4):
        abs_p = (observer + i) % 4
        obs[180 + i] = len(game_state.hands[abs_p]) / 14.0

    # Is leading
    obs[184] = 1.0 if game_state.trick.leading_combo is None else 0.0

    # Mahjong wish
    if game_state.mahjong_wish is not None:
        obs[185] = 1.0
        obs[186] = (game_state.mahjong_wish - 2) / 12.0

    # Current trick info
    if game_state.trick.leading_combo is not None:
        combo = game_state.trick.leading_combo
        obs[187] = combo.type.value / 9.0
        obs[188] = combo.rank / 15.0
        obs[189] = combo.length / 14.0
        obs[190] = _relative_player(observer, game_state.trick.leading_player) / 3.0
    obs[191] = game_state.trick.passes / 3.0

    # Dragon give pending
    obs[192] = 1.0 if game_state.dragon_give_pending is not None else 0.0

    # Round progress
    obs[193] = len(game_state.played_cards) / 56.0

    # Partner finish position
    partner = (observer + 2) % 4
    if partner in game_state.finish_order:
        obs[194] = (game_state.finish_order.index(partner) + 1) / 4.0

    # Partner same-weights bit
    ids = game_state.player_model_ids
    if ids[observer] == ids[partner]:
        obs[195] = 1.0

    return obs


def encode_static_counted_trick_points(
    game_state,
    observer: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Append captured trick points for all seats in relative player order.

    The final four values are ordered self, next clockwise, partner, previous
    clockwise. Each is the sum of card points currently in that player's won
    pile divided by 100. Values are deliberately not clamped because the
    Phoenix can make a pile negative and the Dragon can push one above 100.
    """
    if out is not None:
        obs = out
        obs.fill(0)
    else:
        obs = np.zeros(COUNTED_TRICK_POINTS_OBS_SIZE, dtype=np.float32)

    encode_static(game_state, observer, out=obs[:STATIC_OBS_SIZE])
    for relative_player in range(4):
        absolute_player = (observer + relative_player) % 4
        points = sum(
            CARD_POINTS.get(card, 0)
            for card in game_state.won_cards[absolute_player]
        )
        obs[STATIC_OBS_SIZE + relative_player] = (
            points / COUNTED_TRICK_POINTS_SCALE
        )
    return obs


def encode_pass_source_ids(game_state, observer: int) -> np.ndarray:
    """Encode persistent ids of cards received from left/partner/right.

    Returns int64 array of shape (3,), using PASS_SOURCE_NONE when the
    passing phase has not populated provenance for that source seat.
    """
    incoming = game_state._incoming_pass_source[observer]
    left = (observer + 1) % 4
    partner = (observer + 2) % 4
    right = (observer + 3) % 4
    ids = np.full(PASS_SOURCE_COUNT, PASS_SOURCE_NONE, dtype=np.int64)
    for i, source in enumerate((left, partner, right)):
        card = incoming.get(source)
        if card is not None:
            ids[i] = card
    return ids


def encode_pass_target_ids(game_state, observer: int) -> np.ndarray:
    """Encode persistent ids of cards passed to left/partner/right."""
    outgoing = game_state._outgoing_pass_target[observer]
    left = (observer + 1) % 4
    partner = (observer + 2) % 4
    right = (observer + 3) % 4
    ids = np.full(PASS_SOURCE_COUNT, PASS_SOURCE_NONE, dtype=np.int64)
    for i, target in enumerate((left, partner, right)):
        card = outgoing.get(target)
        if card is not None:
            ids[i] = card
    return ids


def encode_history(game_state, observer: int, limit: int | None = None) -> tuple[np.ndarray, int]:
    """Encode the play history as a padded sequence of tokens.

    Only useful when game_state.track_history is True.
    Returns (zeros, 0) if no play_log exists.
    """
    history = np.zeros((MAX_HISTORY_LEN, PLAY_TOKEN_SIZE), dtype=np.float32)

    if not game_state.play_log:
        return history, 0

    num_plays = len(game_state.play_log)
    if limit is not None:
        num_plays = min(num_plays, limit)
    num_plays = min(num_plays, MAX_HISTORY_LEN)

    # 1. Bulk copy objective parts (cards, pass, trick_idx, is_opener)
    # These are stored at bits 4-62 in the final token, 
    # and bits 0-58 in the cached history_tokens.
    if game_state.history_tokens is not None:
        history[:num_plays, 4:63] = game_state.history_tokens[:num_plays]

    # 2. Fill in relative player one-hot (bits 0-3)
    # This is the only part that depends on the observer.
    if game_state.player_indices is not None:
        players = game_state.player_indices[:num_plays]
        rels = (players - observer) % 4
        # history[i, rels[i]] = 1.0 for all i
        history[np.arange(num_plays), rels] = 1.0

    return history, num_plays


def encode_observation(game_state, observer: int) -> dict:
    """Full observation for a player.

    Returns:
      Always: {
        "static": (STATIC_OBS_SIZE,) float32,
        "pass_source_ids": (3,) int64,
        "pass_target_ids": (3,) int64,
      }
      When track_history=True, also:
        "history": (MAX_HISTORY_LEN, PLAY_TOKEN_SIZE) float32
        "history_len": int
    """
    result = {
        "static": encode_static(game_state, observer),
        "pass_source_ids": encode_pass_source_ids(game_state, observer),
        "pass_target_ids": encode_pass_target_ids(game_state, observer),
    }

    if game_state.track_history:
        history, history_len = encode_history(game_state, observer)
        result["history"] = history
        result["history_len"] = history_len

    return result
