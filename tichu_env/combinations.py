"""Tichu combination types and classification.

Combination types:
  SINGLE:       1 card
  PAIR:         2 cards of same rank
  TRIPLE:       3 cards of same rank
  STRAIGHT:     5+ consecutive ranks (Phoenix can fill a gap)
  FULL_HOUSE:   triple + pair
  STEPS:        2+ consecutive pairs (Phoenix can fill in)
  FOUR_BOMB:    4 cards of same rank (bomb)
  STRAIGHT_BOMB: 5+ consecutive cards of same suit (bomb)

  PASS:         no cards (pass turn)
  DOG:          Dog special card (played alone to pass lead)

The Phoenix can substitute for any normal card in combinations.
When played as a single, it is half a rank above the previous single.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from itertools import combinations as iter_combinations

from tichu_env.cards import (
    DRAGON_INDEX,
    DOG_INDEX,
    MAHJONG_INDEX,
    NUM_CARDS,
    PHOENIX_INDEX,
    Rank,
    Suit,
    card_rank,
    card_suit,
)


class CombinationType(enum.IntEnum):
    PASS = 0
    DOG = 1
    SINGLE = 2
    PAIR = 3
    TRIPLE = 4
    STRAIGHT = 5
    FULL_HOUSE = 6
    STEPS = 7        # consecutive pairs
    FOUR_BOMB = 8
    STRAIGHT_BOMB = 9


@dataclass(frozen=True)
class Combination:
    """A classified combination of cards.

    Attributes:
        type: The combination type.
        cards: Tuple of card indices in the combination.
        rank: The "height" of the combination for comparison purposes.
              For singles/pairs/triples: the rank value (2-14, or 15 for Dragon).
              For straights/steps: the highest rank in the sequence.
              For full house: the rank of the triple.
              For bombs: the rank of the quad / highest rank of straight flush.
        length: Number of cards (relevant for straights and steps where
                only same-length combos can be compared).
        phoenix_as: If Phoenix is used, what rank it substitutes for (None otherwise).
    """
    type: CombinationType
    cards: tuple[int, ...]
    rank: float
    length: int
    phoenix_as: int | None = None

    @property
    def is_bomb(self) -> bool:
        return self.type in (CombinationType.FOUR_BOMB, CombinationType.STRAIGHT_BOMB)


# Sentinel for pass
PASS = Combination(
    type=CombinationType.PASS, cards=(), rank=0, length=0
)


def _rank_counts(card_indices: list[int]) -> dict[int, int]:
    """Count how many cards of each rank are present (ignoring specials)."""
    counts: dict[int, int] = {}
    for c in card_indices:
        r = card_rank(c)
        if r is not None:
            counts[r.value] = counts.get(r.value, 0) + 1
    return counts


def _suits_for_rank(card_indices: list[int], target_rank: int) -> list[Suit]:
    """Return all suits present for a given rank among the cards."""
    suits = []
    for c in card_indices:
        r = card_rank(c)
        s = card_suit(c)
        if r is not None and r.value == target_rank:
            suits.append(s)
    return suits


def classify_cards(
    card_indices: list[int] | tuple[int, ...],
    phoenix_as: int | None = None,
) -> Combination | None:
    """Classify a set of cards into a Combination, or None if invalid.

    This handles Phoenix as wildcard in all combination types. ``phoenix_as``
    can force its rank for an explicit player choice; the default keeps the
    historical deterministic interpretation used by trained policies.
    """
    cards = sorted(card_indices)
    n = len(cards)

    if n == 0:
        return PASS

    # A physical card can occur only once in a Tichu deck.  Reject repeated
    # indices here as well as in GameState.play() so the public classifier
    # cannot manufacture pairs, triples, or bombs from one card.
    if len(set(cards)) != n:
        return None

    has_phoenix = PHOENIX_INDEX in cards
    has_dragon = DRAGON_INDEX in cards
    has_dog = DOG_INDEX in cards
    has_mahjong = MAHJONG_INDEX in cards

    if phoenix_as is not None:
        if (
            isinstance(phoenix_as, bool)
            or not isinstance(phoenix_as, int)
            or not 2 <= phoenix_as <= 14
            or not has_phoenix
            or n == 1
        ):
            return None

    # Dog must be played alone
    if has_dog:
        if n == 1:
            return Combination(
                type=CombinationType.DOG,
                cards=tuple(cards),
                rank=0,
                length=1,
            )
        return None

    # Dragon must be played alone as a single
    if has_dragon:
        if n == 1:
            return Combination(
                type=CombinationType.SINGLE,
                cards=tuple(cards),
                rank=15.0,  # Beats Ace
                length=1,
            )
        return None

    # Normal cards + possibly Phoenix + possibly Mahjong
    normal_cards = [c for c in cards if c not in (PHOENIX_INDEX, MAHJONG_INDEX)]
    rank_counts = _rank_counts(cards)

    # Mahjong counts as rank 1 for straights, but can also be played as single
    if has_mahjong:
        rank_counts[1] = rank_counts.get(1, 0) + 1

    # === SINGLE ===
    if n == 1:
        if has_phoenix:
            # Phoenix as single: rank determined at play time (0.5 above previous)
            # For classification we give it rank 1.5 (just above Mahjong when leading)
            return Combination(
                type=CombinationType.SINGLE,
                cards=tuple(cards),
                rank=1.5,
                length=1,
            )
        if has_mahjong:
            return Combination(
                type=CombinationType.SINGLE,
                cards=tuple(cards),
                rank=1.0,
                length=1,
            )
        r = card_rank(cards[0])
        return Combination(
            type=CombinationType.SINGLE,
            cards=tuple(cards),
            rank=float(r.value),
            length=1,
        )

    # Collect the ranks of all cards except the Phoenix wildcard
    normal_ranks = []
    for c in cards:
        if c == PHOENIX_INDEX:
            continue
        if c == MAHJONG_INDEX:
            normal_ranks.append(1)
        else:
            r = card_rank(c)
            if r is not None:
                normal_ranks.append(r.value)
    normal_ranks.sort()

    num_wild = 1 if has_phoenix else 0
    if len(normal_ranks) + num_wild != n:
        return None  # Duplicate ranks or unexpected indices

    # === PAIR ===
    if n == 2:
        if has_phoenix and len(normal_ranks) == 1 and not has_mahjong:
            r = normal_ranks[0]
            if phoenix_as is not None and phoenix_as != r:
                return None
            return Combination(
                type=CombinationType.PAIR,
                cards=tuple(cards),
                rank=float(r),
                length=2,
                phoenix_as=r,
            )
        if not has_phoenix and len(set(normal_ranks)) == 1 and len(normal_ranks) == 2:
            return Combination(
                type=CombinationType.PAIR,
                cards=tuple(cards),
                rank=float(normal_ranks[0]),
                length=2,
            )
        return None

    # === TRIPLE ===
    if n == 3:
        if (
            has_phoenix
            and not has_mahjong
            and len(normal_ranks) == 2
            and normal_ranks[0] == normal_ranks[1]
        ):
            r = normal_ranks[0]
            if phoenix_as is not None and phoenix_as != r:
                return None
            return Combination(
                type=CombinationType.TRIPLE,
                cards=tuple(cards),
                rank=float(r),
                length=3,
                phoenix_as=r,
            )
        if not has_phoenix and len(set(normal_ranks)) == 1 and len(normal_ranks) == 3:
            return Combination(
                type=CombinationType.TRIPLE,
                cards=tuple(cards),
                rank=float(normal_ranks[0]),
                length=3,
            )
        return None

    # === 4 cards: FOUR_BOMB or steps(2 pairs) ===
    if n == 4:
        # Four of a kind bomb (no phoenix allowed in bombs)
        if not has_phoenix and len(set(normal_ranks)) == 1 and len(normal_ranks) == 4:
            return Combination(
                type=CombinationType.FOUR_BOMB,
                cards=tuple(cards),
                rank=float(normal_ranks[0]),
                length=4,
            )
        # Two consecutive pairs (steps of length 2)
        result = _try_steps(cards, normal_ranks, num_wild, phoenix_as)
        if result is not None:
            return result
        return None

    # === 5 cards: STRAIGHT, FULL_HOUSE, STRAIGHT_BOMB, or steps ===
    if n == 5:
        # Try straight flush bomb first (no phoenix allowed)
        bomb = _try_straight_bomb(cards, normal_ranks, has_phoenix, has_mahjong)
        if bomb is not None:
            return bomb

        # Try full house
        fh = _try_full_house(cards, normal_ranks, num_wild, phoenix_as)
        if fh is not None:
            return fh

        # Try straight
        straight = _try_straight(
            cards, normal_ranks, num_wild, has_mahjong, phoenix_as
        )
        if straight is not None:
            return straight

        # Try steps (but 5 cards = not even number, can't be steps)
        # Steps require even number of cards, skip
        return None

    # === 6+ cards ===
    # Try straight flush bomb (no phoenix)
    bomb = _try_straight_bomb(cards, normal_ranks, has_phoenix, has_mahjong)
    if bomb is not None:
        return bomb

    # Try steps (even number of cards)
    if n % 2 == 0:
        result = _try_steps(cards, normal_ranks, num_wild, phoenix_as)
        if result is not None:
            return result

    # Try straight (any length >= 5)
    if n >= 5:
        straight = _try_straight(
            cards, normal_ranks, num_wild, has_mahjong, phoenix_as
        )
        if straight is not None:
            return straight

    return None


def _try_straight(
    cards: list[int],
    normal_ranks: list[int],
    num_wild: int,
    has_mahjong: bool,
    forced_phoenix_as: int | None = None,
) -> Combination | None:
    """Try to classify cards as a straight (5+ consecutive ranks).

    Mahjong (rank 1) can start a straight. Phoenix can fill one gap or be at ends.
    """
    n = len(cards)
    if n < 5:
        return None

    unique_ranks = set(normal_ranks)
    # n is the total number of cards. 
    # normal_ranks contains all standard cards + Mahjong (if present).
    # num_wild is 1 if Phoenix is present.
    if len(normal_ranks) + num_wild != n:
        return None  # Duplicate ranks or special card mismatch

    if unique_ranks and max(unique_ranks) > 14:
        return None

    # Try all possible start ranks for a straight of length n
    # Start rank can be 1 (Mahjong) up to 14-n+1
    best_combo = None

    # Rank 1 is the physical Mahjong, not a normal rank the Phoenix may
    # represent.  Therefore only consider a rank-1 window when Mahjong is
    # actually among the selected cards.
    first_low = 1 if has_mahjong else 2
    for low in range(first_low, 14 - n + 2):
        high = low + n - 1
        # Which normal ranks (including Mahjong=1) are inside this window?
        in_window = [r for r in normal_ranks if low <= r <= high]
        
        # To be a valid straight:
        # 1. Every normal card MUST be in the window (no cards outside allowed)
        # 2. There MUST be no duplicate ranks in the normal cards
        # 3. The number of missing slots in the window MUST equal our wildcard count
        
        if len(in_window) == len(normal_ranks) and len(set(in_window)) == len(in_window):
            if (n - len(in_window)) == num_wild:
                phoenix_as = None
                if num_wild > 0:
                    for r in range(low, high + 1):
                        if r not in unique_ranks:
                            phoenix_as = r
                            break
                    if (
                        forced_phoenix_as is not None
                        and phoenix_as != forced_phoenix_as
                    ):
                        continue
                
                new_combo = Combination(
                    type=CombinationType.STRAIGHT,
                    cards=tuple(cards),
                    rank=float(high),
                    length=n,
                    phoenix_as=phoenix_as,
                )
                # Keep the one with the highest rank (important for beating tricks)
                if best_combo is None or new_combo.rank > best_combo.rank:
                    best_combo = new_combo

    return best_combo


def _try_full_house(
    cards: list[int],
    normal_ranks: list[int],
    num_wild: int,
    forced_phoenix_as: int | None = None,
) -> Combination | None:
    """Try to classify 5 cards as a full house (triple + pair)."""
    if len(cards) != 5:
        return None

    from collections import Counter
    rank_freq = Counter(normal_ranks)
    if 1 in rank_freq:
        return None  # Mahjong cannot form either part of a full house.
    freq_values = sorted(rank_freq.values(), reverse=True)

    if num_wild == 0:
        # Need exactly one rank with 3 and another with 2
        if freq_values == [3, 2]:
            triple_rank = [r for r, c in rank_freq.items() if c == 3][0]
            return Combination(
                type=CombinationType.FULL_HOUSE,
                cards=tuple(cards),
                rank=float(triple_rank),
                length=5,
            )
    elif num_wild == 1:
        # Phoenix completes either a pair→triple or a single→pair
        # Case 1: we have (3, 1) — phoenix makes the 1 into a pair
        if freq_values == [3, 1]:
            triple_rank = [r for r, c in rank_freq.items() if c == 3][0]
            pair_rank = [r for r, c in rank_freq.items() if c == 1][0]
            if forced_phoenix_as is not None and forced_phoenix_as != pair_rank:
                return None
            return Combination(
                type=CombinationType.FULL_HOUSE,
                cards=tuple(cards),
                rank=float(triple_rank),
                length=5,
                phoenix_as=pair_rank,
            )
        # Case 2: we have (2, 2) — phoenix makes one pair into a triple
        if freq_values == [2, 2]:
            ranks = sorted(rank_freq.keys(), reverse=True)
            # The historical automatic convention uses the higher pair. An
            # explicit player choice may instead promote either natural pair.
            triple_rank = forced_phoenix_as if forced_phoenix_as is not None else ranks[0]
            if triple_rank not in ranks:
                return None
            return Combination(
                type=CombinationType.FULL_HOUSE,
                cards=tuple(cards),
                rank=float(triple_rank),
                length=5,
                phoenix_as=triple_rank,
            )

    return None


def _try_steps(
    cards: list[int],
    normal_ranks: list[int],
    num_wild: int,
    forced_phoenix_as: int | None = None,
) -> Combination | None:
    """Try to classify cards as steps (consecutive pairs). Minimum 2 pairs (4 cards)."""
    n = len(cards)
    if n < 4 or n % 2 != 0:
        return None

    num_pairs_needed = n // 2
    from collections import Counter
    rank_freq = Counter(normal_ranks)
    unique_ranks = set(rank_freq.keys())

    # Try all possible start ranks for the sequence of pairs
    # Rank 1 (Mahjong) cannot pair. 
    for low in range(2, 14 - num_pairs_needed + 2):
        high = low + num_pairs_needed - 1
        
        # Check if all our normal cards fit into this [low, high] window
        # AND check that no rank in this window has more than 2 cards
        # AND check that we don't have any cards outside this window
        if not unique_ranks.issubset(range(low, high + 1)):
            continue
            
        gaps_to_fill = 0
        possible = True
        for r in range(low, high + 1):
            count = rank_freq.get(r, 0)
            if count > 2:
                possible = False
                break
            gaps_to_fill += (2 - count)
            
        if not possible:
            continue
            
        if gaps_to_fill == num_wild:
            phoenix_as = None
            if num_wild > 0:
                # Find which rank in the sequence the phoenix completes
                for r in range(low, high + 1):
                    if rank_freq.get(r, 0) == 1:
                        phoenix_as = r
                        break
                if (
                    forced_phoenix_as is not None
                    and phoenix_as != forced_phoenix_as
                ):
                    continue
            
            return Combination(
                type=CombinationType.STEPS,
                cards=tuple(cards),
                rank=float(high),
                length=n,
                phoenix_as=phoenix_as,
            )

    return None


def phoenix_interpretations(
    card_indices: list[int] | tuple[int, ...],
) -> list[Combination]:
    """Return every distinct valid meaning of a physical Phoenix card set."""

    default = classify_cards(card_indices)
    if default is None:
        return []
    if PHOENIX_INDEX not in card_indices or len(card_indices) == 1:
        return [default]

    interpretations: dict[tuple, Combination] = {}
    for rank in range(2, 15):
        combo = classify_cards(card_indices, phoenix_as=rank)
        if combo is None:
            continue
        signature = (combo.type, combo.rank, combo.length, combo.phoenix_as)
        interpretations[signature] = combo
    if not interpretations:
        return [default]
    return sorted(
        interpretations.values(),
        key=lambda combo: (combo.type, combo.rank, combo.phoenix_as or 0),
    )


# Rank signature -> None or the (type, rank, length, phoenix_as) of the lower
# and higher meanings. A Phoenix set of five or more cards can only be a
# straight, full house or steps (never a bomb), so its meanings depend on
# ranks alone, not suits.
_BINARY_PHOENIX_CACHE: dict[tuple[int, ...], tuple | None] = {}


def binary_phoenix_interpretations(
    card_indices: list[int] | tuple[int, ...],
) -> tuple[Combination, Combination] | None:
    """Return the lower/higher meanings represented by the policy choice bit.

    A physical Phoenix play is strategically ambiguous only when it can be a
    lower or higher straight, or when it can promote either natural pair in a
    full house. Other Phoenix combinations have a single forced meaning.

    Results are cached by rank signature: rollout asks this for every legal
    Phoenix action at every decision, and each miss costs 14 classifications.
    """
    # All relevant ambiguous combinations have at least five cards.  This
    # cheap guard matters because policy rollout calls this helper once for
    # every otherwise-legal action, most of which do not contain the Phoenix.
    if PHOENIX_INDEX not in card_indices or len(card_indices) < 5:
        return None
    if len(set(card_indices)) != len(card_indices):
        return _binary_phoenix_interpretations_uncached(card_indices)

    key = tuple(sorted(c % 13 if c < 52 else c for c in card_indices))
    try:
        cached = _BINARY_PHOENIX_CACHE[key]
    except KeyError:
        result = _binary_phoenix_interpretations_uncached(card_indices)
        _BINARY_PHOENIX_CACHE[key] = None if result is None else tuple(
            (c.type, c.rank, c.length, c.phoenix_as) for c in result
        )
        return result
    if cached is None:
        return None
    cards = tuple(sorted(card_indices))
    (lt, lr, ll, lp), (ht, hr, hl, hp) = cached
    return (
        Combination(type=lt, cards=cards, rank=lr, length=ll, phoenix_as=lp),
        Combination(type=ht, cards=cards, rank=hr, length=hl, phoenix_as=hp),
    )


def _binary_phoenix_interpretations_uncached(
    card_indices: list[int] | tuple[int, ...],
) -> tuple[Combination, Combination] | None:
    interpretations = [
        combo
        for combo in phoenix_interpretations(card_indices)
        if combo.type in (CombinationType.STRAIGHT, CombinationType.FULL_HOUSE)
    ]
    if len(interpretations) <= 1:
        return None
    if len(interpretations) != 2:
        raise RuntimeError(
            "Phoenix policy bit cannot represent "
            f"{len(interpretations)} interpretations for {card_indices}"
        )
    lower, higher = interpretations
    if lower.rank >= higher.rank:
        raise RuntimeError(
            f"Phoenix interpretations are not ordered: {lower}, {higher}"
        )
    return lower, higher


def _try_straight_bomb(
    cards: list[int],
    normal_ranks: list[int],
    has_phoenix: bool,
    has_mahjong: bool,
) -> Combination | None:
    """Try to classify cards as a straight flush bomb.

    No phoenix allowed in bombs. All cards must be the same suit and consecutive.
    Mahjong (rank 1) can be part of a straight bomb if it were suited, but
    Mahjong is special/suitless — so it cannot participate in a straight bomb.
    """
    n = len(cards)
    if n < 5 or has_phoenix:
        return None

    # All cards must be normal (suited) cards — no specials
    normal_only = [c for c in cards if c < 52]
    if len(normal_only) != n:
        return None

    # All same suit
    suits = set(card_suit(c) for c in normal_only)
    if len(suits) != 1:
        return None

    # Consecutive ranks
    ranks = sorted(card_rank(c).value for c in normal_only)
    for i in range(1, len(ranks)):
        if ranks[i] != ranks[i - 1] + 1:
            return None

    return Combination(
        type=CombinationType.STRAIGHT_BOMB,
        cards=tuple(cards),
        rank=float(ranks[-1]),
        length=n,
    )


def beats(play: Combination, current: Combination | None) -> bool:
    """Check if `play` can beat `current` (the leading combination of the trick).

    Returns True if the play is legal on top of current.
    If current is None, any valid combination can be played (leading).
    """
    if current is None or current.type == CombinationType.PASS:
        # Leading: any valid combination can be played
        return play.type != CombinationType.PASS

    # Bombs beat everything
    if play.is_bomb and not current.is_bomb:
        return True
    if play.is_bomb and current.is_bomb:
        # Straight bomb beats four-of-a-kind bomb
        if play.type == CombinationType.STRAIGHT_BOMB and current.type == CombinationType.FOUR_BOMB:
            return True
        if play.type == CombinationType.FOUR_BOMB and current.type == CombinationType.STRAIGHT_BOMB:
            return False
        # Same bomb type: compare by rank, then length (for straight bombs)
        if play.type == current.type:
            if play.type == CombinationType.STRAIGHT_BOMB:
                if play.length != current.length:
                    return play.length > current.length
            return play.rank > current.rank
        return False

    # Non-bomb cannot beat a bomb
    if not play.is_bomb and current.is_bomb:
        return False

    # Same combination type required, same length
    if play.type != current.type:
        return False
    if play.length != current.length:
        return False

    # Higher rank wins
    return play.rank > current.rank
