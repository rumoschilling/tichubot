"""Card representation for Tichu.

56 cards total:
  - 4 suits (Jade, Pagoda, Star, Sword) × 13 ranks (2..Ace) = 52 normal cards
  - 4 special cards: Mahjong (1), Dog, Phoenix, Dragon

Card indices 0-51: normal cards, ordered by suit then rank.
  suit_index * 13 + rank_index
  rank_index: 0=2, 1=3, ..., 8=10, 9=J, 10=Q, 11=K, 12=A

Card indices 52-55: Mahjong(52), Dog(53), Phoenix(54), Dragon(55)
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass


class Suit(enum.IntEnum):
    JADE = 0
    PAGODA = 1
    STAR = 2
    SWORD = 3


class Rank(enum.IntEnum):
    """Rank values correspond to face value (2-14 for 2-Ace)."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


# ── Special Card Indices ──────────────────────────────────────────────
MAHJONG_INDEX = 52  # Rank 1, no suit. Starts the round.
DOG_INDEX     = 53  # Rank 0, no suit. Passes lead to teammate.
PHOENIX_INDEX = 54  # Wildcard. Can fill gaps or be a single (+0.5 rank).
DRAGON_INDEX  = 55  # Rank 15, no suit. Highest single, must be given away.

NUM_CARDS = 56

SPECIAL_CARDS = {MAHJONG_INDEX, DOG_INDEX, PHOENIX_INDEX, DRAGON_INDEX}

# Point values for scoring
CARD_POINTS: dict[int, int] = {}


def _init_card_points():
    for i in range(52):
        rank = Rank(i % 13 + 2)
        if rank == Rank.FIVE:
            CARD_POINTS[i] = 5
        elif rank == Rank.TEN or rank == Rank.KING:
            CARD_POINTS[i] = 10
        else:
            CARD_POINTS[i] = 0
    CARD_POINTS[MAHJONG_INDEX] = 0
    CARD_POINTS[DOG_INDEX] = 0
    CARD_POINTS[PHOENIX_INDEX] = -25
    CARD_POINTS[DRAGON_INDEX] = 25


_init_card_points()


def card_suit(index: int) -> Suit | None:
    """Return the suit of a normal card, or None for special cards."""
    if index >= 52:
        return None
    return Suit(index // 13)


def card_rank(index: int) -> Rank | None:
    """Return the rank of a normal card, or None for special cards."""
    if index >= 52:
        return None
    return Rank(index % 13 + 2)


def card_index(suit: Suit, rank: Rank) -> int:
    """Get the card index for a normal card given suit and rank."""
    return suit.value * 13 + (rank.value - 2)


def card_name(index: int) -> str:
    """Human-readable card name."""
    if index == MAHJONG_INDEX:
        return "Mahjong"
    if index == DOG_INDEX:
        return "Dog"
    if index == PHOENIX_INDEX:
        return "Phoenix"
    if index == DRAGON_INDEX:
        return "Dragon"
    suit = card_suit(index)
    rank = card_rank(index)
    rank_names = {
        Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A",
    }
    suit_letters = {Suit.JADE: "g", Suit.PAGODA: "b", Suit.STAR: "r", Suit.SWORD: "s"}
    r = rank_names.get(rank, str(rank.value))
    return f"{r}{suit_letters[suit]}"


@dataclass
class Deck:
    """A standard 56-card Tichu deck.
    
    Responsibilities: Shuffling and dealing 14-card hands.
    """
    cards: list[int]

    @classmethod
    def full(cls) -> Deck:
        return cls(cards=list(range(NUM_CARDS)))

    def shuffle(self, rng: random.Random | None = None):
        if rng is None:
            random.shuffle(self.cards)
        else:
            rng.shuffle(self.cards)

    def deal(self) -> tuple[list[int], list[int], list[int], list[int]]:
        """Deal 14 cards to each of 4 players. Deck must have 56 cards."""
        assert len(self.cards) == 56
        return (
            sorted(self.cards[0:14]),
            sorted(self.cards[14:28]),
            sorted(self.cards[28:42]),
            sorted(self.cards[42:56]),
        )
