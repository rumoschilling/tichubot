import numpy as np
import pytest

from tichu_env.cards import DRAGON_INDEX, Rank, Suit, card_index
from training.ring_diversity import game_ci, move_groups, pooled


def c(suit, rank):
    return card_index(suit, rank)


def test_moves_equal_up_to_suit_share_a_group():
    ten_j, ten_s = c(Suit.JADE, Rank.TEN), c(Suit.SWORD, Rank.TEN)
    legal = [
        ([], False, False),
        ([ten_j], False, False),
        ([ten_s], False, False),
        ([ten_s], True, False),       # Tichu call is a different move
        ([DRAGON_INDEX], False, False),
    ]
    hand = [ten_j, ten_s, DRAGON_INDEX, c(Suit.STAR, Rank.TWO)]
    groups = move_groups(legal, hand, np.zeros(len(legal), dtype=bool))

    assert groups[1] == groups[2]
    assert len(set(groups.tolist())) == 4


def test_nothing_is_pooled_when_the_hand_holds_a_straight_flush_bomb():
    run = [c(Suit.JADE, r) for r in (Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE)]
    other_nine = c(Suit.STAR, Rank.NINE)
    legal = [([run[-1]], False, False), ([other_nine], False, False)]
    groups = move_groups(legal, run + [other_nine], np.zeros(2, dtype=bool))

    assert groups.tolist() == [0, 1]


def test_pooled_sums_probability_within_groups():
    out = pooled(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 1, 2]))
    assert out.tolist() == pytest.approx([0.1, 0.5, 0.4])


def test_game_ci_is_degenerate_for_constant_values_and_covers_the_mean():
    games = np.repeat(np.arange(50), 4)
    assert game_ci(np.full(200, 0.3), games) == pytest.approx((0.3, 0.3))
    values = np.random.default_rng(0).normal(size=200)
    lo, hi = game_ci(values, games)
    assert lo < values.mean() < hi
