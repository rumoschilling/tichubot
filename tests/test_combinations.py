"""Tests for combination classification and comparison."""

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
    beats,
    classify_cards,
    phoenix_interpretations,
)


# Helper to make card indices concisely
def ci(suit: Suit, rank: Rank) -> int:
    return card_index(suit, rank)


J, P, S, W = Suit.JADE, Suit.PAGODA, Suit.STAR, Suit.SWORD


class TestSingles:
    def test_normal_single(self):
        c = classify_cards([ci(J, Rank.ACE)])
        assert c.type == CombinationType.SINGLE
        assert c.rank == 14.0
        assert c.length == 1

    def test_mahjong_single(self):
        c = classify_cards([MAHJONG_INDEX])
        assert c.type == CombinationType.SINGLE
        assert c.rank == 1.0

    def test_dragon_single(self):
        c = classify_cards([DRAGON_INDEX])
        assert c.type == CombinationType.SINGLE
        assert c.rank == 15.0

    def test_phoenix_single(self):
        c = classify_cards([PHOENIX_INDEX])
        assert c.type == CombinationType.SINGLE
        assert c.rank == 1.5  # Default when leading

    def test_dog_is_dog(self):
        c = classify_cards([DOG_INDEX])
        assert c.type == CombinationType.DOG


class TestPairs:
    def test_normal_pair(self):
        c = classify_cards([ci(J, Rank.KING), ci(P, Rank.KING)])
        assert c.type == CombinationType.PAIR
        assert c.rank == 13.0

    def test_phoenix_pair(self):
        c = classify_cards([ci(J, Rank.SEVEN), PHOENIX_INDEX])
        assert c.type == CombinationType.PAIR
        assert c.rank == 7.0
        assert c.phoenix_as == 7

    def test_two_different_ranks_invalid(self):
        c = classify_cards([ci(J, Rank.KING), ci(P, Rank.QUEEN)])
        assert c is None

    def test_dragon_pair_invalid(self):
        c = classify_cards([DRAGON_INDEX, ci(J, Rank.ACE)])
        assert c is None


class TestTriples:
    def test_normal_triple(self):
        c = classify_cards([ci(J, Rank.TEN), ci(P, Rank.TEN), ci(S, Rank.TEN)])
        assert c.type == CombinationType.TRIPLE
        assert c.rank == 10.0

    def test_phoenix_triple(self):
        c = classify_cards([ci(J, Rank.FIVE), ci(P, Rank.FIVE), PHOENIX_INDEX])
        assert c.type == CombinationType.TRIPLE
        assert c.rank == 5.0
        assert c.phoenix_as == 5


class TestStraights:
    def test_five_card_straight(self):
        cards = [ci(J, Rank.THREE), ci(P, Rank.FOUR), ci(S, Rank.FIVE),
                 ci(W, Rank.SIX), ci(J, Rank.SEVEN)]
        c = classify_cards(cards)
        assert c.type == CombinationType.STRAIGHT
        assert c.rank == 7.0
        assert c.length == 5

    def test_straight_with_mahjong(self):
        # Mahjong (1), 2, 3, 4, 5
        cards = [MAHJONG_INDEX, ci(J, Rank.TWO), ci(P, Rank.THREE),
                 ci(S, Rank.FOUR), ci(W, Rank.FIVE)]
        c = classify_cards(cards)
        assert c.type == CombinationType.STRAIGHT
        assert c.rank == 5.0
        assert c.length == 5

    def test_straight_with_phoenix_filling_gap(self):
        # 3, 4, Phoenix(5), 6, 7
        cards = [ci(J, Rank.THREE), ci(P, Rank.FOUR), PHOENIX_INDEX,
                 ci(W, Rank.SIX), ci(J, Rank.SEVEN)]
        c = classify_cards(cards)
        assert c.type == CombinationType.STRAIGHT
        assert c.rank == 7.0
        assert c.phoenix_as == 5

    def test_six_card_straight(self):
        cards = [ci(J, Rank.EIGHT), ci(P, Rank.NINE), ci(S, Rank.TEN),
                 ci(W, Rank.JACK), ci(J, Rank.QUEEN), ci(P, Rank.KING)]
        c = classify_cards(cards)
        assert c.type == CombinationType.STRAIGHT
        assert c.rank == 13.0
        assert c.length == 6

    def test_four_cards_not_straight(self):
        cards = [ci(J, Rank.THREE), ci(P, Rank.FOUR), ci(S, Rank.FIVE),
                 ci(W, Rank.SIX)]
        c = classify_cards(cards)
        assert c is None  # Straight requires 5+

    def test_non_consecutive_not_straight(self):
        cards = [ci(J, Rank.THREE), ci(P, Rank.FOUR), ci(S, Rank.FIVE),
                 ci(W, Rank.SEVEN), ci(J, Rank.EIGHT)]
        c = classify_cards(cards)
        assert c is None

    def test_open_ended_phoenix_straight_has_both_interpretations(self):
        cards = [ci(J, Rank.THREE), ci(P, Rank.FOUR), ci(S, Rank.FIVE),
                 ci(W, Rank.SIX), PHOENIX_INDEX]

        options = phoenix_interpretations(cards)

        assert [(combo.rank, combo.phoenix_as) for combo in options] == [
            (6.0, 2),
            (7.0, 7),
        ]
        assert classify_cards(cards, phoenix_as=2).rank == 6.0
        assert classify_cards(cards, phoenix_as=7).rank == 7.0
        assert classify_cards(cards, phoenix_as=8) is None


class TestFullHouse:
    def test_normal_full_house(self):
        cards = [ci(J, Rank.KING), ci(P, Rank.KING), ci(S, Rank.KING),
                 ci(J, Rank.THREE), ci(P, Rank.THREE)]
        c = classify_cards(cards)
        assert c.type == CombinationType.FULL_HOUSE
        assert c.rank == 13.0  # Rank of the triple

    def test_phoenix_completes_pair(self):
        # KKK + 3 + Phoenix(3)
        cards = [ci(J, Rank.KING), ci(P, Rank.KING), ci(S, Rank.KING),
                 ci(J, Rank.THREE), PHOENIX_INDEX]
        c = classify_cards(cards)
        assert c.type == CombinationType.FULL_HOUSE
        assert c.rank == 13.0
        assert c.phoenix_as == 3

    def test_phoenix_completes_triple(self):
        # KK + Phoenix(K) + 33
        cards = [ci(J, Rank.KING), ci(P, Rank.KING), PHOENIX_INDEX,
                 ci(J, Rank.THREE), ci(P, Rank.THREE)]
        c = classify_cards(cards)
        assert c.type == CombinationType.FULL_HOUSE
        assert c.rank == 13.0
        assert c.phoenix_as == 13

    def test_two_pairs_with_phoenix_can_choose_either_triple(self):
        cards = [ci(J, Rank.FIVE), ci(P, Rank.FIVE),
                 ci(S, Rank.NINE), ci(W, Rank.NINE), PHOENIX_INDEX]

        options = phoenix_interpretations(cards)

        assert [(combo.rank, combo.phoenix_as) for combo in options] == [
            (5.0, 5),
            (9.0, 9),
        ]
        assert classify_cards(cards, phoenix_as=5).rank == 5.0
        assert classify_cards(cards, phoenix_as=9).rank == 9.0
        assert classify_cards(cards, phoenix_as=7) is None


class TestSteps:
    def test_two_consecutive_pairs(self):
        cards = [ci(J, Rank.FIVE), ci(P, Rank.FIVE),
                 ci(S, Rank.SIX), ci(W, Rank.SIX)]
        c = classify_cards(cards)
        assert c.type == CombinationType.STEPS
        assert c.rank == 6.0
        assert c.length == 4

    def test_three_consecutive_pairs(self):
        cards = [ci(J, Rank.FIVE), ci(P, Rank.FIVE),
                 ci(S, Rank.SIX), ci(W, Rank.SIX),
                 ci(J, Rank.SEVEN), ci(P, Rank.SEVEN)]
        c = classify_cards(cards)
        assert c.type == CombinationType.STEPS
        assert c.rank == 7.0
        assert c.length == 6

    def test_phoenix_in_steps(self):
        # 55 + Phoenix(6) + 6
        cards = [ci(J, Rank.FIVE), ci(P, Rank.FIVE),
                 ci(S, Rank.SIX), PHOENIX_INDEX]
        c = classify_cards(cards)
        assert c.type == CombinationType.STEPS
        assert c.rank == 6.0
        assert c.phoenix_as == 6

    def test_non_consecutive_pairs_invalid(self):
        cards = [ci(J, Rank.FIVE), ci(P, Rank.FIVE),
                 ci(S, Rank.EIGHT), ci(W, Rank.EIGHT)]
        c = classify_cards(cards)
        assert c is None


class TestBombs:
    def test_four_of_a_kind(self):
        cards = [ci(J, Rank.JACK), ci(P, Rank.JACK),
                 ci(S, Rank.JACK), ci(W, Rank.JACK)]
        c = classify_cards(cards)
        assert c.type == CombinationType.FOUR_BOMB
        assert c.rank == 11.0

    def test_straight_flush(self):
        cards = [ci(J, Rank.THREE), ci(J, Rank.FOUR), ci(J, Rank.FIVE),
                 ci(J, Rank.SIX), ci(J, Rank.SEVEN)]
        c = classify_cards(cards)
        assert c.type == CombinationType.STRAIGHT_BOMB
        assert c.rank == 7.0
        assert c.length == 5

    def test_phoenix_not_in_bomb(self):
        # Phoenix can't be in a bomb
        cards = [ci(J, Rank.JACK), ci(P, Rank.JACK),
                 ci(S, Rank.JACK), PHOENIX_INDEX]
        c = classify_cards(cards)
        # Should be classified as something, but NOT a bomb
        assert c is None or c.type != CombinationType.FOUR_BOMB


class TestBeats:
    def test_higher_single_beats_lower(self):
        low = classify_cards([ci(J, Rank.THREE)])
        high = classify_cards([ci(P, Rank.ACE)])
        assert beats(high, low)
        assert not beats(low, high)

    def test_dragon_beats_ace(self):
        ace = classify_cards([ci(J, Rank.ACE)])
        dragon = classify_cards([DRAGON_INDEX])
        assert beats(dragon, ace)
        assert not beats(ace, dragon)

    def test_bomb_beats_non_bomb(self):
        single = classify_cards([DRAGON_INDEX])
        bomb = classify_cards([ci(J, Rank.TWO), ci(P, Rank.TWO),
                               ci(S, Rank.TWO), ci(W, Rank.TWO)])
        assert beats(bomb, single)
        assert not beats(single, bomb)

    def test_higher_bomb_beats_lower(self):
        low_bomb = classify_cards([ci(J, Rank.THREE), ci(P, Rank.THREE),
                                    ci(S, Rank.THREE), ci(W, Rank.THREE)])
        high_bomb = classify_cards([ci(J, Rank.ACE), ci(P, Rank.ACE),
                                     ci(S, Rank.ACE), ci(W, Rank.ACE)])
        assert beats(high_bomb, low_bomb)
        assert not beats(low_bomb, high_bomb)

    def test_straight_bomb_beats_four_bomb(self):
        four = classify_cards([ci(J, Rank.ACE), ci(P, Rank.ACE),
                                ci(S, Rank.ACE), ci(W, Rank.ACE)])
        sf = classify_cards([ci(J, Rank.TWO), ci(J, Rank.THREE),
                              ci(J, Rank.FOUR), ci(J, Rank.FIVE),
                              ci(J, Rank.SIX)])
        assert beats(sf, four)
        assert not beats(four, sf)

    def test_pair_must_match_type(self):
        pair = classify_cards([ci(J, Rank.ACE), ci(P, Rank.ACE)])
        single = classify_cards([ci(S, Rank.TWO)])
        assert not beats(pair, single)
        assert not beats(single, pair)

    def test_straight_same_length_required(self):
        s5 = classify_cards([ci(J, Rank.THREE), ci(P, Rank.FOUR),
                              ci(S, Rank.FIVE), ci(W, Rank.SIX),
                              ci(J, Rank.SEVEN)])
        s6 = classify_cards([ci(J, Rank.FOUR), ci(P, Rank.FIVE),
                              ci(S, Rank.SIX), ci(W, Rank.SEVEN),
                              ci(J, Rank.EIGHT), ci(P, Rank.NINE)])
        assert not beats(s6, s5)  # Different length

    def test_anything_beats_none(self):
        single = classify_cards([ci(J, Rank.TWO)])
        assert beats(single, None)

    def test_pass_beats_nothing(self):
        from tichu_env.combinations import PASS
        single = classify_cards([ci(J, Rank.TWO)])
        assert not beats(PASS, single)
