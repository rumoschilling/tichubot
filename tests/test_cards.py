"""Tests for card representation."""

from tichu_env.cards import (
    CARD_POINTS,
    DRAGON_INDEX,
    DOG_INDEX,
    MAHJONG_INDEX,
    NUM_CARDS,
    PHOENIX_INDEX,
    Deck,
    Rank,
    Suit,
    card_index,
    card_name,
    card_rank,
    card_suit,
)


class TestCardIndexing:
    def test_num_cards(self):
        assert NUM_CARDS == 56

    def test_normal_card_indices(self):
        # First card: Jade 2 = index 0
        assert card_index(Suit.JADE, Rank.TWO) == 0
        assert card_suit(0) == Suit.JADE
        assert card_rank(0) == Rank.TWO

        # Jade Ace = index 12
        assert card_index(Suit.JADE, Rank.ACE) == 12
        assert card_suit(12) == Suit.JADE
        assert card_rank(12) == Rank.ACE

        # Pagoda 2 = index 13
        assert card_index(Suit.PAGODA, Rank.TWO) == 13
        assert card_suit(13) == Suit.PAGODA
        assert card_rank(13) == Rank.TWO

        # Sword Ace = index 51
        assert card_index(Suit.SWORD, Rank.ACE) == 51
        assert card_suit(51) == Suit.SWORD
        assert card_rank(51) == Rank.ACE

    def test_special_card_indices(self):
        assert MAHJONG_INDEX == 52
        assert DOG_INDEX == 53
        assert PHOENIX_INDEX == 54
        assert DRAGON_INDEX == 55

    def test_special_cards_no_suit_rank(self):
        for idx in [MAHJONG_INDEX, DOG_INDEX, PHOENIX_INDEX, DRAGON_INDEX]:
            assert card_suit(idx) is None
            assert card_rank(idx) is None

    def test_card_names(self):
        assert card_name(MAHJONG_INDEX) == "Mahjong"
        assert card_name(DOG_INDEX) == "Dog"
        assert card_name(PHOENIX_INDEX) == "Phoenix"
        assert card_name(DRAGON_INDEX) == "Dragon"
        assert card_name(card_index(Suit.JADE, Rank.ACE)) == "Ag"
        assert card_name(card_index(Suit.STAR, Rank.TEN)) == "10r"

    def test_round_trip(self):
        for suit in Suit:
            for rank in Rank:
                idx = card_index(suit, rank)
                assert card_suit(idx) == suit
                assert card_rank(idx) == rank


class TestCardPoints:
    def test_fives(self):
        for suit in Suit:
            assert CARD_POINTS[card_index(suit, Rank.FIVE)] == 5

    def test_tens(self):
        for suit in Suit:
            assert CARD_POINTS[card_index(suit, Rank.TEN)] == 10

    def test_kings(self):
        for suit in Suit:
            assert CARD_POINTS[card_index(suit, Rank.KING)] == 10

    def test_no_points(self):
        for suit in Suit:
            for rank in [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.SIX,
                         Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.JACK,
                         Rank.QUEEN, Rank.ACE]:
                assert CARD_POINTS[card_index(suit, rank)] == 0

    def test_special_points(self):
        assert CARD_POINTS[DRAGON_INDEX] == 25
        assert CARD_POINTS[PHOENIX_INDEX] == -25
        assert CARD_POINTS[MAHJONG_INDEX] == 0
        assert CARD_POINTS[DOG_INDEX] == 0

    def test_total_points(self):
        total = sum(CARD_POINTS[i] for i in range(NUM_CARDS))
        # 4×5 + 4×10 + 4×10 + 25 - 25 = 100
        assert total == 100


class TestDeck:
    def test_full_deck(self):
        deck = Deck.full()
        assert len(deck.cards) == 56
        assert set(deck.cards) == set(range(56))

    def test_deal(self):
        deck = Deck.full()
        deck.shuffle()
        h0, h1, h2, h3 = deck.deal()
        assert len(h0) == 14
        assert len(h1) == 14
        assert len(h2) == 14
        assert len(h3) == 14
        all_cards = set(h0) | set(h1) | set(h2) | set(h3)
        assert all_cards == set(range(56))

    def test_deal_sorted(self):
        deck = Deck.full()
        deck.shuffle()
        for hand in deck.deal():
            assert hand == sorted(hand)

    def test_seeded_shuffle(self):
        import random
        rng = random.Random(42)
        d1 = Deck.full()
        d1.shuffle(rng)

        rng2 = random.Random(42)
        d2 = Deck.full()
        d2.shuffle(rng2)

        assert d1.cards == d2.cards
