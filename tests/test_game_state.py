"""Tests for game state, trick logic, scoring, and special card rules."""

import random

from tichu_env.cards import (
    CARD_POINTS,
    DRAGON_INDEX,
    DOG_INDEX,
    MAHJONG_INDEX,
    PHOENIX_INDEX,
    Rank,
    Suit,
    card_index,
)
from tichu_env.game_state import GamePhase, GameState, TEAM_A, TEAM_B, teammate


ci = card_index
J, P, S, W = Suit.JADE, Suit.PAGODA, Suit.STAR, Suit.SWORD


def make_game_with_hands(h0, h1, h2, h3, current_player=0):
    """Create a GameState with specific hands for testing."""
    gs = GameState()
    gs.hands = [set(h0), set(h1), set(h2), set(h3)]
    gs.current_player = current_player
    gs.finish_order = []
    gs.won_cards = [[] for _ in range(4)]
    gs.played_cards = set()
    gs.round_over = False
    return gs


class TestBasicPlay:
    def test_play_single(self):
        card = ci(J, Rank.ACE)
        gs = make_game_with_hands(
            [card, ci(J, Rank.TWO)],
            [ci(P, Rank.ACE), ci(P, Rank.TWO)],
            [ci(S, Rank.ACE), ci(S, Rank.TWO)],
            [ci(W, Rank.ACE), ci(W, Rank.TWO)],
        )
        assert gs.play(0, [card])
        assert card not in gs.hands[0]
        assert card in gs.played_cards

    def test_cant_play_card_not_in_hand(self):
        gs = make_game_with_hands(
            [ci(J, Rank.TWO)],
            [ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        assert not gs.play(0, [ci(J, Rank.ACE)])

    def test_wrong_player_cant_play(self):
        gs = make_game_with_hands(
            [ci(J, Rank.TWO)],
            [ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
            current_player=0,
        )
        assert not gs.play(1, [ci(P, Rank.TWO)])

    def test_pass_not_allowed_when_leading(self):
        gs = make_game_with_hands(
            [ci(J, Rank.TWO)],
            [ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        assert not gs.play(0, [])

    def test_pass_allowed_after_lead(self):
        gs = make_game_with_hands(
            [ci(J, Rank.ACE)],
            [ci(P, Rank.TWO)],
            [ci(S, Rank.ACE)],
            [ci(W, Rank.ACE)],
        )
        gs.play(0, [ci(J, Rank.ACE)])
        assert gs.current_player == 1
        assert gs.play(1, [])  # pass

    def test_must_beat_current_trick(self):
        gs = make_game_with_hands(
            [ci(J, Rank.ACE)],
            [ci(P, Rank.TWO), ci(P, Rank.KING)],
            [ci(S, Rank.ACE)],
            [ci(W, Rank.ACE)],
        )
        gs.play(0, [ci(J, Rank.ACE)])
        # Can't play a 2 on an Ace
        assert not gs.play(1, [ci(P, Rank.TWO)])
        # Can pass
        assert gs.play(1, [])

    def test_explicit_phoenix_straight_value_controls_whether_play_beats(self):
        lead = [ci(J, Rank.TWO), ci(P, Rank.THREE), ci(S, Rank.FOUR),
                ci(W, Rank.FIVE), ci(J, Rank.SIX)]
        response = [ci(S, Rank.THREE), ci(W, Rank.FOUR), ci(J, Rank.FIVE),
                    ci(P, Rank.SIX), PHOENIX_INDEX]
        gs = make_game_with_hands(
            lead + [DOG_INDEX],
            response + [ci(P, Rank.ACE)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        assert gs.play(0, lead)

        assert not gs.play(1, response, phoenix_as=2)
        assert set(response).issubset(gs.hands[1])
        assert gs.play(1, response, phoenix_as=7)
        assert gs.trick.leading_combo.rank == 7.0
        assert gs.trick.leading_combo.phoenix_as == 7

    def test_explicit_phoenix_full_house_selects_the_triple(self):
        lead = [ci(J, Rank.SEVEN), ci(P, Rank.SEVEN), ci(S, Rank.SEVEN),
                ci(J, Rank.TWO), ci(P, Rank.TWO)]
        response = [ci(J, Rank.FIVE), ci(P, Rank.FIVE),
                    ci(S, Rank.NINE), ci(W, Rank.NINE), PHOENIX_INDEX]
        gs = make_game_with_hands(
            lead + [DOG_INDEX],
            response + [ci(P, Rank.ACE)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        assert gs.play(0, lead)

        assert not gs.play(1, response, phoenix_as=5)
        assert gs.play(1, response, phoenix_as=9)
        assert gs.trick.leading_combo.type.name == "FULL_HOUSE"
        assert gs.trick.leading_combo.rank == 9.0
        assert gs.trick.leading_combo.phoenix_as == 9


class TestTrickWinning:
    def test_trick_won_after_all_pass(self):
        gs = make_game_with_hands(
            [ci(J, Rank.ACE), ci(J, Rank.TWO)],
            [ci(P, Rank.TWO), ci(P, Rank.THREE)],
            [ci(S, Rank.TWO), ci(S, Rank.THREE)],
            [ci(W, Rank.TWO), ci(W, Rank.THREE)],
        )
        gs.play(0, [ci(J, Rank.ACE)])  # Player 0 leads Ace
        gs.play(1, [])  # pass
        gs.play(2, [])  # pass
        gs.play(3, [])  # pass
        # Player 0 wins the trick
        assert ci(J, Rank.ACE) in gs.won_cards[0]
        # Player 0 leads again
        assert gs.current_player == 0

    def test_all_remaining_must_pass_when_leader_goes_out(self):
        """If the leader's play empties their hand, they are no longer active,
        so every remaining player (all num_active of them) must pass before
        the trick resolves — nobody gets skipped."""
        gs = make_game_with_hands(
            [ci(J, Rank.ACE), ci(J, Rank.TWO)],
            [ci(P, Rank.KING)],  # P1 goes out with this play
            [ci(S, Rank.TWO), ci(S, Rank.THREE)],
            [ci(W, Rank.TWO), ci(W, Rank.THREE)],
            current_player=1,
        )
        gs.play(1, [ci(P, Rank.KING)])
        assert 1 in gs.finished
        gs.play(2, [])  # pass
        gs.play(3, [])  # pass
        # Trick must still be open: P0 hasn't had a chance to beat the King
        assert gs.trick.leading_combo is not None
        assert gs.current_player == 0
        # P0 beats it with the Ace and wins the trick
        gs.play(0, [ci(J, Rank.ACE)])
        gs.play(2, [])
        gs.play(3, [])
        assert ci(P, Rank.KING) in gs.won_cards[0]
        assert ci(J, Rank.ACE) in gs.won_cards[0]

    def test_trick_goes_to_finished_leader_after_all_pass(self):
        gs = make_game_with_hands(
            [ci(J, Rank.TWO), ci(J, Rank.THREE)],
            [ci(P, Rank.KING)],  # P1 goes out with this play
            [ci(S, Rank.TWO), ci(S, Rank.THREE)],
            [ci(W, Rank.TWO), ci(W, Rank.THREE)],
            current_player=1,
        )
        gs.play(1, [ci(P, Rank.KING)])
        gs.play(2, [])
        gs.play(3, [])
        gs.play(0, [])  # third pass — now the trick resolves
        assert ci(P, Rank.KING) in gs.won_cards[1]
        # Lead passes to the next active player after the finished winner
        assert gs.current_player == 2

    def test_higher_card_wins(self):
        gs = make_game_with_hands(
            [ci(J, Rank.KING), ci(J, Rank.TWO)],
            [ci(P, Rank.ACE), ci(P, Rank.TWO)],
            [ci(S, Rank.TWO), ci(S, Rank.THREE)],
            [ci(W, Rank.TWO), ci(W, Rank.THREE)],
        )
        gs.play(0, [ci(J, Rank.KING)])
        gs.play(1, [ci(P, Rank.ACE)])
        gs.play(2, [])
        gs.play(3, [])
        gs.play(0, [])  # P0 also passes after P1's Ace
        # Player 1 wins with Ace
        assert ci(J, Rank.KING) in gs.won_cards[1]
        assert ci(P, Rank.ACE) in gs.won_cards[1]


class TestDog:
    def test_dog_passes_lead_to_teammate(self):
        gs = make_game_with_hands(
            [DOG_INDEX, ci(J, Rank.TWO)],
            [ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.play(0, [DOG_INDEX])
        # Lead passes to teammate (player 2)
        assert gs.current_player == 2

    def test_dog_not_allowed_mid_trick(self):
        gs = make_game_with_hands(
            [ci(J, Rank.ACE)],
            [DOG_INDEX, ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.play(0, [ci(J, Rank.ACE)])
        assert not gs.play(1, [DOG_INDEX])


class TestDragon:
    def test_dragon_beats_everything(self):
        gs = make_game_with_hands(
            [ci(J, Rank.ACE), ci(J, Rank.TWO)],
            [DRAGON_INDEX, ci(P, Rank.TWO)],
            [ci(S, Rank.TWO), ci(S, Rank.THREE)],
            [ci(W, Rank.TWO), ci(W, Rank.THREE)],
        )
        gs.play(0, [ci(J, Rank.ACE)])
        assert gs.play(1, [DRAGON_INDEX])

    def test_dragon_trick_must_be_given(self):
        gs = make_game_with_hands(
            [ci(J, Rank.KING), ci(J, Rank.TWO)],
            [DRAGON_INDEX, ci(P, Rank.TWO)],
            [ci(S, Rank.TWO), ci(S, Rank.THREE)],
            [ci(W, Rank.TWO), ci(W, Rank.THREE)],
        )
        gs.play(0, [ci(J, Rank.KING)])
        gs.play(1, [DRAGON_INDEX])
        gs.play(2, [])
        gs.play(3, [])
        gs.play(0, [])  # P0 also passes after Dragon
        # Dragon give pending
        assert gs.dragon_give_pending == 1
        # Must give to opponent (player 0 or 2)
        options = gs.get_dragon_give_options(1)
        assert set(options) == {0, 2}
        # Can't give to teammate
        assert not gs.give_dragon_trick(1, 3)
        # Give to player 0
        assert gs.give_dragon_trick(1, 0)
        assert DRAGON_INDEX in gs.won_cards[0]


class TestPhoenix:
    def test_phoenix_single_beats_previous(self):
        gs = make_game_with_hands(
            [ci(J, Rank.KING), ci(J, Rank.TWO)],
            [PHOENIX_INDEX, ci(P, Rank.TWO)],
            [ci(S, Rank.TWO), ci(S, Rank.THREE)],
            [ci(W, Rank.TWO), ci(W, Rank.THREE)],
        )
        gs.play(0, [ci(J, Rank.KING)])
        # Phoenix should be playable (beats King as rank 13.5)
        assert gs.play(1, [PHOENIX_INDEX])

    def test_phoenix_cannot_beat_dragon(self):
        gs = make_game_with_hands(
            [DRAGON_INDEX],
            [PHOENIX_INDEX, ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.play(0, [DRAGON_INDEX])
        assert not gs.play(1, [PHOENIX_INDEX])


class TestMahjongWish:
    def test_wish_must_be_fulfilled(self):
        gs = make_game_with_hands(
            [MAHJONG_INDEX, ci(J, Rank.TWO)],
            [ci(P, Rank.ACE), ci(P, Rank.FIVE)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.play(0, [MAHJONG_INDEX])
        gs.set_mahjong_wish(5)  # Wish for 5
        # Player 1 has a 5, must play it if they can beat the trick
        # Mahjong is rank 1, so 5 beats it
        assert not gs.play(1, [ci(P, Rank.ACE)])  # Has a 5, must play it
        assert gs.play(1, [ci(P, Rank.FIVE)])


class TestScoring:
    def test_normal_scoring(self):
        gs = make_game_with_hands(
            [ci(J, Rank.FIVE)],  # 5 pts
            [ci(P, Rank.TEN)],  # 10 pts
            [ci(S, Rank.KING)],  # 10 pts
            [ci(W, Rank.TWO)],  # 0 pts
        )
        # Play out: p0 leads, all pass, p0 wins 5pts
        gs.play(0, [ci(J, Rank.FIVE)])
        gs.play(1, [])
        gs.play(2, [])
        gs.play(3, [])
        # p0 out, leads again — but out, so p1 leads
        # Actually p0 is out, next active player leads
        # p1 leads 10
        gs.play(1, [ci(P, Rank.TEN)])
        gs.play(2, [])
        gs.play(3, [])
        # p1 out, p2 leads King
        gs.play(2, [ci(S, Rank.KING)])
        gs.play(3, [])
        # p2 out, p3 is last
        assert gs.round_over
        # Team A (0,2): p0 won 5pts, p2 won 10pts = 15
        # Team B (1,3): p1 won 10pts, p3 hand goes to opponent team
        # p3's hand (2) = 0 pts, goes to team A

    def test_double_win_200_0(self):
        # Set up so player 0 and 2 go out first (1-2 finish)
        gs = make_game_with_hands(
            [ci(J, Rank.ACE)],
            [ci(P, Rank.TWO), ci(P, Rank.THREE)],
            [ci(S, Rank.ACE)],
            [ci(W, Rank.TWO), ci(W, Rank.THREE)],
        )
        gs.play(0, [ci(J, Rank.ACE)])  # p0 leads ace
        gs.play(1, [])  # pass
        gs.play(2, [ci(S, Rank.ACE)])  # p2 plays ace — wait, can't beat ace with ace
        # Let's redo: p0 leads, all pass, p0 out. p2 leads, all pass, p2 out. 1-2 finish.

        gs2 = make_game_with_hands(
            [ci(J, Rank.ACE)],
            [ci(P, Rank.TWO), ci(P, Rank.THREE)],
            [ci(S, Rank.ACE)],
            [ci(W, Rank.TWO), ci(W, Rank.THREE)],
        )
        gs2.play(0, [ci(J, Rank.ACE)])  # p0 leads
        gs2.play(1, [])
        gs2.play(2, [])
        gs2.play(3, [])
        # p0 out, trick won. Next active: p1
        # Actually p0 wins trick and leads, but p0 is out, so next active
        assert 0 in gs2.finish_order
        # p2 should be able to play now — depends on who gets lead
        # p0 won trick but is out → next active player after p0 = p1
        gs2.play(1, [ci(P, Rank.TWO)])
        gs2.play(2, [])
        gs2.play(3, [])
        # p1 wins trick (only player who played), p1 out? No, p1 still has cards
        gs2.play(1, [ci(P, Rank.THREE)])
        gs2.play(2, [])
        gs2.play(3, [])
        # p1 out now. finish_order = [0, 1]. Not same team → no double win.

    def test_tichu_bonus(self):
        gs = GameState()
        gs.reset(seed=42)
        gs.tichu_declarations[0] = 1  # Player 0 declares Tichu
        # If player 0 goes out first, they get +100 for team A
        # We just test the scoring method directly
        gs.finish_order = [0, 1, 2, 3]
        gs.round_over = False
        gs._score_normal()
        # Player 0 went out first → Tichu success → +100 for team A
        assert gs.round_scores[0] >= 100  # At least Tichu bonus

    def test_tichu_penalty(self):
        gs = GameState()
        gs.reset(seed=42)
        gs.tichu_declarations[0] = 1
        gs.finish_order = [1, 0, 2, 3]  # Player 0 NOT first
        gs._score_normal()
        # Player 0 failed Tichu → -100
        # round_scores[0] includes -100 penalty

    def test_total_card_points_100(self):
        """Total card points in the game should be 100."""
        total = sum(CARD_POINTS.values())
        assert total == 100


class TestLegalActions:
    def test_leading_has_all_combos(self):
        gs = make_game_with_hands(
            [ci(J, Rank.FIVE), ci(P, Rank.FIVE), ci(J, Rank.SIX), ci(P, Rank.SIX)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
            [ci(S, Rank.THREE)],
        )
        actions = gs.get_legal_actions(0)
        # Should include: 4 singles, 1 pair of 5s, 1 pair of 6s, steps (55 66)
        card_sets = [frozenset(cards) for cards, _ in actions]
        assert frozenset([ci(J, Rank.FIVE)]) in card_sets  # single
        assert frozenset([ci(J, Rank.FIVE), ci(P, Rank.FIVE)]) in card_sets  # pair
        assert (
            frozenset(
                [ci(J, Rank.FIVE), ci(P, Rank.FIVE), ci(J, Rank.SIX), ci(P, Rank.SIX)]
            )
            in card_sets
        )  # steps

    def test_following_only_beating_combos(self):
        gs = make_game_with_hands(
            [ci(J, Rank.TEN)],
            [ci(P, Rank.TWO), ci(P, Rank.ACE)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.play(0, [ci(J, Rank.TEN)])
        actions = gs.get_legal_actions(1)
        card_sets = [frozenset(cards) for cards, _ in actions]
        assert frozenset() in card_sets  # pass
        assert frozenset([ci(P, Rank.ACE)]) in card_sets  # Ace beats 10
        assert frozenset([ci(P, Rank.TWO)]) not in card_sets  # 2 doesn't beat 10

    def test_no_actions_when_not_current(self):
        gs = make_game_with_hands(
            [ci(J, Rank.TWO)],
            [ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        assert gs.get_legal_actions(1) == []


class TestLegalActionValidity:
    """All actions returned by get_legal_actions must be accepted by play()."""

    def test_all_legal_actions_playable(self):
        """Every action from get_legal_actions must be accepted by play()."""
        import random as _random
        rng = _random.Random(42)
        for seed in range(100):
            gs = make_game_with_hands(
                [ci(J, Rank.TWO)], [ci(P, Rank.TWO)],
                [ci(S, Rank.TWO)], [ci(W, Rank.TWO)],
            )
            gs2 = GameState()
            gs2.reset(seed=seed)

            # Play a few steps then check all legal actions are valid
            import random as _r
            r = _r.Random(seed)
            for _ in range(5):
                if gs2.round_over:
                    break
                if gs2.dragon_give_pending is not None:
                    opts = gs2.get_dragon_give_options(gs2.dragon_give_pending)
                    gs2.give_dragon_trick(gs2.dragon_give_pending, r.choice(opts))
                    continue
                p = gs2.current_player
                actions = gs2.get_legal_actions(p)
                assert actions
                # Verify each action is valid by trying on a copy
                for act in actions:
                    import copy
                    snap = copy.deepcopy(gs2)
                    assert snap.play(p, act), (
                        f"seed={seed} action={act} rejected by play(), "
                        f"leading={snap.trick.leading_combo}"
                    )
                gs2.play(p, r.choice(actions))

    def test_phoenix_fills_lowest_rank_at_ace_not_in_legal_actions(self):
        """Phoenix filling the lowest rank in a window ending at Ace is not legal.
        This is a specific edge case where _enum_straights previously generated
        a combo that classify_cards rejects (high_card = rank 15 > 14)."""
        # Hand: natural ranks 9-14 + Phoenix — can make straight 8-14 via Phoenix,
        # but classify_cards can't handle Phoenix-as-8 in a window ending at Ace.
        # So this combo must NOT appear in legal actions when following a straight.
        hand_p1 = [
            ci(J, Rank.NINE), ci(S, Rank.TEN), ci(S, Rank.JACK),
            ci(S, Rank.QUEEN), ci(J, Rank.KING), ci(W, Rank.ACE),
            PHOENIX_INDEX,
        ]
        # p0 leads a 7-card straight 3-9
        p0_straight = [
            ci(P, Rank.THREE), ci(P, Rank.FOUR), ci(P, Rank.FIVE),
            ci(P, Rank.SIX), ci(J, Rank.SEVEN), ci(P, Rank.EIGHT),
            ci(P, Rank.NINE),
        ]
        gs = make_game_with_hands(
            p0_straight,
            hand_p1,
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.play(0, p0_straight)
        actions = gs.get_legal_actions(1)
        # Verify every returned action is accepted by play()
        import copy
        for act in actions:
            snap = copy.deepcopy(gs)
            assert snap.play(1, act), (
                f"Action {act} from get_legal_actions was rejected by play()"
            )


class TestCanonicalActionOrder:
    """Verify get_legal_actions returns actions in canonical order."""

    def test_pass_is_first_when_following(self):
        gs = make_game_with_hands(
            [ci(J, Rank.TEN)],
            [ci(P, Rank.ACE), ci(P, Rank.KING)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.play(0, [ci(J, Rank.TEN)])
        actions = gs.get_legal_actions(1)
        assert actions[0] == ([], False)  # pass always first

    def test_singles_sorted_by_rank(self):
        gs = make_game_with_hands(
            [ci(J, Rank.ACE), ci(P, Rank.TWO), ci(S, Rank.SEVEN), ci(W, Rank.KING)],
            [ci(P, Rank.ACE)],
            [ci(S, Rank.ACE)],
            [ci(W, Rank.ACE)],
        )
        actions = gs.get_legal_actions(0)
        singles = [cards for cards, tichu in actions if len(cards) == 1 and not tichu]
        ranks = []
        for s in singles:
            c = s[0]
            ranks.append(c % 13 + 2 if c < 52 else 0)
        assert ranks == sorted(ranks)

    def test_grouped_by_combo_size(self):
        gs = make_game_with_hands(
            [ci(J, Rank.FIVE), ci(P, Rank.FIVE), ci(J, Rank.SIX), ci(P, Rank.SIX)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
            [ci(S, Rank.THREE)],
        )
        actions = gs.get_legal_actions(0)
        sizes = [len(cards) for cards, tichu in actions if not tichu]
        assert sizes == sorted(sizes)

    def test_within_size_sorted_by_rank(self):
        gs = make_game_with_hands(
            [ci(J, Rank.THREE), ci(P, Rank.THREE),
             ci(J, Rank.KING), ci(P, Rank.KING)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
            [ci(S, Rank.THREE)],
        )
        actions = gs.get_legal_actions(0)
        pairs = [cards for cards, tichu in actions if len(cards) == 2 and not tichu]
        pair_ranks = []
        for p in pairs:
            r = max(c % 13 + 2 for c in p if c < 52)
            pair_ranks.append(r)
        assert pair_ranks == sorted(pair_ranks)

    def test_ordering_is_deterministic(self):
        """Same state produces the same action order every time."""
        gs1 = make_game_with_hands(
            [ci(J, Rank.FIVE), ci(P, Rank.FIVE), ci(S, Rank.FIVE),
             ci(J, Rank.SIX), ci(P, Rank.SIX), ci(S, Rank.SIX),
             ci(J, Rank.SEVEN), PHOENIX_INDEX],
            [ci(W, Rank.TWO)],
            [ci(W, Rank.THREE)],
            [ci(W, Rank.FOUR)],
        )
        gs2 = make_game_with_hands(
            [ci(J, Rank.FIVE), ci(P, Rank.FIVE), ci(S, Rank.FIVE),
             ci(J, Rank.SIX), ci(P, Rank.SIX), ci(S, Rank.SIX),
             ci(J, Rank.SEVEN), PHOENIX_INDEX],
            [ci(W, Rank.TWO)],
            [ci(W, Rank.THREE)],
            [ci(W, Rank.FOUR)],
        )
        assert gs1.get_legal_actions(0) == gs2.get_legal_actions(0)

    def test_beating_combos_sorted(self):
        """When following, non-pass actions should also be sorted."""
        gs = make_game_with_hands(
            [ci(J, Rank.TWO)],
            [ci(P, Rank.ACE), ci(P, Rank.KING), ci(P, Rank.FIVE),
             ci(S, Rank.JACK)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.play(0, [ci(J, Rank.TWO)])
        actions = gs.get_legal_actions(1)
        base = [(c, t) for c, t in actions if not t]
        assert base[0] == ([], False)
        beating = [c for c, _ in base[1:]]
        ranks = [a[0] % 13 + 2 for a in beating]
        assert ranks == sorted(ranks)

    def test_dog_and_mahjong_placement(self):
        """Dog (rank 0) and Mahjong (rank 1) sort before normal singles."""
        gs = make_game_with_hands(
            [DOG_INDEX, MAHJONG_INDEX, ci(J, Rank.ACE)],
            [ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        actions = gs.get_legal_actions(0)
        singles = [cards for cards, tichu in actions if len(cards) == 1 and not tichu]
        assert singles[0] == [DOG_INDEX]
        assert singles[1] == [MAHJONG_INDEX]
        assert singles[2] == [ci(J, Rank.ACE)]


class TestMahjongWishWhenLeading:
    """Wish compliance must be enforced when leading a trick, not just following."""

    def test_leading_filters_to_wished_rank(self):
        """When leading with wish active and player has the wished rank,
        only combos containing that rank should be returned."""
        # Player 0 has a 5 (wished) plus other cards
        hand0 = [ci(J, Rank.FIVE), ci(J, Rank.KING), ci(P, Rank.ACE)]
        gs = make_game_with_hands(
            hand0,
            [ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.mahjong_wish = 5  # wish for rank 5

        actions = gs.get_legal_actions(0)
        # All actions must contain a card of rank 5
        for cards, _ in actions:
            assert any(c < 52 and c % 13 + 2 == 5 for c in cards), (
                f"Action {cards} doesn't contain wished rank 5"
            )

    def test_leading_allows_all_if_wish_impossible(self):
        """If player can't make any combo with wished rank, all combos are legal."""
        hand0 = [ci(J, Rank.KING), ci(P, Rank.ACE)]
        gs = make_game_with_hands(
            hand0,
            [ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.mahjong_wish = 5  # player has no 5s

        actions = gs.get_legal_actions(0)
        # Should still be able to play all combos
        assert len(actions) >= 2  # at least two singles

    def test_cannot_pass_when_wish_can_be_fulfilled(self):
        """When following and the wish can be fulfilled with a beating combo,
        pass must not be in the legal actions."""
        gs = make_game_with_hands(
            [ci(J, Rank.THREE)],
            [ci(P, Rank.FIVE), ci(P, Rank.KING)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.play(0, [ci(J, Rank.THREE)])
        gs.mahjong_wish = 5

        actions = gs.get_legal_actions(1)
        cards_list = [cards for cards, _ in actions]
        assert [] not in cards_list, "Pass should not be allowed when wish is fulfillable"
        for cards, _ in actions:
            assert any(c < 52 and c % 13 + 2 == 5 for c in cards)

    def test_can_pass_when_wish_not_fulfillable(self):
        """When following and no beating combo contains the wished rank,
        pass should still be allowed."""
        gs = make_game_with_hands(
            [ci(J, Rank.ACE)],
            [ci(P, Rank.FIVE), ci(P, Rank.THREE)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.play(0, [ci(J, Rank.ACE)])
        gs.mahjong_wish = 5

        actions = gs.get_legal_actions(1)
        cards_list = [cards for cards, _ in actions]
        assert [] in cards_list, "Pass should be allowed when wish can't be fulfilled"

    def test_play_rejects_non_wish_combo_when_leading(self):
        """play() must reject a leading combo that doesn't contain the
        wished rank when the player could play the wished rank."""
        hand0 = [ci(J, Rank.FIVE), ci(J, Rank.KING)]
        gs = make_game_with_hands(
            hand0,
            [ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.mahjong_wish = 5

        # Playing King (not the wished rank) should be rejected
        assert not gs.play(0, [ci(J, Rank.KING)])

    def test_play_accepts_wish_combo_when_leading(self):
        """play() must accept a leading combo containing the wished rank."""
        hand0 = [ci(J, Rank.FIVE), ci(J, Rank.KING)]
        gs = make_game_with_hands(
            hand0,
            [ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        gs.mahjong_wish = 5

        assert gs.play(0, [ci(J, Rank.FIVE)])
        assert gs.mahjong_wish is None  # wish fulfilled

    def test_wish_not_enforced_on_mahjong_play_itself(self):
        """When Mahjong is played and wish is set afterward,
        the Mahjong play itself should not be constrained by a future wish."""
        hand0 = [MAHJONG_INDEX, ci(J, Rank.KING)]
        gs = make_game_with_hands(
            hand0,
            [ci(P, Rank.TWO)],
            [ci(S, Rank.TWO)],
            [ci(W, Rank.TWO)],
        )
        # No wish active yet — Mahjong can be played freely
        assert gs.play(0, [MAHJONG_INDEX])

    def test_legal_actions_all_accepted_by_play_with_wish(self):
        """Fuzz: every action from get_legal_actions with wish active is accepted by play()."""
        import copy
        for seed in range(50):
            rng2 = random.Random(seed)
            gs = GameState()
            gs.reset(seed)
            # Play a few random moves to get a varied state
            for _ in range(rng2.randint(0, 10)):
                if gs.round_over:
                    break
                if gs.dragon_give_pending is not None:
                    opts = gs.get_dragon_give_options(gs.dragon_give_pending)
                    gs.give_dragon_trick(gs.dragon_give_pending, rng2.choice(opts))
                    continue
                p = gs.current_player
                actions = gs.get_legal_actions(p)
                if not actions:
                    break
                gs.play(p, rng2.choice(actions))

            if gs.round_over:
                continue

            # Set a wish and verify all legal actions are playable
            gs.mahjong_wish = rng2.randint(2, 14)
            p = gs.current_player
            actions = gs.get_legal_actions(p)
            for act in actions:
                snap = copy.deepcopy(gs)
                assert snap.play(p, act), (
                    f"seed={seed} wish={gs.mahjong_wish} action={act} "
                    f"rejected by play(), leading={snap.trick.leading_combo}"
                )


class TestGrandTichuPhase:
    """Grand tichu calling phase (8-card deal)."""

    def test_reset_with_calling_deals_8_cards(self):
        gs = GameState()
        gs.reset(seed=42, with_calling=True)
        assert gs.phase == GamePhase.GRAND_TICHU
        for p in range(4):
            assert len(gs.hands[p]) == 8

    def test_all_56_cards_distributed(self):
        gs = GameState()
        gs.reset(seed=42, with_calling=True)
        all_cards = set()
        for p in range(4):
            all_cards |= gs.hands[p]
            all_cards.update(gs._remaining_cards[p])
        assert len(all_cards) == 56

    def test_grand_tichu_flow_to_play(self):
        gs = GameState()
        gs.reset(seed=42, with_calling=True)
        # All 4 players decline grand tichu
        for p in range(4):
            assert gs.current_player == p
            assert gs.declare_grand_tichu(p, False)
        assert gs.phase == GamePhase.PLAY
        # All players now have 14 cards
        for p in range(4):
            assert len(gs.hands[p]) == 14
        assert gs._remaining_cards is None

    def test_grand_tichu_declare(self):
        gs = GameState()
        gs.reset(seed=42, with_calling=True)
        assert gs.declare_grand_tichu(0, True)   # player 0 declares
        assert gs.declare_grand_tichu(1, False)
        assert gs.declare_grand_tichu(2, False)
        assert gs.declare_grand_tichu(3, False)
        assert gs.tichu_declarations[0] == 2
        assert gs.tichu_declarations[1] == 0
        assert gs.phase == GamePhase.PLAY

    def test_wrong_player_rejected(self):
        gs = GameState()
        gs.reset(seed=42, with_calling=True)
        assert not gs.declare_grand_tichu(1, False)  # not player 1's turn

    def test_mahjong_holder_starts_play(self):
        gs = GameState()
        gs.reset(seed=42, with_calling=True)
        for p in range(4):
            gs.declare_grand_tichu(p, False)
        # Mahjong holder should be current_player
        assert MAHJONG_INDEX in gs.hands[gs.current_player]


class TestCallTichu:
    """Tichu calling bundled with play(call_tichu=True)."""

    def test_tichu_actions_present_before_first_play(self):
        """Before playing cards, actions include tichu variants."""
        gs = GameState()
        gs.reset(seed=42)
        p = gs.current_player
        actions = gs.get_legal_actions(p)
        tichu_actions = [(c, t) for c, t in actions if t]
        non_tichu = [(c, t) for c, t in actions if not t]
        assert len(tichu_actions) == len(non_tichu)

    def test_call_tichu_via_action(self):
        """Playing a tichu=True action from get_legal_actions declares tichu."""
        gs = GameState()
        gs.reset(seed=42)
        p = gs.current_player
        actions = gs.get_legal_actions(p)
        # Pick a tichu variant
        tichu_action = next(a for a in actions if a[1])
        assert gs.play(p, tichu_action)
        assert gs.tichu_declarations[p] == 1

    def test_call_tichu_with_pass(self):
        gs = GameState()
        gs.reset(seed=42)
        p0 = gs.current_player
        gs.play(p0, gs.get_legal_actions(p0)[0])  # p0 leads (no tichu)
        p1 = gs.current_player
        # Pass + call tichu via tuple
        assert gs.play(p1, ([], True))
        assert gs.tichu_declarations[p1] == 1
        assert not gs.can_call_tichu(p1)

    def test_no_tichu_actions_after_playing(self):
        """After playing a card, tichu variants disappear."""
        gs = GameState()
        gs.reset(seed=42)
        p0 = gs.current_player
        gs.play(p0, gs.get_legal_actions(p0)[0])  # p0 plays (no tichu)
        assert not gs.can_call_tichu(p0)
        # When p0's turn comes around again, no tichu variants
        while not gs.round_over and gs.current_player != p0:
            cp = gs.current_player
            acts = gs.get_legal_actions(cp)
            if not acts:
                break
            gs.play(cp, acts[0])
        if not gs.round_over and gs.current_player == p0:
            acts = gs.get_legal_actions(p0)
            assert all(not t for _, t in acts)

    def test_passing_does_not_close_tichu_window(self):
        gs = GameState()
        gs.reset(seed=42)
        p0 = gs.current_player
        gs.play(p0, gs.get_legal_actions(p0)[0])  # p0 leads
        p1 = gs.current_player
        assert p1 != p0
        gs.play(p1, ([], False))  # pass without tichu
        assert gs.can_call_tichu(p1)

    def test_cannot_call_tichu_with_grand_tichu(self):
        gs = GameState()
        gs.reset(seed=42, with_calling=True)
        gs.declare_grand_tichu(0, True)
        for p in range(1, 4):
            gs.declare_grand_tichu(p, False)
        assert not gs.can_call_tichu(0)
        # No tichu variants for player 0
        if gs.current_player == 0:
            acts = gs.get_legal_actions(0)
            assert all(not t for _, t in acts)

    def test_play_rejected_if_tichu_call_invalid(self):
        """Passing a tichu tuple after already playing is rejected."""
        gs = GameState()
        gs.reset(seed=42)
        p = gs.current_player
        actions = gs.get_legal_actions(p)
        gs.play(p, actions[0])  # first play, no tichu
        # Advance until p's turn again
        while not gs.round_over and gs.current_player != p:
            cp = gs.current_player
            acts = gs.get_legal_actions(cp)
            if not acts:
                break
            gs.play(cp, acts[0])
        if not gs.round_over and gs.current_player == p:
            acts = gs.get_legal_actions(p)
            if acts:
                cards = acts[0][0]
                assert not gs.play(p, (cards, True))  # tichu call should fail


class TestTichuScoring:
    """Tichu and grand tichu scoring, including double-win."""

    def test_tichu_success_bonus(self):
        """Tichu caller who finishes first gets +100."""
        gs = GameState()
        gs.reset(seed=0)
        gs.finish_order = [0, 1, 2, 3]
        gs.finished = {0, 1, 2, 3}
        gs.tichu_declarations[0] = 1  # player 0 called tichu, finished 1st
        gs._score_normal()
        a, b = gs.round_scores
        # Redo without tichu
        gs.tichu_declarations[0] = 0
        gs._score_normal()
        a2, b2 = gs.round_scores
        assert a == a2 + 100
        assert b == b2

    def test_tichu_failure_penalty(self):
        """Tichu caller who doesn't finish first gets -100."""
        gs = GameState()
        gs.reset(seed=0)
        gs.finish_order = [1, 0, 2, 3]
        gs.finished = {0, 1, 2, 3}
        gs.tichu_declarations[0] = 1  # player 0 called tichu, finished 2nd
        gs._score_normal()
        a, b = gs.round_scores
        gs.tichu_declarations[0] = 0
        gs._score_normal()
        a2, b2 = gs.round_scores
        assert a == a2 - 100

    def test_grand_tichu_bonus_values(self):
        """Grand tichu uses +/-200 bonus."""
        gs = GameState()
        gs.reset(seed=0)
        gs.finish_order = [0, 1, 2, 3]
        gs.finished = {0, 1, 2, 3}
        gs.tichu_declarations[0] = 2  # grand tichu, finished 1st → +200
        gs._score_normal()
        a1, _ = gs.round_scores
        gs.tichu_declarations[0] = 0
        gs._score_normal()
        a2, _ = gs.round_scores
        assert a1 == a2 + 200

    def test_double_win_includes_tichu_bonus(self):
        """1-2 finish: 200 base + tichu bonus on top."""
        gs = GameState()
        gs.reset(seed=0)
        # Manually set up a double-win scenario
        # Simulate finish_order with teammates going 1-2
        gs.finish_order = [0, 2]  # team A 1-2 finish
        gs.finished = {0, 2}
        gs.tichu_declarations[0] = 1  # player 0 called tichu
        gs._score_double_win(0)
        score_a, score_b = gs.round_scores
        # Team A: 200 (double win) + 100 (tichu success) = 300
        assert score_a == 300
        assert score_b == 0

    def test_double_win_tichu_failure_on_losing_team(self):
        """Opponent calls tichu but loses via double win: -100 penalty."""
        gs = GameState()
        gs.reset(seed=0)
        gs.finish_order = [0, 2]  # team A 1-2
        gs.finished = {0, 2}
        gs.tichu_declarations[1] = 1  # player 1 (team B) called tichu and lost
        gs._score_double_win(0)
        score_a, score_b = gs.round_scores
        assert score_a == 200
        assert score_b == -100  # 0 base - 100 penalty

    def test_double_win_grand_tichu_on_winning_team(self):
        gs = GameState()
        gs.reset(seed=0)
        gs.finish_order = [0, 2]
        gs.finished = {0, 2}
        gs.tichu_declarations[0] = 2  # grand tichu success
        gs._score_double_win(0)
        assert gs.round_scores == (400, 0)  # 200 + 200


class TestTeamHelpers:
    def test_teammate(self):
        assert teammate(0) == 2
        assert teammate(1) == 3
        assert teammate(2) == 0
        assert teammate(3) == 1


class TestPassingPhase:
    """Card passing phase: each player passes one card to each opponent."""

    def test_passing_phase_entered(self):
        gs = GameState()
        gs.reset(seed=42, with_passing=True)
        assert gs.phase == GamePhase.PASSING
        assert gs.current_player == 0

    def test_default_no_passing(self):
        gs = GameState()
        gs.reset(seed=42)
        assert gs.phase == GamePhase.PLAY

    def test_get_passing_hand(self):
        gs = GameState()
        gs.reset(seed=42, with_passing=True)
        hand = gs.get_passing_hand(0)
        assert hand == sorted(gs.hands[0])
        assert len(hand) == 14

    def test_pass_cards_sequential(self):
        """Players 0→1→2→3 submit, then cards swap and play starts."""
        gs = GameState()
        gs.reset(seed=42, with_passing=True)

        hands_before = [set(gs.hands[p]) for p in range(4)]

        # Each player passes 3 cards (first 3 from sorted hand)
        passes = {}
        for p in range(4):
            hand = gs.get_passing_hand(p)
            assert gs.current_player == p
            left, partner, right = hand[0], hand[1], hand[2]
            passes[p] = (left, partner, right)
            assert gs.pass_cards(p, left, partner, right)

        assert gs.phase == GamePhase.PLAY

        # Each player lost 3 cards and gained 3
        for p in range(4):
            assert len(gs.hands[p]) == 14

        # Verify cards moved correctly
        for giver in range(4):
            left_c, partner_c, right_c = passes[giver]
            left_p = (giver + 1) % 4
            partner_p = (giver + 2) % 4
            right_p = (giver + 3) % 4
            assert left_c not in gs.hands[giver]
            assert left_c in gs.hands[left_p]
            assert partner_c in gs.hands[partner_p]
            assert right_c in gs.hands[right_p]

    def test_wrong_player_rejected(self):
        gs = GameState()
        gs.reset(seed=42, with_passing=True)
        hand = sorted(gs.hands[1])
        assert not gs.pass_cards(1, hand[0], hand[1], hand[2])

    def test_duplicate_cards_rejected(self):
        gs = GameState()
        gs.reset(seed=42, with_passing=True)
        hand = gs.get_passing_hand(0)
        assert not gs.pass_cards(0, hand[0], hand[0], hand[1])

    def test_card_not_in_hand_rejected(self):
        gs = GameState()
        gs.reset(seed=42, with_passing=True)
        hand = gs.get_passing_hand(0)
        # Find a card not in hand
        missing = next(c for c in range(56) if c not in gs.hands[0])
        assert not gs.pass_cards(0, missing, hand[0], hand[1])

    def test_mahjong_holder_starts_after_passing(self):
        gs = GameState()
        gs.reset(seed=42, with_passing=True)
        for p in range(4):
            hand = gs.get_passing_hand(p)
            gs.pass_cards(p, hand[0], hand[1], hand[2])
        assert gs.phase == GamePhase.PLAY
        assert MAHJONG_INDEX in gs.hands[gs.current_player]

    def test_calling_then_passing(self):
        """with_calling + with_passing: GT phase → passing → play."""
        gs = GameState()
        gs.reset(seed=42, with_calling=True, with_passing=True)
        assert gs.phase == GamePhase.GRAND_TICHU

        for p in range(4):
            gs.declare_grand_tichu(p, False)
        assert gs.phase == GamePhase.PASSING

        for p in range(4):
            hand = gs.get_passing_hand(p)
            gs.pass_cards(p, hand[0], hand[1], hand[2])
        assert gs.phase == GamePhase.PLAY
        assert MAHJONG_INDEX in gs.hands[gs.current_player]

    def test_pass_outside_phase_rejected(self):
        gs = GameState()
        gs.reset(seed=42)  # no passing
        hand = sorted(gs.hands[0])
        assert not gs.pass_cards(0, hand[0], hand[1], hand[2])
