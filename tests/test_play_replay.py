import random

import pytest
import torch

import llm_play
from tichu_env.cards import DOG_INDEX, PHOENIX_INDEX, Rank, Suit, card_index
from tichu_env.encoding import COUNTED_TRICK_POINTS_OBS_SIZE
from tichu_env.game_state import GamePhase, GameState
from training.model import TichuPolicy
from training.model_embed import TichuPolicyEmbed
from training.play import ai_select_dragon_recipient, run_replay_passing_phase


def _initial_hands(seed: int = 17) -> list[list[int]]:
    gs = GameState()
    gs.reset(seed=seed, with_passing=True)
    return [sorted(hand) for hand in gs.hands]


def _replay_state(initial_hands: list[list[int]]) -> GameState:
    gs = GameState()
    gs.reset(seed=999, with_passing=True)
    gs.hands = [set(hand) for hand in initial_hands]
    gs._invalidate_cache()
    return gs


def _recorded_passes(initial_hands: list[list[int]]) -> dict[int, tuple[int, int, int]]:
    return {player: tuple(hand[:3]) for player, hand in enumerate(initial_hands)}


def test_deployed_dragon_choice_uses_history_specialist_head():
    class RightRecipientPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dragon_head = torch.nn.Identity()

        def fused_features(self, obs, history, history_len, pass_src, pass_tgt):
            return torch.zeros((obs.shape[0], 4))

        def get_dragon_gift_logits(self, features):
            return torch.tensor([[-1_000.0, 1_000.0]])

    gs = GameState(track_history=True)
    gs.dragon_give_pending = 1

    recipient = ai_select_dragon_recipient(
        RightRecipientPolicy(),
        gs,
        player=1,
        device=torch.device("cpu"),
        rng=random.Random(1),
        kind="history",
    )

    assert recipient == 0


def test_replay_reuses_recorded_opponent_passes_across_runs():
    initial_hands = _initial_hands()
    recorded = _recorded_passes(initial_hands)

    first = run_replay_passing_phase(
        _replay_state(initial_hands),
        pass_policy=None,
        device=torch.device("cpu"),
        replayed_player=0,
        recorded_passes=recorded,
        rng=random.Random(1),
        standardize_opponent_passes=True,
        announce=False,
    )
    second = run_replay_passing_phase(
        _replay_state(initial_hands),
        pass_policy=None,
        device=torch.device("cpu"),
        replayed_player=0,
        recorded_passes=recorded,
        rng=random.Random(2),
        standardize_opponent_passes=True,
        announce=False,
    )

    for player in (1, 2, 3):
        assert first[player] == recorded[player]
        assert second[player] == recorded[player]


def test_standardized_replay_samples_only_the_replayed_seat(monkeypatch):
    initial_hands = _initial_hands()
    recorded = _recorded_passes(initial_hands)
    sampled: list[int] = []

    def fixed_ai_pass(pass_policy, gs, player, device, rng):
        """A stand-in that records every seat asked to sample a new pass."""
        sampled.append(player)
        choice = tuple(gs.get_passing_hand(player)[-3:])
        assert gs.pass_cards(player, *choice)
        return choice

    monkeypatch.setattr("training.play.ai_pass", fixed_ai_pass)
    applied = run_replay_passing_phase(
        _replay_state(initial_hands),
        pass_policy=None,
        device=torch.device("cpu"),
        replayed_player=0,
        recorded_passes=recorded,
        rng=random.Random(3),
        standardize_opponent_passes=True,
        announce=False,
    )

    assert sampled == [0]
    assert applied[0] != recorded[0]
    assert all(applied[player] == recorded[player] for player in (1, 2, 3))


def test_standardized_replay_rejects_incomplete_opponent_records():
    initial_hands = _initial_hands()

    with pytest.raises(ValueError, match=r"missing P2, P3"):
        run_replay_passing_phase(
            _replay_state(initial_hands),
            pass_policy=None,
            device=torch.device("cpu"),
            replayed_player=0,
            recorded_passes={1: tuple(initial_hands[1][:3])},
            rng=random.Random(3),
            standardize_opponent_passes=True,
            announce=False,
        )


def test_standardized_replay_passes_remain_legal_and_directed():
    initial_hands = _initial_hands()
    recorded = _recorded_passes(initial_hands)
    gs = _replay_state(initial_hands)

    applied = run_replay_passing_phase(
        gs,
        pass_policy=None,
        device=torch.device("cpu"),
        replayed_player=0,
        recorded_passes=recorded,
        rng=random.Random(3),
        standardize_opponent_passes=True,
        announce=False,
    )

    assert gs.phase == GamePhase.PLAY
    assert [len(hand) for hand in gs.hands] == [14, 14, 14, 14]
    assert len(set().union(*gs.hands)) == 56
    for player, (left, partner, right) in applied.items():
        assert len({left, partner, right}) == 3
        assert {left, partner, right}.issubset(initial_hands[player])
        assert gs._outgoing_pass_target[player] == {
            (player + 1) % 4: left,
            (player + 2) % 4: partner,
            (player + 3) % 4: right,
        }


def test_llm_play_supports_fixed_policy_inference():
    gs = GameState()
    gs.reset(seed=31)
    player = gs.current_player
    before = len(gs.hands[player])

    action = llm_play.ai_play(
        TichuPolicy(hidden_dim=16, n_layers=1),
        gs,
        player,
        "fixed",
        random.Random(2),
    )

    assert len(action) == 2
    assert len(gs.hands[player]) < before


def test_llm_play_supports_counted_trick_points_policy():
    gs = GameState()
    gs.reset(seed=37)
    player = gs.current_player
    policy = TichuPolicyEmbed(
        obs_size=COUNTED_TRICK_POINTS_OBS_SIZE,
        hidden_dim=16,
        n_layers=1,
        embed_dim=8,
        action_hidden_dim=16,
    )

    action = llm_play.ai_play(
        policy,
        gs,
        player,
        "embed",
        random.Random(3),
    )

    assert len(action) == 3
    assert gs.current_player != player or player in gs.finish_order


def test_llm_play_preserves_selected_phoenix_component(monkeypatch):
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
    gs.current_player = 0
    gs.hands = [
        set(cards + [DOG_INDEX]),
        {reserve[0]},
        {reserve[1]},
        {reserve[2]},
    ]
    gs._invalidate_cache()
    selected = (cards, False, True)
    monkeypatch.setattr(llm_play, "ai_select_action", lambda *args: selected)

    action = llm_play.ai_play(None, gs, 0, "embed", random.Random(4))

    assert action == selected
    assert gs.trick.leading_combo is not None
    assert gs.trick.leading_combo.phoenix_as == 7
