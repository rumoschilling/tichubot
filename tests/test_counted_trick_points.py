import numpy as np
import torch

from tichu_env.cards import (
    DRAGON_INDEX,
    PHOENIX_INDEX,
    Rank,
    Suit,
    card_index,
)
from tichu_env.encoding import (
    COUNTED_TRICK_POINTS_OBS_SIZE,
    STATIC_OBS_SIZE,
    encode_static,
    encode_static_counted_trick_points,
)
from tichu_env.game_state import GameState
from training.model_embed import TichuPolicyEmbed
from training.checkpoints import load_policy_checkpoint
from training.config import TrainConfig
from training.rollout_embed import collect_rollouts_embed


def test_counted_points_are_appended_in_observer_relative_order():
    gs = GameState()
    gs.reset()
    gs.won_cards = [
        [card_index(Suit.JADE, Rank.FIVE)],
        [DRAGON_INDEX],
        [PHOENIX_INDEX],
        [card_index(Suit.PAGODA, Rank.TEN)],
    ]

    base = encode_static(gs, 3)
    obs = encode_static_counted_trick_points(gs, 3)

    assert obs.shape == (COUNTED_TRICK_POINTS_OBS_SIZE,)
    np.testing.assert_array_equal(obs[:STATIC_OBS_SIZE], base)
    np.testing.assert_allclose(obs[STATIC_OBS_SIZE:], [0.10, 0.05, 0.25, -0.25])


def test_counted_points_encoder_clears_and_reuses_output_buffer():
    gs = GameState()
    gs.reset()
    gs.won_cards[2] = [DRAGON_INDEX]
    out = np.full(COUNTED_TRICK_POINTS_OBS_SIZE, 7.0, dtype=np.float32)

    returned = encode_static_counted_trick_points(gs, 0, out=out)

    assert returned is out
    np.testing.assert_allclose(out[STATIC_OBS_SIZE:], [0.0, 0.0, 0.25, 0.0])


def test_embed_rollout_accepts_counted_point_observations():
    policy = TichuPolicyEmbed(
        obs_size=COUNTED_TRICK_POINTS_OBS_SIZE,
        hidden_dim=32,
        n_layers=1,
        embed_dim=8,
    )

    rollout = collect_rollouts_embed(
        policy,
        2,
        torch.device("cpu"),
        counted_trick_points=True,
    )

    assert rollout["obs"].shape[1] == COUNTED_TRICK_POINTS_OBS_SIZE
    assert np.any(rollout["obs"][:, STATIC_OBS_SIZE:] != 0.0)


def test_counted_points_checkpoint_reconstructs_input_size(tmp_path):
    cfg = TrainConfig(
        hidden_dim=32,
        n_layers=1,
        embed_dim=8,
        counted_trick_points=True,
    )
    policy = TichuPolicyEmbed(
        obs_size=COUNTED_TRICK_POINTS_OBS_SIZE,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        embed_dim=cfg.embed_dim,
    )
    path = tmp_path / "counted_points.pt"
    torch.save(
        {"games_played": 0, "model": policy.state_dict(), "config": cfg},
        path,
    )

    loaded = load_policy_checkpoint(path)

    assert loaded.kind == "embed"
    assert loaded.policy.obs_size == COUNTED_TRICK_POINTS_OBS_SIZE
    assert loaded.policy.trunk[0].in_features == COUNTED_TRICK_POINTS_OBS_SIZE
