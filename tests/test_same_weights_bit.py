"""Tests for the partner same-weights bit (obs[195]) and legacy checkpoint migration."""

import numpy as np
import torch

from tichu_env.game_state import GameState
from tichu_env.encoding import encode_static, STATIC_OBS_SIZE
from training.checkpoints import migrate_legacy_obs_size
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder

BIT = 195


def test_obs_size_is_196():
    assert STATIC_OBS_SIZE == 196
    gs = GameState()
    gs.reset()
    assert encode_static(gs, 0).shape == (196,)


def test_selfplay_default_bit_is_one():
    gs = GameState()
    gs.reset()
    for p in range(4):
        assert encode_static(gs, p)[BIT] == 1.0


def test_team_eval_ids_keep_partner_bit_one():
    gs = GameState()
    gs.player_model_ids = [0, 1, 0, 1]
    gs.reset()
    for p in range(4):
        assert encode_static(gs, p)[BIT] == 1.0


def test_all_foreign_ids_zero_the_bit():
    gs = GameState()
    gs.player_model_ids = [0, 1, 2, 3]
    gs.reset()
    for p in range(4):
        assert encode_static(gs, p)[BIT] == 0.0


def test_mixed_ids_follow_partner_relation():
    gs = GameState()
    gs.player_model_ids = [0, 0, 1, 0]
    gs.reset()
    assert encode_static(gs, 0)[BIT] == 0.0  # partner 2 runs id 1
    assert encode_static(gs, 2)[BIT] == 0.0
    assert encode_static(gs, 1)[BIT] == 1.0  # partner 3 runs id 0
    assert encode_static(gs, 3)[BIT] == 1.0


def test_reset_preserves_model_ids():
    gs = GameState()
    gs.player_model_ids = [0, 1, 2, 3]
    gs.reset()
    gs.reset()
    assert gs.player_model_ids == [0, 1, 2, 3]


def _tiny_model():
    enc = TransformerHistoryEncoder(d_model=32, nhead=2, n_layers=1, ffn_dim=64)
    return TichuPolicyHistory(enc, hidden_dim=64, n_layers=2, embed_dim=32)


def test_legacy_migration_pads_and_is_behavior_neutral():
    torch.manual_seed(0)
    model = _tiny_model()

    # Fabricate a 195-era checkpoint by chopping the obs column.
    legacy = {
        k: (v[:, :195].clone() if v.dim() == 2 and v.shape[1] == STATIC_OBS_SIZE
            else v.clone())
        for k, v in model.state_dict().items()
    }
    assert migrate_legacy_obs_size(legacy)
    assert not migrate_legacy_obs_size(legacy)  # idempotent

    migrated = _tiny_model()
    migrated.load_state_dict(legacy)
    migrated.eval()

    # The padded column is zero, so toggling the new bit must not change
    # anything the model computes.
    gs = GameState()
    gs.reset()
    obs_on = encode_static(gs, 0)
    obs_off = obs_on.copy()
    obs_off[BIT] = 0.0

    def value(obs_np):
        obs = torch.as_tensor(obs_np[None, :])
        hist = torch.zeros((1, 1, 63))
        hist_len = torch.zeros(1, dtype=torch.int32)
        ps = torch.full((1, 3), 56, dtype=torch.int64)
        pt = torch.full((1, 3), 56, dtype=torch.int64)
        with torch.no_grad():
            fused = migrated.fused_features(obs, hist, hist_len, ps, pt)
            return migrated.value_head(fused).item()

    assert value(obs_on) == value(obs_off)
