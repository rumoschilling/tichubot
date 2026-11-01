"""Unit tests for history data integrity, reconstruction, and transition correctness."""

import numpy as np
import torch
import torch.nn as nn
import pytest
from tichu_env.game_state import GameState
from tichu_env.encoding import encode_history, PLAY_TOKEN_SIZE, STATIC_OBS_SIZE
from training.rollout_history import _collect_batched

class DummyHistoryPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(STATIC_OBS_SIZE, 64))
        self.pass_source_embed = nn.Embedding(57, 8)
        self.pass_source_proj = nn.Linear(24, 64)
        self.pass_target_embed = nn.Embedding(57, 8)
        self.pass_target_proj = nn.Linear(24, 64)
        
        class DummyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.output_dim = 64
            def forward(self, h, hl):
                return torch.zeros((h.shape[0], 64))
        
        self.history_encoder = DummyEncoder()
        self.history_fuse = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)
        self.wish_head = nn.Sequential(nn.Linear(64, 14))
        self.dragon_head = nn.Sequential(nn.Linear(64, 2))

    def act(self, obs, legal, n_legal, hist, hist_len, ps, pt):
        B = obs.shape[0]
        # Return first action (index 0)
        return torch.zeros(B, dtype=torch.long), torch.zeros(B), torch.zeros(B)

    def fused_features(self, obs, hist, hist_len, ps, pt):
        return torch.zeros((obs.shape[0], 64))

    def get_mahjong_wish_logits(self, x): return torch.zeros((x.shape[0], 14))
    def get_dragon_gift_logits(self, x): return torch.zeros((x.shape[0], 2))

def test_history_token_clearing():
    """Verify that tokens are cleared before reuse to prevent bit leakage."""
    gs = GameState(track_history=True)
    gs.reset()
    
    # Manually populate some tokens with 1.0s
    gs.history_tokens.fill(1.0)
    
    # Record a play (first play of game)
    gs._record_play(player=0, card_indices=[52], trick_num=0, is_opener=True)
    
    token = gs.history_tokens[0]
    # Bits 0-51 (normal cards) should be 0.0, Bit 52 (Mahjong) should be 1.0
    assert token[52] == 1.0
    assert np.sum(token[0:52]) == 0.0, "Token cards bit leakage! Not cleared."
    
    # Record a pass
    gs._record_play(player=1, card_indices=[], trick_num=0, is_opener=False)
    token1 = gs.history_tokens[1]
    assert token1[56] == 1.0 # is_pass
    assert np.sum(token1[0:56]) == 0.0, "Pass token leaked card bits."

def test_no_double_play():
    """Verify that _collect_batched only advances the game state ONCE per step."""
    device = torch.device("cpu")
    rollout, _ = _collect_batched(
        DummyHistoryPolicy(),
        n_games=1,
        device=device,
        use_rollout_bucketing=False,
    )

    recorded_steps = len(rollout["obs"])
    assert recorded_steps > 0
    assert len(rollout["history_lens"]) == recorded_steps
    assert len(rollout["legal_lens"]) == recorded_steps

    # For one game, each stored transition is recorded before exactly one
    # gs.play() call. Therefore the pre-action history length should increase
    # by one on every transition. A double-play bug shows up as skipped lengths.
    expected_lens = np.arange(recorded_steps, dtype=np.int32)
    np.testing.assert_array_equal(rollout["history_lens"], expected_lens)
    assert np.all(rollout["action_idxs"] < rollout["legal_lens"])

def test_vectorized_history_reconstruction():
    """Verify that vectorized buffer reconstruction matches reference encoder."""
    gs = GameState(track_history=True)
    gs.reset()
    # Play a few moves to build up history
    for _ in range(5):
        if gs.round_over: break
        legal = gs.get_legal_actions(gs.current_player)
        gs.play(gs.current_player, legal[0])
    
    num_steps = len(gs.play_log)
    if num_steps < 3: return # Skip if too short
    
    p_idxs = np.array([gs.play_log[i][0] for i in range(num_steps)], dtype=np.int32)
    h_lens = np.arange(num_steps, dtype=np.int32) 
    
    max_h = max(int(h_lens.max()), 1)
    history_arr = np.zeros((num_steps, max_h, 63), dtype=np.float16)
    
    # Vectorized rebuild logic (STRICT COPY of current rollout_history.py logic)
    row_ids = np.arange(num_steps, dtype=np.int32)[:, None]
    col_ids = np.arange(max_h, dtype=np.int32)[None, :]
    valid = col_ids < h_lens[:, None]
    step_ids, token_ids = np.nonzero(valid)
    
    if len(step_ids) > 0:
        history_arr[step_ids, token_ids, 4:63] = gs.history_tokens[token_ids]
        abs_players = gs.player_indices[token_ids]
        observers = p_idxs[step_ids]
        rel_players = (abs_players - observers) % 4
        history_arr[step_ids, token_ids, rel_players] = 1.0
    
    for step in range(num_steps):
        observer = p_idxs[step]
        actual = history_arr[step, :h_lens[step]].astype(np.float32)
        expected, _ = encode_history(gs, observer, limit=h_lens[step])
        expected = expected[:h_lens[step]]
        np.testing.assert_allclose(actual, expected, atol=1e-5)

if __name__ == "__main__":
    test_history_token_clearing()
    test_no_double_play()
    test_vectorized_history_reconstruction()
