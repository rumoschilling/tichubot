"""Regression tests for rollout/PPO optimization correctness.

These tests compare optimized jagged-buffer paths against simple reference
logic. They are intentionally small, but they cover the failure modes that
would silently damage learning: wrong legal-action offsets, history/observer
misalignment, padding-dependent transformer outputs, and lazy/eager PPO drift.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from tichu_env.cards import MAHJONG_INDEX
from tichu_env.encoding import (
    PLAY_TOKEN_SIZE,
    STATIC_OBS_SIZE,
    encode_history,
    encode_pass_source_ids,
    encode_pass_target_ids,
    encode_static,
)
from tichu_env.game_state import GameState
from training.config import TrainConfig
from training.model_embed import ACTION_DIM
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder
from training.optimizer_batch_experiment import (
    _holm_adjust,
    _one_sided_sign_flip_pvalue,
    _paired_optimizer_log_time_ratios,
    _paired_log_time_differences,
    _summarize_optimizer_runtime,
)
from training.ppo_history import (
    _legal_offsets,
    _pad_legal_eager,
    _pad_legal_lazy,
    _rebuild_shared_history_numpy,
    ppo_update_history,
)
from training.rollout_history import (
    _collect_batched,
    _compile_shared_history,
    _rebuild_history_buffer,
    finish_rollout,
    reset_rollout_stats,
)


def _tiny_history_policy(seed: int = 0) -> TichuPolicyHistory:
    torch.manual_seed(seed)
    encoder = TransformerHistoryEncoder(
        d_model=8,
        nhead=2,
        n_layers=1,
        ffn_dim=16,
    )
    encoder.use_checkpointing = False
    policy = TichuPolicyHistory(
        encoder,
        hidden_dim=16,
        n_layers=1,
        embed_dim=8,
    )
    return policy


def _reference_legal_pad(
    legal_flat: np.ndarray,
    legal_lens: np.ndarray,
    b_idx: np.ndarray,
) -> np.ndarray:
    offsets = _legal_offsets(legal_lens)
    max_n = max(int(legal_lens[b_idx].max()), 1)
    out = np.zeros((len(b_idx), max_n, legal_flat.shape[1]), dtype=np.float32)
    for row, step_idx in enumerate(b_idx):
        n = int(legal_lens[step_idx])
        off = int(offsets[step_idx])
        out[row, :n] = legal_flat[off : off + n]
    return out


def _synthetic_history_rollout(policy: TichuPolicyHistory) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1234)
    n = 9
    legal_lens = np.array([1, 3, 2, 6, 4, 5, 2, 7, 3], dtype=np.int32)
    history_lens = np.array([0, 5, 1, 8, 3, 7, 2, 6, 4], dtype=np.int32)
    max_h = int(history_lens.max())

    obs = rng.normal(0.0, 0.2, size=(n, STATIC_OBS_SIZE)).astype(np.float32)
    pass_source_ids = rng.integers(0, 57, size=(n, 3), dtype=np.int64)
    pass_target_ids = rng.integers(0, 57, size=(n, 3), dtype=np.int64)
    legal_flat = rng.integers(
        0,
        2,
        size=(int(legal_lens.sum()), ACTION_DIM),
        dtype=np.uint8,
    ).astype(np.float32)
    history = np.zeros((n, max_h, PLAY_TOKEN_SIZE), dtype=np.float32)
    for i, h_len in enumerate(history_lens):
        if h_len:
            history[i, :h_len, 4:] = rng.normal(
                0.0,
                0.3,
                size=(int(h_len), PLAY_TOKEN_SIZE - 4),
            )
            relative_players = rng.integers(0, 4, size=int(h_len))
            history[i, np.arange(int(h_len)), relative_players] = 1.0

    action_idxs = np.array(
        [rng.integers(0, int(n_legal)) for n_legal in legal_lens],
        dtype=np.int64,
    )

    legal_pad = _reference_legal_pad(legal_flat, legal_lens, np.arange(n))
    policy.eval()
    with torch.no_grad():
        log_probs, values, _ = policy.evaluate(
            torch.as_tensor(obs),
            torch.as_tensor(legal_pad),
            torch.as_tensor(legal_lens),
            torch.as_tensor(action_idxs),
            torch.as_tensor(history),
            torch.as_tensor(history_lens),
            torch.as_tensor(pass_source_ids),
            torch.as_tensor(pass_target_ids),
        )

    values_np = values.cpu().numpy().astype(np.float32)
    returns = (values_np + rng.normal(0.0, 0.5, size=n)).astype(np.float32)
    return {
        "obs": obs,
        "pass_source_ids": pass_source_ids,
        "pass_target_ids": pass_target_ids,
        "legal_flat": legal_flat,
        "legal_lens": legal_lens,
        "action_idxs": action_idxs,
        "log_probs": log_probs.cpu().numpy().astype(np.float32),
        "values": values_np,
        "returns": returns,
        "advantages": returns - values_np,
        "history": history.astype(np.float16),
        "history_lens": history_lens,
    }


def _dense_to_shared_rollout(
    rollout: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Represent each synthetic transition as a one-prefix game log."""
    shared = {key: value.copy() for key, value in rollout.items()}
    history = shared.pop("history")
    history_lens = shared["history_lens"]
    offsets = np.concatenate(
        [
            np.zeros(1, dtype=np.int32),
            np.cumsum(history_lens, dtype=np.int32),
        ]
    )
    token_rows = [
        history[row, :hist_len, 4:]
        for row, hist_len in enumerate(history_lens)
        if hist_len
    ]
    player_rows = [
        history[row, :hist_len, :4].argmax(axis=1)
        for row, hist_len in enumerate(history_lens)
        if hist_len
    ]
    shared.update(
        {
            "history_tokens": (
                np.concatenate(token_rows).astype(np.float16)
                if token_rows
                else np.zeros((0, PLAY_TOKEN_SIZE - 4), dtype=np.float16)
            ),
            "history_players": (
                np.concatenate(player_rows).astype(np.int8)
                if player_rows
                else np.zeros(0, dtype=np.int8)
            ),
            "history_game_offsets": offsets,
            "history_game_ids": np.arange(len(history_lens), dtype=np.int32),
            "history_observers": np.zeros(len(history_lens), dtype=np.int8),
        }
    )
    return shared


def _ppo_cfg(*, use_ppo_bucketing: bool, use_lazy_gpu_loading: bool = False) -> TrainConfig:
    cfg = TrainConfig()
    cfg.lr = 1e-4
    cfg.batch_size = 3
    cfg.n_epochs = 1
    cfg.entropy_coef = 0.01
    cfg.value_coef = 0.5
    cfg.max_grad_norm = 999.0
    cfg.use_ppo_bucketing = use_ppo_bucketing
    cfg.use_lazy_gpu_loading = use_lazy_gpu_loading
    cfg.use_amp = False
    return cfg


def _updated_policy_state(
    state_dict: dict[str, torch.Tensor],
    rollout: dict[str, np.ndarray],
    cfg: TrainConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    policy = _tiny_history_policy(seed=999)
    policy.load_state_dict(copy.deepcopy(state_dict))
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    np.random.seed(2024)
    stats = ppo_update_history(policy, optimizer, rollout, cfg, torch.device("cpu"))
    return copy.deepcopy(policy.state_dict()), stats


def test_ppo_legal_padding_eager_and_lazy_match_reference():
    legal_lens = np.array([1, 4, 2, 5], dtype=np.int32)
    legal_flat = np.arange(1, int(legal_lens.sum()) * ACTION_DIM + 1).reshape(
        int(legal_lens.sum()),
        ACTION_DIM,
    ).astype(np.float32)
    b_idx = np.array([3, 0, 2], dtype=np.int64)
    expected = _reference_legal_pad(legal_flat, legal_lens, b_idx)

    device = torch.device("cpu")
    offsets = _legal_offsets(legal_lens)
    eager = _pad_legal_eager(
        torch.as_tensor(legal_flat),
        torch.as_tensor(offsets, dtype=torch.int64),
        torch.as_tensor(b_idx, dtype=torch.int64),
        torch.as_tensor(legal_lens[b_idx], dtype=torch.int32),
    )
    lazy = _pad_legal_lazy(legal_flat, offsets, b_idx, legal_lens, device)

    np.testing.assert_array_equal(eager.numpy(), expected)
    np.testing.assert_array_equal(lazy.numpy(), expected)
    assert np.count_nonzero(eager.numpy()[1, 1:]) == 0


def test_history_buffer_rebuild_matches_encode_history_snapshots():
    gs = GameState(track_history=True)
    gs.reset(seed=17)

    steps = []
    expected_snapshots: list[tuple[np.ndarray, int]] = []
    for _ in range(24):
        while gs.dragon_give_pending is not None and not gs.round_over:
            options = gs.get_dragon_give_options(gs.dragon_give_pending)
            gs.give_dragon_trick(gs.dragon_give_pending, options[0])
        if gs.round_over:
            break

        player = gs.current_player
        hist, hist_len = encode_history(gs, player)
        expected_snapshots.append((hist[:hist_len].copy(), hist_len))
        steps.append(
            (
                encode_static(gs, player),
                encode_pass_source_ids(gs, player),
                encode_pass_target_ids(gs, player),
                np.zeros((1, ACTION_DIM), dtype=np.float32),
                0,
                0.0,
                0.0,
                player,
                hist_len,
            )
        )

        action = gs.get_legal_actions(player)[0]
        assert gs.play(player, action)
        if MAHJONG_INDEX in action[0] and not gs.round_over and gs.mahjong_wish is None:
            gs.set_mahjong_wish(None)

    hist_lens = np.array([s[8] for s in steps], dtype=np.int32)
    rebuilt = _rebuild_history_buffer([gs], [steps], hist_lens).astype(np.float32)
    shared = _compile_shared_history([gs], [steps])
    shared["history_lens"] = hist_lens
    rebuilt_shared = _rebuild_shared_history_numpy(
        shared,
        np.arange(len(steps)),
    )

    assert rebuilt.shape == (len(steps), max(int(hist_lens.max()), 1), PLAY_TOKEN_SIZE)
    np.testing.assert_array_equal(rebuilt_shared, rebuilt)
    for row, (expected, hist_len) in enumerate(expected_snapshots):
        np.testing.assert_allclose(rebuilt[row, :hist_len], expected, atol=1e-3)
        if hist_len < rebuilt.shape[1]:
            assert np.count_nonzero(rebuilt[row, hist_len:]) == 0


def test_history_rollout_invariants_with_and_without_forward_bucketing():
    for use_bucketing in (False, True):
        torch.manual_seed(42)
        np.random.seed(42)
        policy = _tiny_history_policy(seed=42)
        rollout, _stats = _collect_batched(
            policy,
            n_games=1,
            device=torch.device("cpu"),
            use_rollout_bucketing=use_bucketing,
        )

        n_steps = rollout["obs"].shape[0]
        assert n_steps > 0
        assert rollout["legal_flat"].shape[0] == int(rollout["legal_lens"].sum())
        assert rollout["legal_flat"].shape[1] == ACTION_DIM
        assert "history" not in rollout
        assert rollout["history_tokens"].shape[1] == PLAY_TOKEN_SIZE - 4
        assert rollout["history_game_ids"].shape == (n_steps,)
        assert rollout["history_observers"].shape == (n_steps,)
        segment_lens = np.diff(rollout["history_game_offsets"])
        assert np.all(
            rollout["history_lens"]
            <= segment_lens[rollout["history_game_ids"]]
        )
        rebuilt = _rebuild_shared_history_numpy(
            rollout,
            np.arange(n_steps),
        )
        assert rebuilt.shape[0] == n_steps
        for row, hist_len in enumerate(rollout["history_lens"]):
            if hist_len:
                np.testing.assert_array_equal(
                    rebuilt[row, :hist_len, :4].sum(axis=1),
                    np.ones(hist_len),
                )
            assert np.count_nonzero(rebuilt[row, hist_len:]) == 0
        assert np.all(rollout["action_idxs"] >= 0)
        assert np.all(rollout["action_idxs"] < rollout["legal_lens"])
        np.testing.assert_allclose(
            rollout["advantages"],
            rollout["returns"] - rollout["values"],
            atol=1e-6,
        )
        for key in ("log_probs", "values", "returns", "advantages"):
            assert np.isfinite(rollout[key]).all(), key


def test_ppo_history_bucketing_does_not_change_update():
    base_policy = _tiny_history_policy(seed=5)
    rollout = _synthetic_history_rollout(base_policy)
    base_state = copy.deepcopy(base_policy.state_dict())

    unbucketed, stats_unbucketed = _updated_policy_state(
        base_state,
        rollout,
        _ppo_cfg(use_ppo_bucketing=False),
    )
    bucketed, stats_bucketed = _updated_policy_state(
        base_state,
        rollout,
        _ppo_cfg(use_ppo_bucketing=True),
    )

    for key in base_state:
        torch.testing.assert_close(bucketed[key], unbucketed[key], atol=2e-5, rtol=2e-5)
    for key in stats_unbucketed:
        assert np.isfinite(stats_bucketed[key])


def test_ppo_history_lazy_and_eager_loading_match():
    base_policy = _tiny_history_policy(seed=6)
    rollout = _dense_to_shared_rollout(_synthetic_history_rollout(base_policy))
    base_state = copy.deepcopy(base_policy.state_dict())

    eager, _ = _updated_policy_state(
        base_state,
        rollout,
        _ppo_cfg(use_ppo_bucketing=True, use_lazy_gpu_loading=False),
    )
    lazy, _ = _updated_policy_state(
        base_state,
        rollout,
        _ppo_cfg(use_ppo_bucketing=True, use_lazy_gpu_loading=True),
    )

    for key in base_state:
        torch.testing.assert_close(lazy[key], eager[key], atol=2e-5, rtol=2e-5)


def test_ppo_shared_history_matches_legacy_dense_update():
    base_policy = _tiny_history_policy(seed=7)
    dense_rollout = _synthetic_history_rollout(base_policy)
    shared_rollout = _dense_to_shared_rollout(dense_rollout)
    base_state = copy.deepcopy(base_policy.state_dict())
    cfg = _ppo_cfg(use_ppo_bucketing=True)

    dense, dense_stats = _updated_policy_state(base_state, dense_rollout, cfg)
    shared, shared_stats = _updated_policy_state(base_state, shared_rollout, cfg)

    for key in base_state:
        torch.testing.assert_close(shared[key], dense[key], atol=2e-5, rtol=2e-5)
    for key in dense_stats:
        if key == "ppo_seconds":
            continue
        assert abs(shared_stats[key] - dense_stats[key]) < 2e-5


def test_optimizer_batch_size_controls_steps_independently_of_microbatch():
    base_policy = _tiny_history_policy(seed=8)
    rollout = _synthetic_history_rollout(base_policy)
    policy = _tiny_history_policy(seed=999)
    policy.load_state_dict(copy.deepcopy(base_policy.state_dict()))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    cfg = _ppo_cfg(use_ppo_bucketing=True)
    cfg.n_epochs = 2
    cfg.batch_size = 3
    cfg.optimizer_batch_size = 4

    np.random.seed(2024)
    stats = ppo_update_history(
        policy,
        optimizer,
        rollout,
        cfg,
        torch.device("cpu"),
    )

    # ceil(9 transitions / 4 per optimizer batch) * 2 epochs.
    assert stats["optimizer_steps"] == 6
    assert stats["transitions_per_optimizer_step"] == 3
    trunk_parameter = next(policy.trunk.parameters())
    assert int(optimizer.state[trunk_parameter]["step"]) == 6


def test_full_rollout_optimizer_batch_retains_one_step_per_epoch():
    base_policy = _tiny_history_policy(seed=9)
    rollout = _synthetic_history_rollout(base_policy)
    policy = _tiny_history_policy(seed=999)
    policy.load_state_dict(copy.deepcopy(base_policy.state_dict()))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    cfg = _ppo_cfg(use_ppo_bucketing=True)
    cfg.n_epochs = 2
    cfg.batch_size = 3
    cfg.optimizer_batch_size = None

    np.random.seed(2024)
    stats = ppo_update_history(
        policy,
        optimizer,
        rollout,
        cfg,
        torch.device("cpu"),
    )

    assert stats["optimizer_steps"] == 2
    assert stats["transitions_per_optimizer_step"] == 9
    trunk_parameter = next(policy.trunk.parameters())
    assert int(optimizer.state[trunk_parameter]["step"]) == 2


def test_paired_memory_batch_analysis_uses_per_rollout_medians():
    measurements = []
    for rollout_index, winner_times, other_times in (
        (0, (2.0, 2.2, 2.1), (3.0, 3.2, 3.1)),
        (1, (4.0, 4.2, 4.1), (5.0, 5.2, 5.1)),
    ):
        for batch_size, times in ((2048, winner_times), (1024, other_times)):
            for repeat, seconds in enumerate(times):
                measurements.append(
                    {
                        "rollout_index": rollout_index,
                        "batch_size": batch_size,
                        "status": "ok",
                        "repeat": repeat,
                        "ppo_seconds": seconds,
                    }
                )

    differences = _paired_log_time_differences(
        measurements,
        winner=2048,
        other=1024,
        n_rollouts=2,
        repeats=3,
    )

    np.testing.assert_allclose(
        differences,
        np.log(np.array([3.1 / 2.1, 5.1 / 4.1])),
    )


def test_sign_flip_and_holm_detect_clear_paired_speedup():
    differences = np.full(24, np.log(1.10), dtype=np.float64)

    pvalue = _one_sided_sign_flip_pvalue(
        differences,
        samples=20_000,
        seed=123,
    )

    assert pvalue < 0.001
    np.testing.assert_allclose(
        _holm_adjust([0.01, 0.03, 0.04]),
        [0.03, 0.06, 0.06],
    )


def test_optimizer_runtime_analysis_pairs_rollouts_and_reports_steps():
    measurements = []
    for rollout_index, transitions in ((0, 100), (1, 120)):
        for optimizer_batch, times, steps in (
            (2048, (2.0, 2.2), 2),
            (0, (1.8, 2.0), 1),
        ):
            for repeat, seconds in enumerate(times):
                measurements.append(
                    {
                        "rollout_index": rollout_index,
                        "optimizer_batch_size": optimizer_batch,
                        "status": "ok",
                        "repeat": repeat,
                        "ppo_seconds": seconds,
                        "transitions": transitions,
                        "optimizer_steps": steps,
                        "peak_allocated_mb": 100.0,
                        "peak_reserved_mb": 120.0,
                    }
                )

    ratios = _paired_optimizer_log_time_ratios(
        measurements,
        candidate=2048,
        reference=0,
        n_rollouts=2,
        repeats=2,
    )
    np.testing.assert_allclose(ratios, np.full(2, np.log(2.1 / 1.9)))

    summary = _summarize_optimizer_runtime(
        measurements,
        candidates=[2048, 0],
        n_rollouts=2,
        repeats=2,
    )
    assert summary[0]["optimizer_batch_label"] == "2048"
    assert summary[0]["median_optimizer_steps"] == 2
    assert summary[1]["optimizer_batch_label"] == "full_rollout"
    assert summary[1]["median_optimizer_steps"] == 1


class _ReadyResult:
    def __init__(self, payload):
        self.payload = payload

    def ready(self) -> bool:
        return True

    def get(self):
        return self.payload


def _fake_rollout(n_steps: int, hist_len: int, offset: float) -> dict[str, np.ndarray]:
    legal_lens = np.arange(1, n_steps + 1, dtype=np.int32)
    legal_flat = np.full((int(legal_lens.sum()), ACTION_DIM), offset, dtype=np.float32)
    return {
        "obs": np.full((n_steps, STATIC_OBS_SIZE), offset, dtype=np.float32),
        "pass_source_ids": np.zeros((n_steps, 3), dtype=np.int64),
        "pass_target_ids": np.zeros((n_steps, 3), dtype=np.int64),
        "legal_flat": legal_flat,
        "legal_lens": legal_lens,
        "action_idxs": np.zeros(n_steps, dtype=np.int64),
        "log_probs": np.zeros(n_steps, dtype=np.float32),
        "values": np.zeros(n_steps, dtype=np.float32),
        "returns": np.ones(n_steps, dtype=np.float32) * offset,
        "advantages": np.ones(n_steps, dtype=np.float32) * offset,
        "history": np.full(
            (n_steps, hist_len, PLAY_TOKEN_SIZE),
            offset,
            dtype=np.float16,
        ),
        "history_lens": np.full(n_steps, hist_len, dtype=np.int32),
    }


def test_finish_rollout_merges_variable_history_and_skips_empty_buffers():
    reset_rollout_stats()
    first = _fake_rollout(n_steps=2, hist_len=3, offset=1.0)
    second = _fake_rollout(n_steps=1, hist_len=5, offset=2.0)
    stats = {"total_steps": 1}
    result = _ReadyResult([(first, stats), ({}, stats), (second, stats)])

    merged = finish_rollout(result, torch.device("cpu"))

    assert merged["history"].shape == (3, 5, PLAY_TOKEN_SIZE)
    np.testing.assert_array_equal(merged["history"][:2, :3], first["history"])
    assert np.count_nonzero(merged["history"][:2, 3:]) == 0
    np.testing.assert_array_equal(merged["history"][2:, :5], second["history"])
    np.testing.assert_array_equal(
        merged["legal_lens"],
        np.concatenate([first["legal_lens"], second["legal_lens"]]),
    )
    assert merged["legal_flat"].shape[0] == int(merged["legal_lens"].sum())


def test_finish_rollout_merges_shared_history_references():
    reset_rollout_stats()
    first = _dense_to_shared_rollout(
        _fake_rollout(n_steps=2, hist_len=3, offset=1.0)
    )
    second = _dense_to_shared_rollout(
        _fake_rollout(n_steps=1, hist_len=5, offset=2.0)
    )
    result = _ReadyResult(
        [
            (first, {"total_steps": 2}),
            (second, {"total_steps": 1}),
        ]
    )

    merged = finish_rollout(result, torch.device("cpu"))
    rebuilt = _rebuild_shared_history_numpy(merged, np.arange(3))

    assert merged["history_game_offsets"].tolist() == [0, 3, 6, 11]
    assert merged["history_game_ids"].tolist() == [0, 1, 2]
    np.testing.assert_array_equal(
        rebuilt[:2, :3, 4:],
        first["history_tokens"]
        .reshape(2, 3, PLAY_TOKEN_SIZE - 4)
        .astype(np.float32),
    )
    np.testing.assert_array_equal(
        rebuilt[2, :5, 4:],
        second["history_tokens"].astype(np.float32),
    )
