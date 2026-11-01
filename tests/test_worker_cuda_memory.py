from __future__ import annotations

import numpy as np
import pytest
import torch

import training.rollout_history as rollout_history


_MIB = 2**20
_GIB = 2**30


def _patch_cuda_counters(
    monkeypatch,
    *,
    allocated: list[int],
    reserved: list[int],
    free: list[int],
    total: int = 16 * _GIB,
) -> list[bool]:
    empty_cache_calls: list[bool] = []
    allocated_values = iter(allocated)
    reserved_values = iter(reserved)
    free_values = iter(free)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda _device: next(allocated_values),
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda _device: next(reserved_values),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (next(free_values), total),
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append(True),
    )
    return empty_cache_calls


def test_worker_cuda_check_keeps_cache_on_healthy_fast_path(monkeypatch):
    monkeypatch.setattr(
        rollout_history,
        "_WORKER_CUDA_ALLOCATED_FLOORS",
        {},
    )
    calls = _patch_cuda_counters(
        monkeypatch,
        allocated=[100 * _MIB],
        reserved=[900 * _MIB, 900 * _MIB],
        free=[6 * _GIB, 6 * _GIB],
    )

    stats = rollout_history._check_worker_cuda_memory(torch.device("cuda"))

    assert calls == []
    assert stats["worker_cuda_check_count"] == 1
    assert stats["worker_cuda_cleanup_count"] == 0
    assert stats["worker_cuda_cache_released_mb"] == 0
    assert stats["worker_cuda_leak_suspected_count"] == 0


def test_worker_cuda_check_releases_only_unused_cache_under_pressure(monkeypatch):
    monkeypatch.setattr(
        rollout_history,
        "_WORKER_CUDA_ALLOCATED_FLOORS",
        {},
    )
    calls = _patch_cuda_counters(
        monkeypatch,
        allocated=[100 * _MIB],
        reserved=[900 * _MIB, 200 * _MIB],
        free=[2 * _GIB, 2700 * _MIB],
    )

    stats = rollout_history._check_worker_cuda_memory(torch.device("cuda"))

    assert calls == [True]
    assert stats["worker_cuda_cleanup_count"] == 1
    assert stats["worker_cuda_allocated_mb"] == 100
    assert stats["worker_cuda_reserved_mb"] == 200
    assert stats["worker_cuda_cache_released_mb"] == 700


def test_worker_cuda_check_flags_live_allocation_growth(monkeypatch):
    monkeypatch.setattr(
        rollout_history,
        "_WORKER_CUDA_ALLOCATED_FLOORS",
        {},
    )
    calls = _patch_cuda_counters(
        monkeypatch,
        allocated=[100 * _MIB, 200 * _MIB],
        reserved=[300 * _MIB, 300 * _MIB, 400 * _MIB, 400 * _MIB],
        free=[6 * _GIB, 6 * _GIB, 6 * _GIB, 6 * _GIB],
    )

    first = rollout_history._check_worker_cuda_memory(torch.device("cuda"))
    second = rollout_history._check_worker_cuda_memory(torch.device("cuda"))

    assert calls == []
    assert first["worker_cuda_leak_suspected_count"] == 0
    assert second["worker_cuda_leak_suspected_count"] == 1


class _CachedPolicy:
    def to(self, _device):
        return self

    def load_state_dict(self, _state_dict):
        return None

    def eval(self):
        return self


@pytest.mark.parametrize("gt_probe", [False, True])
def test_worker_checks_cuda_after_normal_and_gt_rollouts(monkeypatch, gt_probe):
    # The worker adds the history-norm flags read from the (empty) weights.
    policy_key = ("policy", (("final_norm", False),), (("history_pool_norm", False),))
    monkeypatch.setattr(
        rollout_history,
        "_WORKER_MODELS",
        {policy_key: _CachedPolicy()},
    )
    observed_probe_modes: list[bool] = []
    checks: list[bool] = []

    def fake_collect(*_args, **kwargs):
        observed_probe_modes.append(bool(kwargs["gt_probe"]))
        return {"marker": np.array([1])}, {}

    def fake_check(_device):
        checks.append(True)
        return {"worker_cuda_check_count": 1}

    monkeypatch.setattr(rollout_history, "_collect_batched", fake_collect)
    monkeypatch.setattr(
        rollout_history,
        "_check_worker_cuda_memory",
        fake_check,
    )

    _buffer, stats = rollout_history._worker_collect(
        (
            {},
            None,
            None,
            5,
            "cpu",
            1,
            {},
            {},
            True,
            True,
            gt_probe,
            False,
            123,
        )
    )

    assert observed_probe_modes == [gt_probe]
    assert checks == [True]
    assert stats["worker_cuda_check_count"] == 1
