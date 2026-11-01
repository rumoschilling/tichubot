"""Rollout scheduling regressions for the shared play-training loop."""

from __future__ import annotations

import numpy as np
import torch

from training.config import TrainConfig
from training.play_training import (
    PlayTrainingBundle,
    PlayTrainingHooks,
    run_play_training,
)


def _bundle() -> PlayTrainingBundle:
    policy = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    gt_policy = torch.nn.Linear(1, 1, bias=False)
    gt_optimizer = torch.optim.SGD(gt_policy.parameters(), lr=0.1)
    with torch.no_grad():
        policy.weight.zero_()
    return PlayTrainingBundle(
        policy=policy,
        optimizer=optimizer,
        gt_policy=gt_policy,
        gt_optimizer=gt_optimizer,
    )


def _cfg(tmp_path, *, synchronous: bool) -> TrainConfig:
    cfg = TrainConfig()
    cfg.total_games = 300
    cfg.games_per_update = 100
    cfg.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.log_file = str(tmp_path / "train.csv")
    cfg.checkpoint_every = 1_000_000
    cfg.log_every = 1_000_000
    cfg.pass_every = 100
    cfg.train_gt = True
    cfg.gt_probe_every = 200
    cfg.gt_probe_games = 5
    cfg.gt_probe_chunk = 5
    cfg.synchronous = synchronous
    return cfg


def _run_schedule(monkeypatch, tmp_path, *, synchronous: bool):
    bundle = _bundle()
    events: list[tuple] = []
    next_job = 0

    def start_rollout(blocking):
        nonlocal next_job
        job = next_job
        next_job += 1
        events.append(
            ("start", job, blocking, float(bundle.policy.weight.item()))
        )
        return job

    def finish_rollout(job):
        events.append(("finish", job))
        return {
            "obs": np.zeros((2, 1), dtype=np.float32),
            "returns": np.zeros(2, dtype=np.float32),
        }

    def ppo_update(_rollout):
        with torch.no_grad():
            bundle.policy.weight.add_(1.0)
        events.append(("ppo", float(bundle.policy.weight.item())))
        return {}

    def fake_probe(*_args, **_kwargs):
        with torch.no_grad():
            bundle.policy.weight.add_(10.0)
        events.append(("probe", float(bundle.policy.weight.item())))

    def fake_pass_step(*_args, **_kwargs):
        events.append(("pass",))

    monkeypatch.setattr(
        "training.play_training._run_gt_probe_block",
        fake_probe,
    )
    monkeypatch.setattr(
        "training.play_training._train_passing_step",
        fake_pass_step,
    )
    hooks = PlayTrainingHooks(
        start_rollout=start_rollout,
        finish_rollout=finish_rollout,
        ppo_update=ppo_update,
        save_checkpoint=lambda *_args: None,
        metric_fields=(),
        supports_async=True,
        collect_gt_probe=lambda _n: {},
    )
    run_play_training(
        _cfg(tmp_path, synchronous=synchronous),
        bundle,
        hooks,
        torch.device("cpu"),
    )
    return events


def test_synchronous_rollouts_start_after_each_policy_update(monkeypatch, tmp_path):
    events = _run_schedule(
        monkeypatch,
        tmp_path,
        synchronous=True,
    )
    starts = [event for event in events if event[0] == "start"]
    second_start_idx = next(
        i for i, event in enumerate(events) if event[:2] == ("start", 1)
    )

    assert [event[2] for event in starts] == [True, True, True]
    assert [event[3] for event in starts] == [0.0, 1.0, 12.0]
    assert events[second_start_idx - 1] == ("pass",)


def test_async_rollout_is_not_prefetched_across_probe_block(monkeypatch, tmp_path):
    events = _run_schedule(
        monkeypatch,
        tmp_path,
        synchronous=False,
    )
    probe_idx = next(i for i, event in enumerate(events) if event[0] == "probe")
    third_start_idx = next(
        i
        for i, event in enumerate(events)
        if event[:2] == ("start", 2)
    )

    assert third_start_idx == probe_idx + 1
    assert events[third_start_idx][2:] == (False, 12.0)
