"""Controlled history-policy memory- and optimizer-batch experiments.

Each runtime benchmark replays real frozen rollouts through fresh checkpoint
clones. No update survives a timing measurement. The memory benchmark holds
the effective optimizer batch at the full rollout; the optimizer benchmark
holds the validated forward/backward memory microbatch fixed.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from training.config import TrainConfig
from training.model_gt import TichuGrandTichuPolicy
from training.model_passing import TichuPassingPolicy
from training.ppo_history import ppo_update_history
from training.rollout_history import collect_rollouts_history
from training.train_history import (
    _build_policy,
    _encoder_cfg,
    _make_optimizer,
    _model_cfg,
    _prepare_policy_for_resume,
    load_checkpoint,
)


def _config_from_checkpoint(path: Path, device: torch.device) -> TrainConfig:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("config")
    cfg = TrainConfig()
    if saved is not None:
        for field in fields(TrainConfig):
            if isinstance(saved, dict):
                if field.name in saved:
                    setattr(cfg, field.name, copy.deepcopy(saved[field.name]))
            elif hasattr(saved, field.name):
                setattr(cfg, field.name, copy.deepcopy(getattr(saved, field.name)))
    cfg.device = str(device)
    cfg.use_amp = False
    cfg.synchronous = True
    cfg.n_workers = 1
    return cfg


def _load_training_bundle(
    checkpoint_path: Path,
    cfg: TrainConfig,
    device: torch.device,
):
    policy = _build_policy(cfg).to(device)
    _prepare_policy_for_resume(policy, str(checkpoint_path), device)
    optimizer = _make_optimizer(policy, cfg)
    pass_policy = TichuPassingPolicy(
        hidden_dim=cfg.pass_hidden_dim,
        n_layers=cfg.pass_n_layers,
    ).to(device)
    pass_optimizer = torch.optim.Adam(
        pass_policy.parameters(),
        lr=cfg.pass_lr,
    )
    gt_policy = TichuGrandTichuPolicy(
        hidden_dim=cfg.gt_hidden_dim,
        n_layers=cfg.gt_n_layers,
    ).to(device)
    gt_optimizer = torch.optim.Adam(gt_policy.parameters(), lr=cfg.gt_lr)
    load_checkpoint(
        policy,
        optimizer,
        str(checkpoint_path),
        device,
        pass_policy,
        pass_optimizer,
        gt_policy,
        gt_optimizer,
    )
    return policy, optimizer, pass_policy, gt_policy


def _run_memory_candidate(
    checkpoint_path: Path,
    base_cfg: TrainConfig,
    rollout: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = copy.deepcopy(base_cfg)
    cfg.batch_size = batch_size
    cfg.optimizer_batch_size = None
    cfg.n_epochs = 1
    policy = optimizer = pass_policy = gt_policy = None
    try:
        policy, optimizer, pass_policy, gt_policy = _load_training_bundle(
            checkpoint_path,
            cfg,
            device,
        )
        stats = ppo_update_history(policy, optimizer, rollout, cfg, device)
        return {
            "batch_size": batch_size,
            "status": "ok",
            "ppo_seconds": stats["ppo_seconds"],
            "transitions_per_second": len(rollout["obs"])
            / stats["ppo_seconds"],
            "peak_allocated_mb": stats["peak_allocated_mb"],
            "peak_reserved_mb": stats["peak_reserved_mb"],
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        message = str(exc)
        if "out of memory" not in message.lower():
            raise
        return {
            "batch_size": batch_size,
            "status": "oom",
            "error": message,
        }
    finally:
        del policy, optimizer, pass_policy, gt_policy
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _run_optimizer_candidate(
    checkpoint_path: Path,
    base_cfg: TrainConfig,
    rollout: dict[str, np.ndarray],
    optimizer_batch_size: int,
    memory_batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, float | int | str]:
    """Time one disposable PPO epoch for one effective Adam batch.

    ``optimizer_batch_size == 0`` is the explicit benchmark sentinel for the
    full-rollout legacy behavior. Model and optimizer state are freshly loaded
    for every call, so no learning persists between timing measurements.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = copy.deepcopy(base_cfg)
    cfg.batch_size = memory_batch_size
    cfg.optimizer_batch_size = (
        None if optimizer_batch_size == 0 else optimizer_batch_size
    )
    cfg.n_epochs = 1
    policy = optimizer = pass_policy = gt_policy = None
    try:
        policy, optimizer, pass_policy, gt_policy = _load_training_bundle(
            checkpoint_path,
            cfg,
            device,
        )
        stats = ppo_update_history(policy, optimizer, rollout, cfg, device)
        return {
            "optimizer_batch_size": optimizer_batch_size,
            "optimizer_batch_label": (
                "full_rollout" if optimizer_batch_size == 0
                else str(optimizer_batch_size)
            ),
            "status": "ok",
            "ppo_seconds": stats["ppo_seconds"],
            "transitions_per_second": (
                len(rollout["obs"]) / stats["ppo_seconds"]
            ),
            "optimizer_steps": int(stats["optimizer_steps"]),
            "transitions_per_optimizer_step": stats[
                "transitions_per_optimizer_step"
            ],
            "peak_allocated_mb": stats["peak_allocated_mb"],
            "peak_reserved_mb": stats["peak_reserved_mb"],
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        message = str(exc)
        if "out of memory" not in message.lower():
            raise
        return {
            "optimizer_batch_size": optimizer_batch_size,
            "optimizer_batch_label": (
                "full_rollout" if optimizer_batch_size == 0
                else str(optimizer_batch_size)
            ),
            "status": "oom",
            "error": message,
        }
    finally:
        del policy, optimizer, pass_policy, gt_policy
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _collect_benchmark_rollout(
    checkpoint_path: Path,
    base_cfg: TrainConfig,
    rollout_games: int,
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy, optimizer, pass_policy, gt_policy = _load_training_bundle(
        checkpoint_path,
        base_cfg,
        device,
    )
    try:
        return collect_rollouts_history(
            policy,
            rollout_games,
            device,
            n_workers=1,
            encoder_cfg=_encoder_cfg(base_cfg),
            model_cfg=_model_cfg(base_cfg),
            use_rollout_bucketing=base_cfg.use_rollout_bucketing,
            with_calling=True,
            gt_policy=gt_policy,
            pass_policy=pass_policy,
            block=True,
            disable_gt_explore=False,
            seed=seed,
        )
    finally:
        del policy, optimizer, pass_policy, gt_policy
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _measure_rollout_stage(
    *,
    stage: str,
    checkpoint_path: Path,
    base_cfg: TrainConfig,
    candidates: list[int],
    n_rollouts: int,
    rollout_games: int,
    repeats: int,
    device: torch.device,
    seed: int,
    warmup: bool,
    on_rollout_complete,
) -> list[dict]:
    measurements: list[dict] = []
    for rollout_index in range(n_rollouts):
        rollout_seed = seed + rollout_index
        rollout = _collect_benchmark_rollout(
            checkpoint_path,
            base_cfg,
            rollout_games,
            device,
            rollout_seed,
        )
        transitions = len(rollout["obs"])
        if warmup and rollout_index == 0:
            _run_memory_candidate(
                checkpoint_path,
                base_cfg,
                rollout,
                min(candidates),
                device,
                rollout_seed + 1_000_000,
            )

        for repeat in range(repeats):
            order_rng = np.random.default_rng(
                rollout_seed + 10_000 * (repeat + 1)
            )
            candidate_order = list(candidates)
            order_rng.shuffle(candidate_order)
            for order_index, batch_size in enumerate(candidate_order):
                row = _run_memory_candidate(
                    checkpoint_path,
                    base_cfg,
                    rollout,
                    batch_size,
                    device,
                    rollout_seed + 100_000 * (repeat + 1),
                )
                row.update(
                    {
                        "stage": stage,
                        "rollout_index": rollout_index,
                        "rollout_seed": rollout_seed,
                        "transitions": transitions,
                        "repeat": repeat,
                        "order_index": order_index,
                    }
                )
                measurements.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        del rollout
        if device.type == "cuda":
            torch.cuda.empty_cache()
        on_rollout_complete(measurements, rollout_index + 1)
    return measurements


def _summarize_stage(
    measurements: list[dict],
    candidates: list[int],
    n_rollouts: int,
    repeats: int,
) -> list[dict]:
    summary: list[dict] = []
    for batch_size in candidates:
        block_seconds: list[float] = []
        block_throughputs: list[float] = []
        candidate_rows = [
            row for row in measurements if row["batch_size"] == batch_size
        ]
        for rollout_index in range(n_rollouts):
            rows = [
                row
                for row in candidate_rows
                if row["rollout_index"] == rollout_index
                and row["status"] == "ok"
            ]
            if len(rows) != repeats:
                continue
            seconds = statistics.median(
                float(row["ppo_seconds"]) for row in rows
            )
            transitions = int(rows[0]["transitions"])
            block_seconds.append(seconds)
            block_throughputs.append(transitions / seconds)

        complete = len(block_seconds) == n_rollouts
        row: dict = {
            "batch_size": batch_size,
            "status": "ok" if complete else "oom_or_incomplete",
            "completed_rollouts": len(block_seconds),
            "required_rollouts": n_rollouts,
        }
        if complete:
            row.update(
                {
                    "geometric_mean_ppo_seconds": math.exp(
                        statistics.mean(math.log(value) for value in block_seconds)
                    ),
                    "geometric_mean_transitions_per_second": math.exp(
                        statistics.mean(
                            math.log(value) for value in block_throughputs
                        )
                    ),
                    "median_transitions_per_second": statistics.median(
                        block_throughputs
                    ),
                    "max_peak_allocated_mb": max(
                        float(measurement["peak_allocated_mb"])
                        for measurement in candidate_rows
                        if measurement["status"] == "ok"
                    ),
                    "max_peak_reserved_mb": max(
                        float(measurement["peak_reserved_mb"])
                        for measurement in candidate_rows
                        if measurement["status"] == "ok"
                    ),
                }
            )
        summary.append(row)
    return summary


def _paired_log_time_differences(
    measurements: list[dict],
    winner: int,
    other: int,
    n_rollouts: int,
    repeats: int,
) -> np.ndarray:
    differences = []
    for rollout_index in range(n_rollouts):
        medians = {}
        for batch_size in (winner, other):
            rows = [
                row
                for row in measurements
                if row["rollout_index"] == rollout_index
                and row["batch_size"] == batch_size
                and row["status"] == "ok"
            ]
            if len(rows) != repeats:
                break
            medians[batch_size] = statistics.median(
                float(row["ppo_seconds"]) for row in rows
            )
        if len(medians) == 2:
            # Positive means the independently selected pilot winner is faster.
            differences.append(math.log(medians[other] / medians[winner]))
    return np.asarray(differences, dtype=np.float64)


def _one_sided_sign_flip_pvalue(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float:
    """Randomization p-value for mean(differences) > 0 under symmetry."""
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    exceedances = 1
    completed = 0
    while completed < samples:
        block = min(10_000, samples - completed)
        signs = rng.integers(
            0,
            2,
            size=(block, len(differences)),
            dtype=np.int8,
        )
        signs = signs.astype(np.float64) * 2.0 - 1.0
        null_means = (signs * differences).mean(axis=1)
        exceedances += int(np.count_nonzero(null_means >= observed))
        completed += block
    return exceedances / (samples + 1)


def _bootstrap_geometric_speedup_ci(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    completed = 0
    while completed < samples:
        block = min(10_000, samples - completed)
        indices = rng.integers(
            0,
            len(differences),
            size=(block, len(differences)),
        )
        means[completed : completed + block] = differences[indices].mean(axis=1)
        completed += block
    lower, upper = np.percentile(means, [2.5, 97.5])
    return [math.exp(float(lower)), math.exp(float(upper))]


def _holm_adjust(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(pvalues)
    running_max = 0.0
    comparisons = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (comparisons - rank) * pvalues[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def _paired_optimizer_log_time_ratios(
    measurements: list[dict],
    candidate: int,
    reference: int,
    n_rollouts: int,
    repeats: int,
) -> np.ndarray:
    """Return paired log(candidate time / reference time) per rollout."""
    differences = []
    for rollout_index in range(n_rollouts):
        medians = {}
        for optimizer_batch_size in (candidate, reference):
            rows = [
                row
                for row in measurements
                if row["rollout_index"] == rollout_index
                and row["optimizer_batch_size"] == optimizer_batch_size
                and row["status"] == "ok"
            ]
            if len(rows) != repeats:
                break
            medians[optimizer_batch_size] = statistics.median(
                float(row["ppo_seconds"]) for row in rows
            )
        if len(medians) == 2:
            differences.append(
                math.log(medians[candidate] / medians[reference])
            )
    return np.asarray(differences, dtype=np.float64)


def _summarize_optimizer_runtime(
    measurements: list[dict],
    candidates: list[int],
    n_rollouts: int,
    repeats: int,
) -> list[dict]:
    summary = []
    for optimizer_batch_size in candidates:
        block_seconds = []
        block_throughputs = []
        block_steps = []
        candidate_rows = [
            row
            for row in measurements
            if row["optimizer_batch_size"] == optimizer_batch_size
        ]
        for rollout_index in range(n_rollouts):
            rows = [
                row
                for row in candidate_rows
                if row["rollout_index"] == rollout_index
                and row["status"] == "ok"
            ]
            if len(rows) != repeats:
                continue
            seconds = statistics.median(
                float(row["ppo_seconds"]) for row in rows
            )
            transitions = int(rows[0]["transitions"])
            block_seconds.append(seconds)
            block_throughputs.append(transitions / seconds)
            block_steps.append(int(rows[0]["optimizer_steps"]))

        complete = len(block_seconds) == n_rollouts
        row: dict = {
            "optimizer_batch_size": optimizer_batch_size,
            "optimizer_batch_label": (
                "full_rollout" if optimizer_batch_size == 0
                else str(optimizer_batch_size)
            ),
            "status": "ok" if complete else "oom_or_incomplete",
            "completed_rollouts": len(block_seconds),
            "required_rollouts": n_rollouts,
        }
        if complete:
            row.update(
                {
                    "geometric_mean_ppo_seconds": math.exp(
                        statistics.mean(math.log(value) for value in block_seconds)
                    ),
                    "geometric_mean_transitions_per_second": math.exp(
                        statistics.mean(
                            math.log(value) for value in block_throughputs
                        )
                    ),
                    "median_optimizer_steps": statistics.median(block_steps),
                    "optimizer_steps_range": [min(block_steps), max(block_steps)],
                    "max_peak_allocated_mb": max(
                        float(measurement["peak_allocated_mb"])
                        for measurement in candidate_rows
                        if measurement["status"] == "ok"
                    ),
                    "max_peak_reserved_mb": max(
                        float(measurement["peak_reserved_mb"])
                        for measurement in candidate_rows
                        if measurement["status"] == "ok"
                    ),
                }
            )
        summary.append(row)
    return summary


def benchmark_optimizer_runtime_statistical(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base_cfg = _config_from_checkpoint(checkpoint_path, device)
    candidates = list(args.optimizer_batch_sizes)

    result: dict = {
        "status": "running",
        "design": {
            "analysis": "paired independent-rollout Adam batch runtime curve",
            "rollout_games": args.rollout_games,
            "rollouts": args.rollouts,
            "timing_repeats_per_rollout": args.repeats,
            "memory_batch_size": args.memory_batch_size,
            "optimizer_batch_sizes": candidates,
            "optimizer_batch_zero_means": "full_rollout",
            "runtime_margin_fraction": args.runtime_margin,
            "runtime_margin_interpretation": (
                "smallest finite batch whose paired bootstrap 95% upper "
                "runtime-ratio bound is within this margin of full rollout"
            ),
            "alpha": args.alpha,
            "one_sided_sign_flip_samples": args.permutation_samples,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "ppo_epochs_per_measurement": 1,
            "persistent_training": False,
        },
        "checkpoint": str(checkpoint_path),
        "measurements": [],
    }
    _atomic_write_json(output_path, result)

    measurements: list[dict] = []
    for rollout_index in range(args.rollouts):
        rollout_seed = args.seed + rollout_index
        rollout = _collect_benchmark_rollout(
            checkpoint_path,
            base_cfg,
            args.rollout_games,
            device,
            rollout_seed,
        )
        transitions = len(rollout["obs"])
        if rollout_index == 0:
            _run_optimizer_candidate(
                checkpoint_path,
                base_cfg,
                rollout,
                0,
                args.memory_batch_size,
                device,
                rollout_seed + 1_000_000,
            )

        for repeat in range(args.repeats):
            order_rng = np.random.default_rng(
                rollout_seed + 10_000 * (repeat + 1)
            )
            candidate_order = list(candidates)
            order_rng.shuffle(candidate_order)
            for order_index, optimizer_batch_size in enumerate(candidate_order):
                row = _run_optimizer_candidate(
                    checkpoint_path,
                    base_cfg,
                    rollout,
                    optimizer_batch_size,
                    args.memory_batch_size,
                    device,
                    rollout_seed + 100_000 * (repeat + 1),
                )
                row.update(
                    {
                        "rollout_index": rollout_index,
                        "rollout_seed": rollout_seed,
                        "transitions": transitions,
                        "repeat": repeat,
                        "order_index": order_index,
                    }
                )
                measurements.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        del rollout
        if device.type == "cuda":
            torch.cuda.empty_cache()
        result["measurements"] = measurements
        result["rollouts_completed"] = rollout_index + 1
        _atomic_write_json(output_path, result)

    summary = _summarize_optimizer_runtime(
        measurements,
        candidates,
        args.rollouts,
        args.repeats,
    )
    full_row = next(
        row
        for row in summary
        if row["optimizer_batch_size"] == 0 and row["status"] == "ok"
    )

    comparisons = []
    finite_candidates = sorted(candidate for candidate in candidates if candidate)
    for candidate in finite_candidates:
        differences = _paired_optimizer_log_time_ratios(
            measurements,
            candidate,
            0,
            args.rollouts,
            args.repeats,
        )
        if len(differences) != args.rollouts:
            comparisons.append(
                {
                    "optimizer_batch_size": candidate,
                    "status": "oom_or_incomplete",
                    "paired_rollouts": len(differences),
                }
            )
            continue
        comparison_seed = args.seed + 2_000_000 + candidate
        ratio_ci = _bootstrap_geometric_speedup_ci(
            differences,
            samples=args.bootstrap_samples,
            seed=comparison_seed,
        )
        runtime_ratio = math.exp(float(differences.mean()))
        comparisons.append(
            {
                "optimizer_batch_size": candidate,
                "status": "ok",
                "paired_rollouts": len(differences),
                "runtime_ratio_vs_full_rollout": runtime_ratio,
                "bootstrap_95_percent_runtime_ratio_ci": ratio_ci,
                "one_sided_p_candidate_slower_than_full": (
                    _one_sided_sign_flip_pvalue(
                        differences,
                        samples=args.permutation_samples,
                        seed=comparison_seed + 1,
                    )
                ),
                "within_runtime_margin_of_full": (
                    float(ratio_ci[1]) <= 1.0 + args.runtime_margin
                ),
            }
        )

    valid_comparisons = [row for row in comparisons if row["status"] == "ok"]
    adjusted = _holm_adjust(
        [
            float(row["one_sided_p_candidate_slower_than_full"])
            for row in valid_comparisons
        ]
    )
    for row, adjusted_p in zip(valid_comparisons, adjusted):
        row["holm_adjusted_p_candidate_slower_than_full"] = adjusted_p
        row["significantly_slower_than_full"] = (
            float(row["runtime_ratio_vs_full_rollout"]) > 1.0
            and adjusted_p < args.alpha
        )

    adjacent_gains = []
    ordered = finite_candidates + [0]
    for smaller, larger in zip(ordered, ordered[1:]):
        differences = _paired_optimizer_log_time_ratios(
            measurements,
            smaller,
            larger,
            args.rollouts,
            args.repeats,
        )
        if len(differences) != args.rollouts:
            continue
        comparison_seed = args.seed + 3_000_000 + smaller
        adjacent_gains.append(
            {
                "from_optimizer_batch_size": smaller,
                "to_optimizer_batch_size": larger,
                "to_optimizer_batch_label": (
                    "full_rollout" if larger == 0 else str(larger)
                ),
                "runtime_speedup_from_increasing_batch": math.exp(
                    float(differences.mean())
                ),
                "bootstrap_95_percent_speedup_ci": (
                    _bootstrap_geometric_speedup_ci(
                        differences,
                        samples=args.bootstrap_samples,
                        seed=comparison_seed,
                    )
                ),
            }
        )

    within_margin = [
        int(row["optimizer_batch_size"])
        for row in comparisons
        if row["status"] == "ok" and row["within_runtime_margin_of_full"]
    ]
    runtime_knee = min(within_margin) if within_margin else None
    result.update(
        {
            "status": "complete",
            "summary": summary,
            "full_rollout_summary": full_row,
            "candidate_vs_full_rollout": comparisons,
            "adjacent_runtime_gains": adjacent_gains,
            "smallest_optimizer_batch_within_runtime_margin_of_full": (
                runtime_knee
            ),
            "interpretation": (
                f"The smallest tested effective Adam batch whose paired 95% "
                f"upper runtime-ratio bound is within "
                f"{100 * args.runtime_margin:.1f}% of one-step full-rollout "
                f"runtime is {runtime_knee}."
                if runtime_knee is not None
                else "No finite tested Adam batch was demonstrably within the "
                "predeclared runtime margin of the one-step full-rollout runtime."
            ),
        }
    )
    _atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


def benchmark_memory_batches_statistical(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base_cfg = _config_from_checkpoint(checkpoint_path, device)
    candidates = list(args.batch_sizes)

    result: dict = {
        "status": "pilot_running",
        "design": {
            "analysis": "two-stage paired independent-rollout benchmark",
            "rollout_games": args.rollout_games,
            "pilot_rollouts": args.pilot_rollouts,
            "pilot_repeats_per_rollout": args.pilot_repeats,
            "confirmation_rollouts": args.confirmation_rollouts,
            "confirmation_repeats_per_rollout": args.confirmation_repeats,
            "batch_sizes": candidates,
            "alpha": args.alpha,
            "one_sided_sign_flip_samples": args.permutation_samples,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "optimizer_batch": "full frozen rollout",
            "ppo_epochs_per_measurement": 1,
            "persistent_training": False,
        },
        "checkpoint": str(checkpoint_path),
        "pilot_measurements": [],
        "confirmation_measurements": [],
    }
    _atomic_write_json(output_path, result)

    def save_pilot(measurements: list[dict], completed: int) -> None:
        result["pilot_measurements"] = measurements
        result["pilot_rollouts_completed"] = completed
        _atomic_write_json(output_path, result)

    pilot = _measure_rollout_stage(
        stage="pilot",
        checkpoint_path=checkpoint_path,
        base_cfg=base_cfg,
        candidates=candidates,
        n_rollouts=args.pilot_rollouts,
        rollout_games=args.rollout_games,
        repeats=args.pilot_repeats,
        device=device,
        seed=args.seed,
        warmup=True,
        on_rollout_complete=save_pilot,
    )
    pilot_summary = _summarize_stage(
        pilot,
        candidates,
        args.pilot_rollouts,
        args.pilot_repeats,
    )
    feasible_candidates = [
        int(row["batch_size"])
        for row in pilot_summary
        if row["status"] == "ok"
    ]
    if len(feasible_candidates) < 2:
        raise RuntimeError("fewer than two batch sizes completed the pilot")
    pilot_winner_row = min(
        (row for row in pilot_summary if row["status"] == "ok"),
        key=lambda row: float(row["geometric_mean_ppo_seconds"]),
    )
    pilot_winner = int(pilot_winner_row["batch_size"])
    result.update(
        {
            "status": "confirmation_running",
            "pilot_summary": pilot_summary,
            "pilot_selected_batch_size": pilot_winner,
            "confirmation_candidates": feasible_candidates,
        }
    )
    _atomic_write_json(output_path, result)
    print(
        f"Pilot selected {pilot_winner}; confirmation uses fresh rollouts",
        flush=True,
    )

    def save_confirmation(measurements: list[dict], completed: int) -> None:
        result["confirmation_measurements"] = measurements
        result["confirmation_rollouts_completed"] = completed
        _atomic_write_json(output_path, result)

    confirmation = _measure_rollout_stage(
        stage="confirmation",
        checkpoint_path=checkpoint_path,
        base_cfg=base_cfg,
        candidates=feasible_candidates,
        n_rollouts=args.confirmation_rollouts,
        rollout_games=args.rollout_games,
        repeats=args.confirmation_repeats,
        device=device,
        seed=args.seed + 1_000_000,
        warmup=False,
        on_rollout_complete=save_confirmation,
    )
    confirmation_summary = _summarize_stage(
        confirmation,
        feasible_candidates,
        args.confirmation_rollouts,
        args.confirmation_repeats,
    )

    comparisons = []
    for other in feasible_candidates:
        if other == pilot_winner:
            continue
        differences = _paired_log_time_differences(
            confirmation,
            pilot_winner,
            other,
            args.confirmation_rollouts,
            args.confirmation_repeats,
        )
        if len(differences) != args.confirmation_rollouts:
            comparisons.append(
                {
                    "other_batch_size": other,
                    "status": "oom_or_incomplete",
                    "paired_rollouts": len(differences),
                }
            )
            continue
        comparison_seed = args.seed + 2_000_000 + other
        comparisons.append(
            {
                "other_batch_size": other,
                "status": "ok",
                "paired_rollouts": len(differences),
                "pilot_winner_geometric_speedup": math.exp(
                    float(differences.mean())
                ),
                "bootstrap_95_percent_speedup_ci": (
                    _bootstrap_geometric_speedup_ci(
                        differences,
                        samples=args.bootstrap_samples,
                        seed=comparison_seed,
                    )
                ),
                "one_sided_sign_flip_p": _one_sided_sign_flip_pvalue(
                    differences,
                    samples=args.permutation_samples,
                    seed=comparison_seed + 1,
                ),
            }
        )

    valid_comparisons = [row for row in comparisons if row["status"] == "ok"]
    adjusted = _holm_adjust(
        [float(row["one_sided_sign_flip_p"]) for row in valid_comparisons]
    )
    for row, adjusted_p in zip(valid_comparisons, adjusted):
        row["holm_adjusted_p"] = adjusted_p
        row["pilot_winner_significantly_faster"] = (
            float(row["pilot_winner_geometric_speedup"]) > 1.0
            and adjusted_p < args.alpha
        )

    indistinguishable = [
        int(row["other_batch_size"])
        for row in comparisons
        if row["status"] == "ok"
        and not row.get("pilot_winner_significantly_faster", False)
    ]
    confirmation_infeasible = [
        int(row["other_batch_size"])
        for row in comparisons
        if row["status"] != "ok"
    ]
    uniquely_confirmed = not indistinguishable
    result.update(
        {
            "status": "complete",
            "confirmation_summary": confirmation_summary,
            "confirmatory_comparisons": comparisons,
            "pilot_winner_uniquely_confirmed_fastest": uniquely_confirmed,
            "statistically_indistinguishable_or_faster_than_pilot_winner": (
                indistinguishable
            ),
            "confirmation_infeasible_batch_sizes": confirmation_infeasible,
            "recommended_batch_size": pilot_winner if uniquely_confirmed else None,
            "interpretation": (
                "The pilot-selected batch is faster than every other feasible "
                "candidate at the family-wise alpha level."
                if uniquely_confirmed
                else "The experiment did not establish one unique fastest batch "
                "at the family-wise alpha level; inspect the listed tied or "
                "faster candidates and confidence intervals."
            ),
        }
    )
    _atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


def benchmark_memory_batches(args: argparse.Namespace) -> int:
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base_cfg = _config_from_checkpoint(checkpoint_path, device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy, optimizer, pass_policy, gt_policy = _load_training_bundle(
        checkpoint_path,
        base_cfg,
        device,
    )
    rollout = collect_rollouts_history(
        policy,
        args.rollout_games,
        device,
        n_workers=1,
        encoder_cfg=_encoder_cfg(base_cfg),
        model_cfg=_model_cfg(base_cfg),
        use_rollout_bucketing=base_cfg.use_rollout_bucketing,
        with_calling=True,
        gt_policy=gt_policy,
        pass_policy=pass_policy,
        block=True,
        disable_gt_explore=False,
    )
    del policy, optimizer, pass_policy, gt_policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Discard one warmup so CUDA initialization is not charged to a candidate.
    _run_memory_candidate(
        checkpoint_path,
        base_cfg,
        rollout,
        min(args.batch_sizes),
        device,
        args.seed + 1,
    )

    measurements: list[dict[str, float | int | str]] = []
    # Alternate ascending/descending passes to reduce bias from GPU warm-up,
    # temperature, or shared-host load.  A given repeat uses the same seed for
    # every candidate, so microbatch size remains the only changed variable.
    for repeat in range(args.repeats):
        batch_sizes = list(args.batch_sizes)
        if repeat % 2:
            batch_sizes.reverse()
        for batch_size in batch_sizes:
            row = _run_memory_candidate(
                checkpoint_path,
                base_cfg,
                rollout,
                batch_size,
                device,
                args.seed + 100 + repeat,
            )
            row["repeat"] = repeat
            measurements.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row["status"] == "oom":
                break

    summary = []
    for batch_size in args.batch_sizes:
        rows = [
            row
            for row in measurements
            if row["batch_size"] == batch_size and row["status"] == "ok"
        ]
        if len(rows) != args.repeats:
            summary.append({"batch_size": batch_size, "status": "oom"})
            continue
        summary.append(
            {
                "batch_size": batch_size,
                "status": "ok",
                "median_ppo_seconds": statistics.median(
                    float(row["ppo_seconds"]) for row in rows
                ),
                "median_transitions_per_second": statistics.median(
                    float(row["transitions_per_second"]) for row in rows
                ),
                "max_peak_allocated_mb": max(
                    float(row["peak_allocated_mb"]) for row in rows
                ),
                "max_peak_reserved_mb": max(
                    float(row["peak_reserved_mb"]) for row in rows
                ),
            }
        )

    successful = [row for row in summary if row["status"] == "ok"]
    eligible = [
        row
        for row in successful
        if int(row["batch_size"]) <= args.max_sweep_microbatch
    ]
    if not eligible:
        raise RuntimeError("no memory microbatch completed within the sweep cap")
    fastest_overall = min(
        successful,
        key=lambda row: float(row["median_ppo_seconds"]),
    )
    selected = min(
        eligible,
        key=lambda row: float(row["median_ppo_seconds"]),
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "rollout_games": args.rollout_games,
        "transitions": len(rollout["obs"]),
        "repeats": args.repeats,
        "measurements": measurements,
        "summary": summary,
        "fastest_overall_batch_size": fastest_overall["batch_size"],
        "max_sweep_microbatch": args.max_sweep_microbatch,
        "selected_sweep_batch_size": selected["batch_size"],
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark-memory")
    benchmark.add_argument("checkpoint")
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--rollout-games", type=int, default=512)
    benchmark.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096],
    )
    benchmark.add_argument("--repeats", type=int, default=2)
    benchmark.add_argument("--max-sweep-microbatch", type=int, default=1024)
    benchmark.add_argument("--seed", type=int, default=20260812)
    benchmark.set_defaults(func=benchmark_memory_batches)

    statistical = subparsers.add_parser("benchmark-memory-statistical")
    statistical.add_argument("checkpoint")
    statistical.add_argument("--output", required=True)
    statistical.add_argument("--device", default="cuda")
    statistical.add_argument("--rollout-games", type=int, default=512)
    statistical.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[
            256,
            512,
            768,
            1024,
            1280,
            1536,
            1792,
            2048,
            2560,
            3072,
            4096,
            6144,
            8192,
        ],
    )
    statistical.add_argument("--pilot-rollouts", type=int, default=8)
    statistical.add_argument("--pilot-repeats", type=int, default=2)
    statistical.add_argument("--confirmation-rollouts", type=int, default=24)
    statistical.add_argument("--confirmation-repeats", type=int, default=3)
    statistical.add_argument("--permutation-samples", type=int, default=200_000)
    statistical.add_argument("--bootstrap-samples", type=int, default=50_000)
    statistical.add_argument("--alpha", type=float, default=0.05)
    statistical.add_argument("--seed", type=int, default=20260812)
    statistical.set_defaults(func=benchmark_memory_batches_statistical)

    optimizer_runtime = subparsers.add_parser(
        "benchmark-optimizer-runtime-statistical"
    )
    optimizer_runtime.add_argument("checkpoint")
    optimizer_runtime.add_argument("--output", required=True)
    optimizer_runtime.add_argument("--device", default="cuda")
    optimizer_runtime.add_argument("--rollout-games", type=int, default=1024)
    optimizer_runtime.add_argument("--memory-batch-size", type=int, default=512)
    optimizer_runtime.add_argument(
        "--optimizer-batch-sizes",
        type=int,
        nargs="+",
        default=[
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            12288,
            16384,
            24576,
            32768,
            49152,
            65536,
            0,
        ],
        help="Effective Adam batches; zero means the full rollout",
    )
    optimizer_runtime.add_argument("--rollouts", type=int, default=24)
    optimizer_runtime.add_argument("--repeats", type=int, default=2)
    optimizer_runtime.add_argument("--runtime-margin", type=float, default=0.02)
    optimizer_runtime.add_argument(
        "--permutation-samples", type=int, default=200_000
    )
    optimizer_runtime.add_argument("--bootstrap-samples", type=int, default=50_000)
    optimizer_runtime.add_argument("--alpha", type=float, default=0.05)
    optimizer_runtime.add_argument("--seed", type=int, default=20260813)
    optimizer_runtime.set_defaults(func=benchmark_optimizer_runtime_statistical)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"benchmark-memory", "benchmark-memory-statistical"}:
        if args.rollout_games <= 0:
            raise ValueError("rollout games must be positive")
        if any(batch_size <= 0 for batch_size in args.batch_sizes):
            raise ValueError("batch sizes must be positive")
        if len(set(args.batch_sizes)) != len(args.batch_sizes):
            raise ValueError("batch sizes must be unique")
    if args.command == "benchmark-memory":
        if args.repeats <= 0:
            raise ValueError("repeats must be positive")
    if args.command == "benchmark-memory-statistical":
        positive_counts = (
            args.pilot_rollouts,
            args.pilot_repeats,
            args.confirmation_rollouts,
            args.confirmation_repeats,
            args.permutation_samples,
            args.bootstrap_samples,
        )
        if any(value <= 0 for value in positive_counts):
            raise ValueError("rollout, repeat, and resample counts must be positive")
        if not 0.0 < args.alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
    if args.command == "benchmark-optimizer-runtime-statistical":
        positive_counts = (
            args.rollout_games,
            args.memory_batch_size,
            args.rollouts,
            args.repeats,
            args.permutation_samples,
            args.bootstrap_samples,
        )
        if any(value <= 0 for value in positive_counts):
            raise ValueError("games, batches, repeats, and resamples must be positive")
        if len(set(args.optimizer_batch_sizes)) != len(args.optimizer_batch_sizes):
            raise ValueError("optimizer batch sizes must be unique")
        if args.optimizer_batch_sizes.count(0) != 1:
            raise ValueError("optimizer batch sizes must contain one full-rollout zero")
        if any(value < 0 for value in args.optimizer_batch_sizes):
            raise ValueError("optimizer batch sizes cannot be negative")
        if not 0.0 < args.alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
        if not 0.0 < args.runtime_margin < 1.0:
            raise ValueError("runtime margin must be between zero and one")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
