"""Replicated short-horizon PPO replay-drift curve.

Every branch starts from the same mature checkpoint and performs independent
production-style PPO updates.  Ages are scored on one held-out fixed rollout,
so the crossed bootstrap can vary both training path and evaluation-state
blocks.  This estimates typical policy drift across update sequences rather
than conditioning on one historical checkpoint pair.
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.checkpoints import resolve_checkpoint
from training.model_gt import TichuGrandTichuPolicy
from training.model_passing import TichuPassingPolicy
from training.paired_checkpoint_eval import _atomic_write_json
from training.play_training import PASS_BUFFER_KEYS
from training.ppo_history import ppo_update_history
from training.ppo_passing import ppo_update_passing
from training.replay_policy_drift import (
    PRIMARY_METRICS,
    _atomic_save_npz,
    _distribution_metrics,
    _quantiles,
    _score_policy,
)
from training.rollout_embed import set_action_vec_cache_enabled
from training.rollout_history import collect_rollouts_history
from training.train_history import (
    _build_policy,
    _encoder_cfg,
    _make_optimizer,
    _model_cfg,
    _prepare_policy_for_resume,
    load_checkpoint,
)


def _parse_ages(text: str, updates: int) -> list[int]:
    ages = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not ages or ages[0] <= 0 or ages[-1] > updates:
        raise ValueError("ages must be unique positive integers no larger than updates")
    return ages


def _build_runtime(
    checkpoint: Path,
    cfg,
    device: torch.device,
) -> dict[str, Any]:
    policy = _build_policy(cfg).to(device)
    _prepare_policy_for_resume(policy, str(checkpoint), device)
    optimizer = _make_optimizer(policy, cfg)
    pass_policy = TichuPassingPolicy(
        hidden_dim=cfg.pass_hidden_dim,
        n_layers=cfg.pass_n_layers,
    ).to(device)
    pass_optimizer = torch.optim.Adam(pass_policy.parameters(), lr=cfg.pass_lr)
    gt_policy = TichuGrandTichuPolicy(
        hidden_dim=cfg.gt_hidden_dim,
        n_layers=cfg.gt_n_layers,
    ).to(device)
    gt_optimizer = torch.optim.Adam(gt_policy.parameters(), lr=cfg.gt_lr)
    games = load_checkpoint(
        policy,
        optimizer,
        str(checkpoint),
        device,
        pass_policy,
        pass_optimizer,
        gt_policy,
        gt_optimizer,
    )
    return {
        "policy": policy,
        "optimizer": optimizer,
        "pass_policy": pass_policy,
        "pass_optimizer": pass_optimizer,
        "gt_policy": gt_policy,
        "gt_optimizer": gt_optimizer,
        "games_played": games,
    }


def _collect(
    runtime: dict[str, Any],
    cfg,
    device: torch.device,
    pool,
    *,
    games: int,
    seed: int,
) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    return collect_rollouts_history(
        runtime["policy"],
        games,
        device,
        pool=pool,
        n_workers=cfg.n_workers,
        encoder_cfg=_encoder_cfg(cfg),
        model_cfg=_model_cfg(cfg),
        use_rollout_bucketing=cfg.use_rollout_bucketing,
        with_calling=True,
        gt_policy=runtime["gt_policy"],
        pass_policy=runtime["pass_policy"],
        block=True,
        disable_gt_explore=False,
        seed=seed,
    )


def _update_passing(
    runtime: dict[str, Any],
    pass_buffer: list[dict[str, np.ndarray]],
    cfg,
    device: torch.device,
) -> dict[str, float] | None:
    if not pass_buffer:
        return None
    combined = {
        key: np.concatenate([buffer[key] for buffer in pass_buffer])
        for key in PASS_BUFFER_KEYS
    }
    return ppo_update_passing(
        runtime["pass_policy"],
        runtime["pass_optimizer"],
        {
            "pass_obs": combined["pass_train_obs"],
            "hand_cards": combined["pass_train_hand_cards"],
            "choices": combined["pass_train_choices"],
            "log_probs": combined["pass_train_log_probs"],
            "returns": combined["pass_train_returns"],
        },
        cfg,
        device,
    )


def _metric_block_sums(
    metrics: dict[str, np.ndarray],
    game_ids: np.ndarray,
    discretionary: np.ndarray,
    *,
    games_per_block: int,
    block_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    block_ids = game_ids[discretionary] // games_per_block
    counts = np.bincount(block_ids, minlength=block_count).astype(np.float64)
    sums = np.stack(
        [
            np.bincount(
                block_ids,
                weights=metrics[metric][discretionary],
                minlength=block_count,
            ).astype(np.float64)
            for metric in PRIMARY_METRICS
        ],
        axis=-1,
    )
    return sums, counts


def _crossed_bootstrap(
    block_sums: np.ndarray,
    block_counts: np.ndarray,
    *,
    samples: int,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Bootstrap independent training branches and shared state blocks.

    ``block_sums`` is (branches, ages, blocks, metrics). State blocks are
    resampled once per draw and shared across resampled branches, preserving
    the paired fixed-state design.
    """
    if block_sums.ndim != 4:
        raise ValueError("block_sums must have shape branches x ages x blocks x metrics")
    branches, ages, blocks, metrics = block_sums.shape
    if branches < 2 or blocks < 2:
        raise ValueError("crossed bootstrap needs at least two branches and blocks")
    if block_counts.shape != (blocks,) or np.any(block_counts <= 0):
        raise ValueError("block_counts must be positive with one entry per block")

    branch_means = block_sums.sum(axis=2) / block_counts.sum()
    point = branch_means.mean(axis=0)
    draws = np.empty((samples, ages, metrics), dtype=np.float64)
    rng = np.random.default_rng(seed)
    chunk = 128
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        count = stop - start
        branch_indices = rng.integers(0, branches, size=(count, branches))
        block_indices = rng.integers(0, blocks, size=(count, blocks))
        # Average resampled branches for every original state block.
        branch_averaged = block_sums[branch_indices].mean(axis=1)
        selected_blocks = np.take_along_axis(
            branch_averaged,
            block_indices[:, None, :, None],
            axis=2,
        )
        draws[start:stop] = selected_blocks.sum(axis=2) / (
            block_counts[block_indices].sum(axis=1)[:, None, None]
        )

    ordinary = np.stack(
        (
            np.quantile(draws, alpha / 2.0, axis=0),
            np.quantile(draws, 1.0 - alpha / 2.0, axis=0),
        ),
        axis=-1,
    )
    family_size = ages * metrics
    simultaneous = np.stack(
        (
            np.quantile(draws, alpha / (2.0 * family_size), axis=0),
            np.quantile(draws, 1.0 - alpha / (2.0 * family_size), axis=0),
        ),
        axis=-1,
    )
    adjacent_draws = np.diff(
        np.concatenate((np.zeros((samples, 1, metrics)), draws), axis=1),
        axis=1,
    )
    adjacent_simultaneous = np.stack(
        (
            np.quantile(
                adjacent_draws, alpha / (2.0 * family_size), axis=0
            ),
            np.quantile(
                adjacent_draws, 1.0 - alpha / (2.0 * family_size), axis=0
            ),
        ),
        axis=-1,
    )
    return branch_means, draws, {
        "point": point,
        "ordinary_ci": ordinary,
        "simultaneous_ci": simultaneous,
        "adjacent_point": np.diff(
            np.concatenate((np.zeros((1, metrics)), point), axis=0), axis=0
        ),
        "adjacent_simultaneous_ci": adjacent_simultaneous,
    }


def _ratio_summary(ratio: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(ratio.mean()),
        "std": float(ratio.std(ddof=1)),
        "quantiles": _quantiles(ratio),
        "p999": float(np.quantile(ratio, 0.999)),
        "maximum": float(ratio.max()),
        "effective_sample_size_fraction": float(
            ratio.sum() ** 2 / (np.square(ratio).sum() * len(ratio))
        ),
    }


def _rescore_max_tolerance(device: torch.device) -> float:
    # CPU and CUDA kernels use different reduction orders when the same
    # policy is evaluated in rollout buckets versus fixed-size scoring
    # batches.  The bulk tolerance remains strict on both devices; permit a
    # few extremely low-probability CPU rows to move slightly farther in log
    # space while staying well inside PPO's nearest log clipping boundary.
    return 0.05 if device.type == "cpu" else 0.02


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    if args.branches < 2:
        raise ValueError("at least two independent branches are required")
    ages = _parse_ages(args.ages, args.updates)
    if args.eval_games % args.eval_games_per_block:
        raise ValueError("eval-games must divide into complete state blocks")
    checkpoint = resolve_checkpoint(args.checkpoint)
    raw_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base_games = int(raw_checkpoint["games_played"])
    cfg = copy.deepcopy(raw_checkpoint["config"])
    cfg.device = args.device
    cfg.worker_device = args.device
    cfg.n_workers = args.workers
    cfg.games_per_update = args.games_per_update
    cfg.batch_size = args.batch
    cfg.optimizer_batch_size = args.optimizer_batch
    cfg.n_epochs = args.epochs
    cfg.synchronous = True
    cfg.use_amp = False
    cfg.profile_update_timings = False
    set_action_vec_cache_enabled(cfg.use_action_vec_cache)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)

    final_games = base_games + args.updates * args.games_per_update
    if cfg.train_gt and cfg.gt_probe_every > 0:
        for update in range(1, args.updates + 1):
            next_games = base_games + update * args.games_per_update
            if next_games % cfg.gt_probe_every < cfg.games_per_update:
                raise ValueError("experiment window crosses a GT probe boundary")

    output = Path(args.output)
    raw_output = (
        Path(args.raw_output) if args.raw_output else output.with_suffix(".npz")
    )
    result: dict[str, object] = {
        "design": {
            "checkpoint": str(checkpoint),
            "base_games": base_games,
            "final_games": final_games,
            "independent_training_branches": args.branches,
            "updates_per_branch": args.updates,
            "measured_ages_updates": ages,
            "games_per_update": args.games_per_update,
            "production_sized_updates": args.games_per_update == 1024,
            "memory_batch": args.batch,
            "optimizer_batch": args.optimizer_batch,
            "ppo_epochs": args.epochs,
            "workers": args.workers,
            "eval_games": args.eval_games,
            "eval_games_per_block": args.eval_games_per_block,
            "eval_state_blocks": args.eval_games // args.eval_games_per_block,
            "fixed_eval_states_shared_across_all_branches_and_ages": True,
            "bootstrap_samples": args.bootstrap_samples,
            "crossed_bootstrap_training_branches_and_state_blocks": True,
            "simultaneous_family_size": len(ages) * len(PRIMARY_METRICS),
            "alpha": args.alpha,
            "clip_epsilon": args.clip_eps,
            "seed": args.seed,
            "device": str(device),
        },
        "primary_metrics": list(PRIMARY_METRICS),
        "branches": [],
        "completed_branches": 0,
        "complete": False,
    }
    _atomic_write_json(output, result)

    pool = None
    if args.workers > 1:
        pool = mp.get_context("spawn").Pool(args.workers)
    started = time.perf_counter()
    completed_block_sums: list[np.ndarray] = []
    completed_ratio_summaries: list[list[dict[str, object]]] = []
    try:
        print("Collecting held-out fixed evaluation states", flush=True)
        base_runtime = _build_runtime(checkpoint, cfg, device)
        eval_rollout = _collect(
            base_runtime,
            cfg,
            device,
            pool,
            games=args.eval_games,
            seed=args.seed + 7,
        )
        behavior_score = _score_policy(
            base_runtime["policy"], eval_rollout, device, args.eval_batch
        )
        validation_error = np.abs(
            behavior_score["selected_log_probs"] - eval_rollout["log_probs"]
        )
        validation_p99 = float(np.quantile(validation_error, 0.99))
        validation_max = float(validation_error.max())
        maximum_tolerance = _rescore_max_tolerance(device)
        result["behavior_rescore_validation"] = {
            "p99_absolute_log_probability_error": validation_p99,
            "maximum_absolute_log_probability_error": validation_max,
            "fraction_above_1e_4": float((validation_error > 1e-4).mean()),
            "fraction_above_1e_3": float((validation_error > 1e-3).mean()),
            "p99_tolerance": 1e-4,
            "maximum_tolerance": maximum_tolerance,
            "passed": bool(
                validation_p99 < 1e-4
                and validation_max < maximum_tolerance
            ),
        }
        if not result["behavior_rescore_validation"]["passed"]:
            _atomic_write_json(output, result)
            raise RuntimeError(
                "behavior policy rescore does not reproduce rollout log-probs; "
                f"maximum error={validation_max:.9g}, p99={validation_p99:.9g}"
            )
        game_ids = eval_rollout["history_game_ids"].astype(np.int64, copy=False)
        discretionary = eval_rollout["legal_lens"] > 1
        block_count = args.eval_games // args.eval_games_per_block
        expected_blocks = np.arange(block_count)
        observed_blocks = np.unique(
            game_ids[discretionary] // args.eval_games_per_block
        )
        if not np.array_equal(observed_blocks, expected_blocks):
            raise ValueError("every evaluation block needs discretionary states")
        reference_counts = None
        result["evaluation_population"] = {
            "total_decisions": int(len(discretionary)),
            "discretionary_decisions": int(discretionary.sum()),
            "discretionary_fraction": float(discretionary.mean()),
            "forced_decisions": int((~discretionary).sum()),
        }
        del base_runtime
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for branch_index in range(args.branches):
            branch_seed = args.seed + 1_000_000 * (branch_index + 1)
            print(
                f"\nBranch {branch_index + 1}/{args.branches} "
                f"(seed {branch_seed})",
                flush=True,
            )
            torch.manual_seed(branch_seed)
            np.random.seed(branch_seed % (2**32))
            random.seed(branch_seed)
            runtime = _build_runtime(checkpoint, cfg, device)
            pass_buffer: list[dict[str, np.ndarray]] = []
            branch_age_sums: list[np.ndarray] = []
            branch_ratios: list[dict[str, object]] = []
            branch_record: dict[str, object] = {
                "branch": branch_index,
                "seed": branch_seed,
                "updates": [],
                "ages": [],
            }

            for update in range(1, args.updates + 1):
                update_seed = branch_seed + 10_000 * update
                update_started = time.perf_counter()
                rollout = _collect(
                    runtime,
                    cfg,
                    device,
                    pool,
                    games=args.games_per_update,
                    seed=update_seed,
                )
                if "pass_train_obs" in rollout:
                    pass_buffer.append(
                        {key: rollout[key] for key in PASS_BUFFER_KEYS}
                    )
                torch.manual_seed(update_seed + 1)
                np.random.seed((update_seed + 1) % (2**32))
                ppo_stats = ppo_update_history(
                    runtime["policy"],
                    runtime["optimizer"],
                    rollout,
                    cfg,
                    device,
                )
                runtime["games_played"] += args.games_per_update
                pass_updated = False
                pass_stats = None
                if (
                    runtime["games_played"] % cfg.pass_every
                    < args.games_per_update
                ):
                    pass_stats = _update_passing(
                        runtime, pass_buffer, cfg, device
                    )
                    pass_buffer = []
                    pass_updated = pass_stats is not None
                update_record = {
                    "update": update,
                    "games_played": runtime["games_played"],
                    "transitions": int(len(rollout["obs"])),
                    "policy_loss": float(ppo_stats["policy_loss"]),
                    "value_loss": float(ppo_stats["value_loss"]),
                    "entropy": float(ppo_stats["entropy"]),
                    "approx_kl": float(ppo_stats["approx_kl"]),
                    "clip_frac": float(ppo_stats["clip_frac"]),
                    "ppo_seconds": float(ppo_stats["ppo_seconds"]),
                    "pass_updated_after_main_policy": pass_updated,
                    "elapsed_seconds": time.perf_counter() - update_started,
                }
                if pass_stats is not None:
                    update_record["pass_approx_kl"] = float(
                        pass_stats["pass_approx_kl"]
                    )
                branch_record["updates"].append(update_record)
                del rollout

                if update in ages:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    candidate_score = _score_policy(
                        runtime["policy"],
                        eval_rollout,
                        device,
                        args.eval_batch,
                    )
                    metrics = _distribution_metrics(
                        behavior_score,
                        candidate_score,
                        eval_rollout,
                        args.clip_eps,
                    )
                    sums, counts = _metric_block_sums(
                        metrics,
                        game_ids,
                        discretionary,
                        games_per_block=args.eval_games_per_block,
                        block_count=block_count,
                    )
                    if reference_counts is None:
                        reference_counts = counts
                    elif not np.array_equal(reference_counts, counts):
                        raise RuntimeError("evaluation block counts changed")
                    ratio_summary = _ratio_summary(
                        metrics["ratio"][discretionary]
                    )
                    means = sums.sum(axis=0) / counts.sum()
                    branch_age_sums.append(sums)
                    branch_ratios.append(ratio_summary)
                    branch_record["ages"].append(
                        {
                            "age_updates": update,
                            "games_played": runtime["games_played"],
                            "metric_means": {
                                metric: float(means[index])
                                for index, metric in enumerate(PRIMARY_METRICS)
                            },
                            "importance_ratio": ratio_summary,
                        }
                    )
                    print(
                        f"  age {update:>2}: "
                        f"clip={means[PRIMARY_METRICS.index('outside_clip_range')]:.3f} "
                        f"sat={means[PRIMARY_METRICS.index('ppo_objective_saturated')]:.3f} "
                        f"TV={means[PRIMARY_METRICS.index('total_variation')]:.3f}",
                        flush=True,
                    )

            completed_block_sums.append(np.stack(branch_age_sums, axis=0))
            completed_ratio_summaries.append(branch_ratios)
            result["branches"].append(branch_record)
            result["completed_branches"] = len(completed_block_sums)
            result["elapsed_seconds"] = time.perf_counter() - started
            _atomic_write_json(output, result)
            _atomic_save_npz(
                raw_output,
                {
                    "block_sums": np.stack(completed_block_sums),
                    "block_counts": reference_counts,
                    "ages": np.asarray(ages, dtype=np.int64),
                    "metric_names": np.asarray(PRIMARY_METRICS),
                },
            )
            del runtime
            if device.type == "cuda":
                torch.cuda.empty_cache()

        block_sums = np.stack(completed_block_sums)
        branch_means, bootstrap_draws, inference = _crossed_bootstrap(
            block_sums,
            reference_counts,
            samples=args.bootstrap_samples,
            alpha=args.alpha,
            seed=args.seed + 99,
        )
        result["age_results"] = []
        for age_index, age in enumerate(ages):
            age_result = {
                "age_updates": age,
                "age_games": age * args.games_per_update,
                "metrics": {},
                "importance_ratio": {},
            }
            for metric_index, metric in enumerate(PRIMARY_METRICS):
                age_result["metrics"][metric] = {
                    "mean": float(inference["point"][age_index, metric_index]),
                    "ordinary_ci": inference["ordinary_ci"][
                        age_index, metric_index
                    ].tolist(),
                    "simultaneous_ci": inference["simultaneous_ci"][
                        age_index, metric_index
                    ].tolist(),
                    "between_branch_std": float(
                        branch_means[:, age_index, metric_index].std(ddof=1)
                    ),
                    "branch_means": branch_means[
                        :, age_index, metric_index
                    ].tolist(),
                    "increment_from_previous_measured_age": float(
                        inference["adjacent_point"][age_index, metric_index]
                    ),
                    "increment_simultaneous_ci": inference[
                        "adjacent_simultaneous_ci"
                    ][age_index, metric_index].tolist(),
                }
            for ratio_metric in (
                "effective_sample_size_fraction",
                "mean",
                "std",
                "p999",
                "maximum",
            ):
                values = np.asarray(
                    [
                        completed_ratio_summaries[branch][age_index][ratio_metric]
                        for branch in range(args.branches)
                    ],
                    dtype=np.float64,
                )
                age_result["importance_ratio"][ratio_metric] = {
                    "mean_across_branches": float(values.mean()),
                    "between_branch_std": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "branch_values": values.tolist(),
                }
            result["age_results"].append(age_result)

        result["raw_output"] = str(raw_output)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["complete"] = True
        _atomic_write_json(output, result)
        _atomic_save_npz(
            raw_output,
            {
                "block_sums": block_sums,
                "block_counts": reference_counts,
                "branch_means": branch_means,
                "bootstrap_draws": bootstrap_draws,
                "ages": np.asarray(ages, dtype=np.int64),
                "metric_names": np.asarray(PRIMARY_METRICS),
            },
        )
        return result
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replicated short-horizon replay-drift curve"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--branches", type=int, default=12)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--ages", default="2,3,4,5,6,7,8,9,10")
    parser.add_argument("--games-per-update", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--optimizer-batch", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-games", type=int, default=1024)
    parser.add_argument("--eval-games-per-block", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    result = run_experiment(args)
    print("\nReplicated replay-age curve:", flush=True)
    for age in result["age_results"]:
        metrics = age["metrics"]
        print(
            f"  age {age['age_updates']:>2}: "
            f"clip={metrics['outside_clip_range']['mean']:.3f}, "
            f"sat={metrics['ppo_objective_saturated']['mean']:.3f}, "
            f"TV={metrics['total_variation']['mean']:.3f}, "
            f"argmax={metrics['argmax_changed']['mean']:.3f}",
            flush=True,
        )
    print(f"Wrote {args.output} and {result['raw_output']}", flush=True)


if __name__ == "__main__":
    main()
