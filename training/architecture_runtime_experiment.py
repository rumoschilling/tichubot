"""Paired runtime attribution for the standard and compact history policies.

The benchmark is intentionally not a training run.  Each block collects an
independent rollout for both architectures with the same seed, alternates
condition order, and replays one shared frozen rollout through both PPO
implementations.  Model and populated Adam state are restored before every
timed PPO update, so no measured update survives into the next block.

Inference uses the production four-worker synchronous CUDA path.  PPO holds
the measured compact production settings (memory batch 512, effective Adam
batch 2048, two epochs) fixed for both architectures so architecture is the
only PPO variable.  Warm-up blocks are excluded from every statistical test.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np
import torch

from training.config import TrainConfig
from training.init_gt import init_gt_smartly
from training.model_gt import TichuGrandTichuPolicy
from training.model_passing import TichuPassingPolicy
from training.optimizer_batch_experiment import (
    _bootstrap_geometric_speedup_ci,
    _holm_adjust,
    _one_sided_sign_flip_pvalue,
)
from training.ppo_history import ppo_update_history
from training import rollout_history
from training.rollout_history import collect_rollouts_history
from training.train_history import (
    _build_policy,
    _encoder_cfg,
    _make_optimizer,
    _model_cfg,
)


ARCHITECTURES = ("standard", "compact")
ROLLOUT_WORKER_TIMELINE = (
    "worker_setup_s",
    "initialization_s",
    "immediate_decisions_s",
    "legal_actions_s",
    "obs_encode_s",
    "action_vec_s",
    "history_encode_s",
    "forward_s",
    "env_step_s",
    "buffer_compile_s",
)
ROLLOUT_NESTED_DIAGNOSTICS = (
    "specialist_preplay_s",
    "specialist_postplay_s",
    "buffer_pass1_s",
    "buffer_legal_concat_s",
    "buffer_history_rebuild_s",
)
PPO_STAGES = (
    "setup",
    "epoch_order",
    "microbatch_prep",
    "forward_loss",
    "backward",
    "metrics",
    "specialists",
    "optimizer",
)


def _architecture_config(name: str, device: torch.device) -> TrainConfig:
    cfg = TrainConfig()
    cfg.device = str(device)
    cfg.synchronous = True
    cfg.worker_device = str(device)
    cfg.n_workers = 4
    cfg.games_per_update = 1024
    cfg.batch_size = 512
    cfg.optimizer_batch_size = 2048
    cfg.n_epochs = 2
    cfg.use_amp = False
    cfg.use_rollout_bucketing = True
    cfg.use_ppo_bucketing = True
    cfg.use_action_vec_cache = True
    cfg.profile_update_timings = True

    if name == "standard":
        cfg.hidden_dim = 256
        cfg.n_layers = 4
        cfg.trunk_dims = None
        cfg.mlp_activation = "tanh"
        cfg.mlp_layer_norm = False
        cfg.embed_dim = 64
        cfg.hist_d_model = 128
        cfg.hist_nhead = 4
        cfg.hist_n_layers = 2
        cfg.hist_ffn_dim = 256
        cfg.hist_norm_first = False
        cfg.hist_activation = "relu"
        cfg.split_actor_critic_history = False
        cfg.static_attention = False
    elif name == "compact":
        cfg.hidden_dim = 128
        cfg.n_layers = 3
        cfg.trunk_dims = (192, 160, 128)
        cfg.mlp_activation = "silu"
        cfg.mlp_layer_norm = True
        cfg.embed_dim = 32
        cfg.hist_d_model = 96
        cfg.hist_nhead = 4
        cfg.hist_n_layers = 2
        cfg.hist_ffn_dim = 192
        cfg.hist_norm_first = True
        cfg.hist_activation = "gelu"
        cfg.split_actor_critic_history = True
        cfg.static_attention = True
    else:
        raise ValueError(f"unknown architecture: {name}")
    return cfg


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _build_runtime(name: str, cfg: TrainConfig, seed: int, device: torch.device):
    torch.manual_seed(seed)
    rollout_policy = _build_policy(cfg).to(device)
    # Keep the environment trajectory identical across architectures while
    # still executing every trunk/history/action-encoder operation. A zero
    # query makes every legal action equiprobable; zero specialist heads do
    # the same for dragon/wish decisions. Workers are reseeded before each
    # condition, so both policies consume the same samples and visit the same
    # states. Runtime differences therefore cannot come from trajectory shape.
    with torch.no_grad():
        for module in (
            rollout_policy.query_head,
            rollout_policy.dragon_head,
            rollout_policy.wish_head,
        ):
            for parameter in module.parameters():
                parameter.zero_()
    training_policy = _build_policy(cfg).to(device)
    training_policy.load_state_dict(rollout_policy.state_dict())
    optimizer = _make_optimizer(training_policy, cfg)
    return {
        "name": name,
        "cfg": cfg,
        "rollout_policy": rollout_policy,
        "training_policy": training_policy,
        "optimizer": optimizer,
        "baseline_model": None,
        "baseline_optimizer": None,
        "parameters": sum(
            parameter.numel() for parameter in rollout_policy.parameters()
        ),
    }


def _capture_populated_baseline(runtime: dict[str, Any]) -> None:
    runtime["baseline_model"] = {
        key: value.detach().cpu().clone()
        for key, value in runtime["training_policy"].state_dict().items()
    }
    runtime["baseline_optimizer"] = _to_cpu(
        runtime["optimizer"].state_dict()
    )


def _restore_populated_baseline(runtime: dict[str, Any]) -> None:
    if runtime["baseline_model"] is None:
        return
    runtime["training_policy"].load_state_dict(runtime["baseline_model"])
    runtime["optimizer"].load_state_dict(
        copy.deepcopy(runtime["baseline_optimizer"])
    )


def _rollout_measurement(
    runtime: dict[str, Any],
    *,
    games: int,
    seed: int,
    device: torch.device,
    pool,
    pass_policy,
    gt_policy,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    cfg = runtime["cfg"]
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    started = time.perf_counter()
    rollout = collect_rollouts_history(
        runtime["rollout_policy"],
        games,
        device,
        pool=pool,
        n_workers=cfg.n_workers,
        encoder_cfg=_encoder_cfg(cfg),
        model_cfg=_model_cfg(cfg),
        use_rollout_bucketing=cfg.use_rollout_bucketing,
        with_calling=True,
        gt_policy=gt_policy,
        pass_policy=pass_policy,
        block=True,
        disable_gt_explore=False,
        seed=seed,
    )
    wall_seconds = time.perf_counter() - started
    profile = dict(rollout_history.ROLLOUT_STATS)
    transitions = len(rollout["obs"])
    worker_count = max(int(profile.get("worker_count", 0)), 1)
    worker_wall_mean = float(profile["worker_wall_s"]) / worker_count
    worker_values = {
        f"rollout_{key[:-2]}_mean_seconds": float(profile[key]) / worker_count
        for key in ROLLOUT_WORKER_TIMELINE
    }
    worker_profiled = sum(worker_values.values())
    worker_normalized = {
        f"rollout_{key[:-2]}_seconds_per_100k_transitions": (
            float(profile[key]) / transitions * 100_000.0
        )
        for key in ROLLOUT_WORKER_TIMELINE
    }
    worker_unprofiled_sum = max(
        0.0,
        float(profile["worker_wall_s"])
        - sum(float(profile[key]) for key in ROLLOUT_WORKER_TIMELINE),
    )
    result = {
        "rollout_seconds": wall_seconds,
        "rollout_games_per_second": games / wall_seconds,
        "rollout_transitions": float(transitions),
        "rollout_transitions_per_second": transitions / wall_seconds,
        "rollout_worker_wall_mean_seconds": worker_wall_mean,
        "rollout_worker_wall_max_seconds": float(profile["worker_wall_max_s"]),
        "rollout_worker_unprofiled_mean_seconds": max(
            0.0, worker_wall_mean - worker_profiled
        ),
        "rollout_worker_unprofiled_seconds_per_100k_transitions": (
            worker_unprofiled_sum / transitions * 100_000.0
        ),
        **worker_values,
        **worker_normalized,
    }
    for key in ROLLOUT_NESTED_DIAGNOSTICS:
        result[f"rollout_nested_{key[:-2]}_mean_seconds"] = (
            float(profile[key]) / worker_count
        )
    for key in ("main_dispatch_s", "main_wait_s", "main_ipc_s", "main_merge_s"):
        result[f"rollout_{key[:-2]}_seconds"] = float(profile[key])
    return rollout, result


def _ppo_measurement(
    runtime: dict[str, Any],
    rollout: dict[str, np.ndarray],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    _restore_populated_baseline(runtime)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    stats = ppo_update_history(
        runtime["training_policy"],
        runtime["optimizer"],
        rollout,
        runtime["cfg"],
        device,
    )
    stage_values = {
        f"ppo_{stage}_seconds": float(
            stats.get(f"ppo_stage_{stage}_seconds", 0.0)
        )
        for stage in PPO_STAGES
    }
    ppo_seconds = float(stats["ppo_seconds"])
    return {
        "ppo_seconds": ppo_seconds,
        "ppo_transitions_per_second": len(rollout["obs"]) / ppo_seconds,
        "ppo_optimizer_steps": float(stats["optimizer_steps"]),
        "ppo_peak_allocated_mb": float(stats["peak_allocated_mb"]),
        "ppo_peak_reserved_mb": float(stats["peak_reserved_mb"]),
        "ppo_unprofiled_seconds": max(
            0.0, ppo_seconds - sum(stage_values.values())
        ),
        **stage_values,
    }


def _path_value(row: dict[str, Any], path: str) -> float:
    value: Any = row
    for part in path.split("."):
        value = value[part]
    return float(value)


def _two_sided_sign_flip_pvalue(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float:
    positive = _one_sided_sign_flip_pvalue(
        differences, samples=samples, seed=seed
    )
    negative = _one_sided_sign_flip_pvalue(
        -differences, samples=samples, seed=seed + 1
    )
    return min(1.0, 2.0 * min(positive, negative))


def _effect(
    measurements: list[dict[str, Any]],
    path: str,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
    equivalence_margin: float,
) -> dict[str, Any]:
    standard = np.asarray(
        [_path_value(row["standard"], path) for row in measurements],
        dtype=np.float64,
    )
    compact = np.asarray(
        [_path_value(row["compact"], path) for row in measurements],
        dtype=np.float64,
    )
    valid = (
        np.isfinite(standard)
        & np.isfinite(compact)
        & (standard > 0.0)
        & (compact > 0.0)
    )
    standard = standard[valid]
    compact = compact[valid]
    if len(standard) < 2:
        return {"metric": path, "status": "insufficient_positive_pairs"}
    differences = np.log(standard / compact)
    speedup = math.exp(float(differences.mean()))
    confidence_interval = _bootstrap_geometric_speedup_ci(
        differences,
        samples=bootstrap_samples,
        seed=seed,
    )
    lower_equivalence = 1.0 / (1.0 + equivalence_margin)
    upper_equivalence = 1.0 + equivalence_margin
    return {
        "metric": path,
        "status": "ok",
        "paired_blocks": int(len(differences)),
        "standard_median_seconds": float(statistics.median(standard)),
        "compact_median_seconds": float(statistics.median(compact)),
        "geometric_speedup_standard_over_compact": speedup,
        "bootstrap_95_percent_speedup_ci": confidence_interval,
        "two_sided_sign_flip_p": _two_sided_sign_flip_pvalue(
            differences,
            samples=permutation_samples,
            seed=seed + 1,
        ),
        "compact_significantly_faster_unadjusted": (
            confidence_interval[0] > 1.0
        ),
        "compact_significantly_slower_unadjusted": (
            confidence_interval[1] < 1.0
        ),
        "equivalent_within_margin_by_ci": (
            confidence_interval[0] >= lower_equivalence
            and confidence_interval[1] <= upper_equivalence
        ),
    }


def _analyze_family(
    measurements: list[dict[str, Any]],
    metrics: list[str],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
    alpha: float,
    equivalence_margin: float,
) -> list[dict[str, Any]]:
    results = [
        _effect(
            measurements,
            metric,
            bootstrap_samples=bootstrap_samples,
            permutation_samples=permutation_samples,
            seed=seed + 10_000 * index,
            equivalence_margin=equivalence_margin,
        )
        for index, metric in enumerate(metrics)
    ]
    valid = [row for row in results if row["status"] == "ok"]
    adjusted = _holm_adjust(
        [float(row["two_sided_sign_flip_p"]) for row in valid]
    )
    for row, adjusted_p in zip(valid, adjusted):
        row["holm_adjusted_p"] = adjusted_p
        row["significant_after_holm"] = adjusted_p < alpha
        row["direction_after_holm"] = (
            "compact_faster"
            if adjusted_p < alpha
            and row["geometric_speedup_standard_over_compact"] > 1.0
            else "compact_slower"
            if adjusted_p < alpha
            else "not_distinguished"
        )
    return results


def _run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested device")
    configs = {
        name: _architecture_config(name, device) for name in ARCHITECTURES
    }
    for cfg in configs.values():
        cfg.n_workers = args.workers
        cfg.games_per_update = args.rollout_games
    runtimes = {
        name: _build_runtime(name, configs[name], args.seed + index, device)
        for index, name in enumerate(ARCHITECTURES)
    }

    torch.manual_seed(args.seed + 100)
    pass_policy = TichuPassingPolicy(
        hidden_dim=configs["standard"].pass_hidden_dim,
        n_layers=configs["standard"].pass_n_layers,
    ).to(device)
    gt_policy = TichuGrandTichuPolicy(
        hidden_dim=configs["standard"].gt_hidden_dim,
        n_layers=configs["standard"].gt_n_layers,
    ).to(device)
    init_gt_smartly(gt_policy)
    pass_policy.eval()
    gt_policy.eval()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "running",
        "design": {
            "analysis": "paired architecture runtime attribution",
            "warmup_blocks_excluded": args.warmup_blocks,
            "target_measured_blocks": args.blocks,
            "minimum_measured_blocks": args.minimum_blocks,
            "rollout_games_per_architecture_per_block": args.rollout_games,
            "independent_rollout_seeds_per_block": True,
            "condition_order": "alternating AB/BA",
            "shared_frozen_ppo_rollout_within_pair": True,
            "shared_rollout_origin": "alternates standard/compact by block",
            "memory_batch_size_held_fixed": 512,
            "optimizer_batch_size_held_fixed": 2048,
            "ppo_epochs_held_fixed": 2,
            "rollout_workers": args.workers,
            "persistent_training": False,
            "populated_adam_state_restored_before_every_measured_update": True,
            "bootstrap_samples": args.bootstrap_samples,
            "sign_flip_samples": args.permutation_samples,
            "family_wise_alpha": args.alpha,
            "equivalence_margin_fraction": args.equivalence_margin,
            "max_minutes": args.max_minutes,
            "seed": args.seed,
        },
        "parameters": {
            name: int(runtime["parameters"])
            for name, runtime in runtimes.items()
        },
        "measurements": [],
    }
    _atomic_write_json(output_path, result)

    context = mp.get_context("spawn")
    pool = context.Pool(args.workers)
    started = time.monotonic()
    measured: list[dict[str, Any]] = []
    total_blocks = args.warmup_blocks + args.blocks
    try:
        for block in range(total_blocks):
            is_warmup = block < args.warmup_blocks
            measured_index = block - args.warmup_blocks
            if (
                not is_warmup
                and measured_index >= args.minimum_blocks
                and time.monotonic() - started >= args.max_minutes * 60.0
            ):
                print(
                    f"Stopping at the predeclared {args.max_minutes:g}-minute "
                    f"cap after {measured_index} measured blocks.",
                    flush=True,
                )
                break

            block_seed = args.seed + 1000 * block
            order = list(ARCHITECTURES)
            if block % 2:
                order.reverse()
            rollouts: dict[str, dict[str, np.ndarray]] = {}
            block_rows: dict[str, dict[str, float]] = {}
            for name in order:
                rollout, rollout_row = _rollout_measurement(
                    runtimes[name],
                    games=args.rollout_games,
                    seed=block_seed,
                    device=device,
                    pool=pool,
                    pass_policy=pass_policy,
                    gt_policy=gt_policy,
                )
                rollouts[name] = rollout
                block_rows[name] = rollout_row

            trajectory_keys = (
                "obs",
                "legal_lens",
                "action_idxs",
                "history_lens",
                "returns",
            )
            mismatched = [
                key
                for key in trajectory_keys
                if not np.array_equal(
                    rollouts["standard"][key], rollouts["compact"][key]
                )
            ]
            if mismatched:
                raise RuntimeError(
                    "paired architectures produced different trajectories "
                    f"despite neutral behavior heads: {mismatched}"
                )

            shared_source = ARCHITECTURES[block % 2]
            shared_rollout = rollouts[shared_source]
            ppo_order = list(reversed(order))
            for name in ppo_order:
                ppo_row = _ppo_measurement(
                    runtimes[name],
                    shared_rollout,
                    seed=block_seed + 500,
                    device=device,
                )
                block_rows[name].update(ppo_row)
                block_rows[name]["modeled_cycle_seconds"] = (
                    block_rows[name]["rollout_seconds"]
                    + block_rows[name]["ppo_seconds"]
                )
                if block == 0:
                    _capture_populated_baseline(runtimes[name])

            record = {
                "block": block,
                "measured_block": None if is_warmup else measured_index,
                "seed": block_seed,
                "rollout_order": order,
                "ppo_order": ppo_order,
                "shared_rollout_source": shared_source,
                "shared_rollout_transitions": len(shared_rollout["obs"]),
                "standard": block_rows["standard"],
                "compact": block_rows["compact"],
            }
            if not is_warmup:
                measured.append(record)
                result["measurements"] = measured
                result["measured_blocks_completed"] = len(measured)
                _atomic_write_json(output_path, result)
            print(json.dumps(record, sort_keys=True), flush=True)
            del rollouts, shared_rollout
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        pool.close()
        pool.join()

    if len(measured) < args.minimum_blocks:
        result["status"] = "insufficient_blocks"
        result["measured_blocks_completed"] = len(measured)
        _atomic_write_json(output_path, result)
        raise RuntimeError(
            f"only {len(measured)} measured blocks completed; "
            f"minimum is {args.minimum_blocks}"
        )

    primary_metrics = [
        "modeled_cycle_seconds",
        "rollout_seconds",
        "ppo_seconds",
    ]
    diagnostic_metrics = [
        *(
            f"rollout_{key[:-2]}_seconds_per_100k_transitions"
            for key in ROLLOUT_WORKER_TIMELINE
        ),
        "rollout_worker_unprofiled_seconds_per_100k_transitions",
        *(f"ppo_{stage}_seconds" for stage in PPO_STAGES),
        "ppo_unprofiled_seconds",
    ]
    result["primary_comparisons"] = _analyze_family(
        measured,
        primary_metrics,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed + 2_000_000,
        alpha=args.alpha,
        equivalence_margin=args.equivalence_margin,
    )
    result["diagnostic_comparisons"] = _analyze_family(
        measured,
        diagnostic_metrics,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed + 4_000_000,
        alpha=args.alpha,
        equivalence_margin=args.equivalence_margin,
    )
    result["status"] = "complete"
    result["elapsed_minutes"] = (time.monotonic() - started) / 60.0
    result["measured_blocks_completed"] = len(measured)
    _atomic_write_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired standard-vs-compact runtime attribution"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout-games", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-blocks", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=48)
    parser.add_argument("--minimum-blocks", type=int, default=32)
    parser.add_argument("--max-minutes", type=float, default=55.0)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--permutation-samples", type=int, default=200_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--equivalence-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    positive = (
        args.rollout_games,
        args.workers,
        args.warmup_blocks,
        args.blocks,
        args.minimum_blocks,
        args.max_minutes,
        args.bootstrap_samples,
        args.permutation_samples,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("games, workers, blocks, time, and samples must be positive")
    if args.minimum_blocks > args.blocks:
        raise ValueError("minimum blocks cannot exceed target blocks")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if not 0.0 < args.equivalence_margin < 1.0:
        raise ValueError("equivalence margin must be between zero and one")
    result = _run_experiment(args)
    print("\nPrimary paired comparisons:", flush=True)
    for row in result["primary_comparisons"]:
        print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
