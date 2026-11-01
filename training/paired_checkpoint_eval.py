"""Statistically powered, deal-paired checkpoint strength evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from training.arena import load_arena_model, play_matchup
from training.checkpoints import LEGACY_HISTORY_TAG, checkpoint_info


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[int(index)])
        running = max(running, value)
        adjusted[int(index)] = running
    return adjusted


def _block_sign_flip_pvalue(
    block_means: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float:
    observed = abs(float(block_means.mean()))
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    chunk = 10_000
    while completed < samples:
        count = min(chunk, samples - completed)
        signs = rng.integers(
            0,
            2,
            size=(count, len(block_means)),
            dtype=np.int8,
        )
        signs = signs * 2 - 1
        permuted = np.abs((signs * block_means).mean(axis=1))
        exceedances += int(np.count_nonzero(permuted >= observed))
        completed += count
    return (exceedances + 1) / (samples + 1)


def _block_bootstrap_intervals(
    block_means: np.ndarray,
    *,
    samples: int,
    alpha: float,
    comparisons: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(block_means),
        size=(samples, len(block_means)),
    )
    means = block_means[indices].mean(axis=1)
    ordinary = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    family_tail = alpha / (2 * comparisons)
    simultaneous = np.quantile(means, [family_tail, 1 - family_tail])
    return ordinary.tolist(), simultaneous.tolist()


def _summarize_comparison(
    candidate_team_a: np.ndarray,
    opponent_team_a: np.ndarray,
    *,
    block_size: int,
    bootstrap_samples: int,
    sign_flip_samples: int,
    alpha: float,
    comparisons: int,
    seed: int,
) -> dict:
    # The second orientation's raw diff favors the opponent. Sign-correct it,
    # then average the two outcomes from the identical deal to cancel team and
    # hand-set bias at the deal level.
    candidate_team_b = -opponent_team_a
    paired_effects = (candidate_team_a + candidate_team_b) / 2.0
    if len(paired_effects) % block_size:
        raise ValueError("deal count must be divisible by block_size")
    block_means = paired_effects.reshape(-1, block_size).mean(axis=1)
    ci95, simultaneous_ci = _block_bootstrap_intervals(
        block_means,
        samples=bootstrap_samples,
        alpha=alpha,
        comparisons=comparisons,
        seed=seed + 1,
    )
    raw_p = _block_sign_flip_pvalue(
        block_means,
        samples=sign_flip_samples,
        seed=seed + 2,
    )
    all_candidate_scores = np.concatenate(
        (candidate_team_a, candidate_team_b)
    )
    mean = float(paired_effects.mean())
    return {
        "deal_pairs": int(len(paired_effects)),
        "rounds": int(2 * len(paired_effects)),
        "block_size_deals": block_size,
        "independent_blocks": int(len(block_means)),
        "candidate_mean_score_advantage": mean,
        "candidate_team_a_mean": float(candidate_team_a.mean()),
        "candidate_team_b_mean": float(candidate_team_b.mean()),
        "paired_effect_std": float(paired_effects.std(ddof=1)),
        "bootstrap_95_percent_ci": ci95,
        "bootstrap_simultaneous_ci": simultaneous_ci,
        "raw_two_sided_block_sign_flip_p": raw_p,
        "candidate_win_rate": float(np.mean(all_candidate_scores > 0)),
        "candidate_loss_rate": float(np.mean(all_candidate_scores < 0)),
        "draw_rate": float(np.mean(all_candidate_scores == 0)),
        "direction": (
            "candidate_stronger"
            if mean > 0
            else "candidate_weaker" if mean < 0 else "tied"
        ),
        "block_means": block_means.tolist(),
        "candidate_team_a_scores": candidate_team_a.tolist(),
        "opponent_team_a_scores": opponent_team_a.tolist(),
    }


def _checkpoint_manifest(spec: str) -> dict:
    label, separator, ref = spec.partition("=")
    if not separator:
        ref = label
        label = Path(ref).stem
    info = checkpoint_info(ref)
    if info.legacy_history and not label.endswith(f"-{LEGACY_HISTORY_TAG}"):
        label = f"{label}-{LEGACY_HISTORY_TAG}"  # same label load_arena_model prints
    return {
        "label": label,
        "legacy_history": info.legacy_history,
        "path": str(info.path),
        "games_played": info.games_played,
        "kind": info.kind,
        "has_pass_model": info.has_pass_model,
        "has_gt_model": info.has_gt_model,
        "config": info.config,
    }


DEFAULT_DEALS = 16_384  # +-~1.05 points/round SE at the measured ~135 per-deal sd

_WORKER_MODELS: dict[tuple, list] = {}


def _play_shard(task: tuple) -> np.ndarray:
    """One worker process: play a shard of deals in one seating."""
    specs, device_s, seating, seeds, batch, with_calling, crn, torch_seed, py_seed = task
    device = torch.device(device_s)
    key = (tuple(specs), device_s)
    if key not in _WORKER_MODELS:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):  # load logs once, in the parent
            _WORKER_MODELS[key] = [load_arena_model(spec, device) for spec in specs]
    torch.manual_seed(torch_seed)
    return play_matchup(
        _WORKER_MODELS[key], seating, n_games=len(seeds), batch=batch, device=device,
        rng=random.Random(py_seed), game_seeds=np.asarray(seeds, dtype=np.int64),
        with_calling=with_calling, common_random_numbers=crn,
    )


def _worker_init() -> None:
    torch.set_num_threads(1)


def _play_both_orientations(args, specs, models, opponent_index, deal_seeds, seed, pool):
    """Per-deal results with the candidate on seats 0/2 and on seats 1/3."""
    crn = bool(getattr(args, "common_random_numbers", False))
    seatings = ((0, opponent_index, 0, opponent_index), (opponent_index, 0, opponent_index, 0))
    if pool is None:
        out = []
        for k, seating in enumerate(seatings):
            torch.manual_seed(seed + 1 + 2 * k)
            out.append(play_matchup(
                models, seating, n_games=args.deals, batch=args.batch, device=torch.device(args.device),
                rng=random.Random(seed + 2 + 2 * k), game_seeds=deal_seeds,
                with_calling=not args.no_gt, common_random_numbers=crn,
            ))
        return out
    n_shards = max(1, min(len(deal_seeds), 4 * args.procs))
    bounds = np.linspace(0, len(deal_seeds), n_shards + 1).astype(int)
    tasks = [
        (specs, args.device, seating, deal_seeds[lo:hi].tolist(), args.batch, not args.no_gt, crn,
         seed + 1_000 * k + j, seed + 1_000 * k + j + 500)
        for k, seating in enumerate(seatings)
        for j, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]))
    ]
    shards = pool.map(_play_shard, tasks)
    return [np.concatenate(shards[:n_shards]), np.concatenate(shards[n_shards:])]


def run_experiment(args: argparse.Namespace) -> dict:
    if args.deals <= 0 or args.deals % args.block_size:
        raise ValueError("deals must be positive and divisible by block-size")
    if len(args.opponents) < 1:
        raise ValueError("at least one opponent is required")
    if args.bootstrap_samples <= 0 or args.sign_flip_samples <= 0:
        raise ValueError("statistical sample counts must be positive")

    device = torch.device(args.device)
    specs = [args.candidate, *args.opponents]
    manifests = [_checkpoint_manifest(spec) for spec in specs]
    models = [load_arena_model(spec, device) for spec in specs]
    comparisons = len(args.opponents)
    output = Path(args.output)
    result = {
        "design": {
            "primary_endpoint": "deal-paired mean score advantage per round",
            "deal_pairs_per_opponent": args.deals,
            "rounds_per_opponent": 2 * args.deals,
            "team_orientations": [
                "candidate seats 0,2",
                "candidate seats 1,3",
            ],
            "identical_deal_seed_across_orientations": True,
            "passing_heads_enabled": True,
            "grand_tichu_heads_enabled": not args.no_gt,
            "dragon_and_mahjong_heads_enabled": True,
            "block_size_deals": args.block_size,
            "independent_blocks_per_opponent": args.deals // args.block_size,
            "bootstrap_samples": args.bootstrap_samples,
            "sign_flip_samples": args.sign_flip_samples,
            "family_wise_alpha": args.alpha,
            "multiple_comparison_method": "Holm p-values and Bonferroni bootstrap intervals",
            "fixed_sample_no_early_stopping": True,
            "seed": args.seed,
            "device": str(device),
            "batch": args.batch,
            "procs": int(getattr(args, "procs", 1)),
            "common_random_numbers": bool(getattr(args, "common_random_numbers", False)),
        },
        "checkpoints": manifests,
        "comparisons": [],
        "complete": False,
    }
    _atomic_write_json(output, result)

    started = time.perf_counter()
    procs = int(getattr(args, "procs", 1))
    pool = None
    if procs > 1:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(procs, initializer=_worker_init)
    try:
        _run_comparisons(args, specs, models, comparisons, result, output, pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return _finish(args, result, output, started)


def _run_comparisons(args, specs, models, comparisons, result, output, pool) -> None:
    for opponent_index, opponent in enumerate(models[1:], start=1):
        comparison_seed = args.seed + 100_000 * opponent_index
        seed_rng = np.random.default_rng(comparison_seed)
        deal_seeds = seed_rng.integers(
            0,
            2**63,
            size=args.deals,
            dtype=np.int64,
        )
        print(
            f"\nComparing {models[0].label} against {opponent.label}: "
            f"{args.deals:,} paired deals, {2 * args.deals:,} rounds",
            flush=True,
        )

        orientation_started = time.perf_counter()
        candidate_team_a, opponent_team_a = _play_both_orientations(
            args, specs, models, opponent_index, deal_seeds, comparison_seed, pool)
        print(
            f"  both seatings complete in {time.perf_counter() - orientation_started:.0f}s",
            flush=True,
        )

        summary = _summarize_comparison(
            candidate_team_a,
            opponent_team_a,
            block_size=args.block_size,
            bootstrap_samples=args.bootstrap_samples,
            sign_flip_samples=args.sign_flip_samples,
            alpha=args.alpha,
            comparisons=comparisons,
            seed=comparison_seed + 5,
        )
        summary.update(
            {
                "candidate": models[0].label,
                "opponent": opponent.label,
                "deal_seeds": deal_seeds.tolist(),
            }
        )
        result["comparisons"].append(summary)
        _atomic_write_json(output, result)
        print(
            f"  paired advantage: "
            f"{summary['candidate_mean_score_advantage']:+.2f} points/round "
            f"(simultaneous CI {summary['bootstrap_simultaneous_ci']})",
            flush=True,
        )



def _finish(args, result, output, started) -> dict:
    adjusted = _holm_adjust(
        [
            comparison["raw_two_sided_block_sign_flip_p"]
            for comparison in result["comparisons"]
        ]
    )
    for comparison, adjusted_p in zip(result["comparisons"], adjusted):
        comparison["holm_adjusted_p"] = adjusted_p
        lo, hi = comparison["bootstrap_simultaneous_ci"]
        comparison["statistically_significant_fwer"] = bool(
            adjusted_p < args.alpha and not (lo <= 0 <= hi)
        )
    result["elapsed_seconds"] = time.perf_counter() - started
    result["complete"] = True
    _atomic_write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deal-paired checkpoint strength comparison"
    )
    parser.add_argument("candidate", help="NAME=checkpoint")
    parser.add_argument("opponents", nargs="+", help="NAME=checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deals", type=int, default=DEFAULT_DEALS)
    parser.add_argument("--procs", type=int, default=min(10, max(1, (os.cpu_count() or 2) - 2)),
                        help="worker processes sharding the deals (1 = in-process)")
    parser.add_argument("--crn", dest="common_random_numbers", action="store_true",
                        help="common random numbers: both seatings share per-deal random streams")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--sign-flip-samples", type=int, default=200_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--no-gt", action="store_true")
    args = parser.parse_args()
    result = run_experiment(args)

    print("\nFinal family-wise results:", flush=True)
    for comparison in result["comparisons"]:
        print(
            f"  {comparison['candidate']} vs {comparison['opponent']}: "
            f"{comparison['candidate_mean_score_advantage']:+.2f} "
            f"points/round, simultaneous CI "
            f"{comparison['bootstrap_simultaneous_ci']}, "
            f"Holm p={comparison['holm_adjusted_p']:.6g}, "
            f"significant={comparison['statistically_significant_fwer']}",
            flush=True,
        )
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
