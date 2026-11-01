"""Adjacent-checkpoint learning curve with shared deals and family-wise inference."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from training.arena import load_arena_model, play_matchup
from training.paired_checkpoint_eval import (
    _atomic_write_json,
    _checkpoint_manifest,
    _holm_adjust,
    _summarize_comparison,
)


def _validate_checkpoint_grid(
    manifests: list[dict],
    *,
    tolerance_games: int = 0,
) -> tuple[int, list[int]]:
    games = [manifest["games_played"] for manifest in manifests]
    if any(not isinstance(value, int) for value in games):
        raise ValueError("every checkpoint must record integer games_played")
    if len(games) < 2:
        raise ValueError("at least two ordered checkpoints are required")
    deltas = [later - earlier for earlier, later in zip(games, games[1:])]
    if any(delta <= 0 for delta in deltas):
        raise ValueError("checkpoints must be ordered by increasing games_played")
    nominal = int(np.median(deltas))
    if any(abs(delta - nominal) > tolerance_games for delta in deltas):
        raise ValueError(
            "checkpoint grid spacing exceeds tolerance; "
            f"nominal={nominal}, tolerance={tolerance_games}, "
            f"observed deltas={deltas}"
        )
    return nominal, deltas


def run_curve_experiment(args: argparse.Namespace) -> dict:
    if args.deals <= 0 or args.deals % args.block_size:
        raise ValueError("deals must be positive and divisible by block-size")
    if args.bootstrap_samples <= 0 or args.sign_flip_samples <= 0:
        raise ValueError("statistical sample counts must be positive")

    manifests = [_checkpoint_manifest(spec) for spec in args.checkpoints]
    spacing, interval_spacings = _validate_checkpoint_grid(
        manifests,
        tolerance_games=args.spacing_tolerance,
    )
    device = torch.device(args.device)
    models = [load_arena_model(spec, device) for spec in args.checkpoints]
    interval_count = len(models) - 1
    output = Path(args.output)

    seed_rng = np.random.default_rng(args.seed)
    deal_seeds = seed_rng.integers(
        0,
        2**63,
        size=args.deals,
        dtype=np.int64,
    )
    result = {
        "design": {
            "primary_endpoint": (
                "shared-deal paired score advantage of each checkpoint over "
                "the immediately preceding checkpoint"
            ),
            "checkpoint_count": len(models),
            "interval_count": interval_count,
            "nominal_games_per_interval": spacing,
            "observed_games_per_interval": interval_spacings,
            "spacing_tolerance_games": args.spacing_tolerance,
            "deal_pairs_per_interval": args.deals,
            "rounds_per_interval": 2 * args.deals,
            "same_deal_seeds_for_every_interval": True,
            "team_orientations": [
                "later checkpoint seats 0,2",
                "later checkpoint seats 1,3",
            ],
            "passing_heads_enabled": True,
            "grand_tichu_heads_enabled": not args.no_gt,
            "dragon_and_mahjong_heads_enabled": True,
            "block_size_deals": args.block_size,
            "independent_blocks_per_interval": args.deals // args.block_size,
            "bootstrap_samples": args.bootstrap_samples,
            "sign_flip_samples": args.sign_flip_samples,
            "family_wise_alpha": args.alpha,
            "multiple_comparison_method": (
                "Holm p-values and Bonferroni bootstrap intervals"
            ),
            "fixed_sample_no_early_stopping": True,
            "seed": args.seed,
            "device": str(device),
            "batch": args.batch,
        },
        "checkpoints": manifests,
        "shared_deal_seeds": deal_seeds.tolist(),
        "intervals": [],
        "complete": False,
    }
    _atomic_write_json(output, result)

    started = time.perf_counter()
    for later_index in range(1, len(models)):
        prior_index = later_index - 1
        later = manifests[later_index]
        prior = manifests[prior_index]
        sampling_seed = args.seed + 10_000 * later_index
        print(
            f"\nInterval {later_index}/{interval_count}: "
            f"{prior['games_played']:,} -> {later['games_played']:,} games, "
            f"{args.deals:,} paired deals",
            flush=True,
        )

        torch.manual_seed(sampling_seed + 1)
        orientation_started = time.perf_counter()
        later_team_a = play_matchup(
            models,
            (later_index, prior_index, later_index, prior_index),
            n_games=args.deals,
            batch=args.batch,
            device=device,
            rng=random.Random(sampling_seed + 2),
            game_seeds=deal_seeds,
            with_calling=not args.no_gt,
        )
        print(
            f"  later checkpoint on seats 0,2 complete in "
            f"{time.perf_counter() - orientation_started:.0f}s",
            flush=True,
        )

        torch.manual_seed(sampling_seed + 3)
        orientation_started = time.perf_counter()
        prior_team_a = play_matchup(
            models,
            (prior_index, later_index, prior_index, later_index),
            n_games=args.deals,
            batch=args.batch,
            device=device,
            rng=random.Random(sampling_seed + 4),
            game_seeds=deal_seeds,
            with_calling=not args.no_gt,
        )
        print(
            f"  later checkpoint on seats 1,3 complete in "
            f"{time.perf_counter() - orientation_started:.0f}s",
            flush=True,
        )

        summary = _summarize_comparison(
            later_team_a,
            prior_team_a,
            block_size=args.block_size,
            bootstrap_samples=args.bootstrap_samples,
            sign_flip_samples=args.sign_flip_samples,
            alpha=args.alpha,
            comparisons=interval_count,
            seed=sampling_seed + 5,
        )
        summary.update(
            {
                "from_label": prior["label"],
                "to_label": later["label"],
                "from_games": prior["games_played"],
                "to_games": later["games_played"],
                "games_delta": later["games_played"] - prior["games_played"],
            }
        )
        result["intervals"].append(summary)
        _atomic_write_json(output, result)
        print(
            f"  later-checkpoint advantage: "
            f"{summary['candidate_mean_score_advantage']:+.2f} points/round "
            f"(simultaneous CI {summary['bootstrap_simultaneous_ci']})",
            flush=True,
        )

    adjusted = _holm_adjust(
        [
            interval["raw_two_sided_block_sign_flip_p"]
            for interval in result["intervals"]
        ]
    )
    cumulative = 0.0
    result["descriptive_curve"] = [
        {
            "games_played": manifests[0]["games_played"],
            "cumulative_adjacent_advantage": cumulative,
        }
    ]
    for interval, adjusted_p in zip(result["intervals"], adjusted):
        interval["holm_adjusted_p"] = adjusted_p
        lo, hi = interval["bootstrap_simultaneous_ci"]
        interval["statistically_significant_fwer"] = bool(
            adjusted_p < args.alpha and not (lo <= 0 <= hi)
        )
        cumulative += interval["candidate_mean_score_advantage"]
        result["descriptive_curve"].append(
            {
                "games_played": interval["to_games"],
                "cumulative_adjacent_advantage": cumulative,
            }
        )

    result["descriptive_curve_warning"] = (
        "Cumulative adjacent advantages are a visualization aid, not an "
        "additive Elo scale; head-to-head checkpoint strength can be "
        "non-transitive."
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    result["complete"] = True
    _atomic_write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared-deal adjacent-checkpoint learning curve"
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="Ordered NAME=checkpoint entries on an equally spaced grid",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deals", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--spacing-tolerance", type=int, default=1024)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--sign-flip-samples", type=int, default=200_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--no-gt", action="store_true")
    args = parser.parse_args()

    result = run_curve_experiment(args)
    print("\nFinal family-wise interval results:", flush=True)
    for interval in result["intervals"]:
        print(
            f"  {interval['from_games']:,} -> {interval['to_games']:,}: "
            f"{interval['candidate_mean_score_advantage']:+.2f} "
            f"points/round, simultaneous CI "
            f"{interval['bootstrap_simultaneous_ci']}, "
            f"Holm p={interval['holm_adjusted_p']:.6g}, "
            f"significant={interval['statistically_significant_fwer']}",
            flush=True,
        )
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
