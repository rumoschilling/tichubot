"""Promotion gate for the best checkpoint (``BEST_CHECKPOINT``).

``training/best_checkpoint.json`` names the strongest known checkpoint. A candidate
replaces it only when a deal-paired evaluation shows it at least
``MIN_Z`` = 2 standard errors better (about 95% confidence that it is truly
stronger). Every attempt, promoted or not, leaves its evaluation file behind.

Usage:
    python -m training.promote_best CANDIDATE.pt
    python -m training.promote_best CANDIDATE.pt --deals 4096
    python -m training.promote_best CANDIDATE.pt --from-eval local_evaluations/x.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from training.checkpoints import (
    BEST_CHECKPOINT_RECORD,
    REPO_ROOT,
    checkpoint_info,
    read_best_record,
    resolve_checkpoint,
)
from training.paired_checkpoint_eval import _atomic_write_json

MIN_Z = 2.0


def paired_z(block_means) -> dict[str, float]:
    """Mean advantage, its standard error, and z from independent block means."""
    blocks = np.asarray(block_means, dtype=np.float64)
    if blocks.size < 2:
        raise ValueError("need at least two independent blocks")
    mean = float(blocks.mean())
    se = float(blocks.std(ddof=1) / math.sqrt(blocks.size))
    z = mean / se if se > 0 else (math.inf if mean > 0 else -math.inf if mean < 0 else 0.0)
    return {"mean": mean, "se": se, "z": z}


def dominates(block_means, min_z: float = MIN_Z) -> bool:
    return paired_z(block_means)["z"] >= min_z


def _repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _same_checkpoint(a: str | Path, b: str | Path) -> bool:
    return resolve_checkpoint(a).resolve() == resolve_checkpoint(b).resolve()


def _comparison_against_best(result: dict, candidate: str, best: str) -> dict:
    paths = {c["label"]: c["path"] for c in result["checkpoints"]}
    for comparison in result["comparisons"]:
        if _same_checkpoint(paths[comparison["candidate"]], candidate) and _same_checkpoint(
            paths[comparison["opponent"]], best
        ):
            return comparison
    raise ValueError(f"evaluation has no comparison of {candidate} against best {best}")


def decide(
    candidate: str,
    comparison: dict,
    evaluation_file: str,
    *,
    record_path: Path = BEST_CHECKPOINT_RECORD,
    min_z: float = MIN_Z,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Apply the gate to one finished comparison; rewrite the record if passed."""
    current = read_best_record(record_path)
    stats = paired_z(comparison["block_means"])
    promoted = stats["z"] >= min_z
    outcome = {
        "candidate": _repo_relative(resolve_checkpoint(candidate)),
        "best_before": current["path"],
        "promoted": promoted,
        "min_z": min_z,
        **stats,
        "deal_pairs": comparison["deal_pairs"],
        "evaluation_file": _repo_relative(evaluation_file),
    }
    if promoted:
        info = checkpoint_info(candidate)
        record = {
            "path": outcome["candidate"],
            "games_played": info.games_played,
            "promoted_at": (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds"),
            "rule": f"paired mean advantage >= {min_z} standard errors over the previous best",
            "evidence": {
                "previous_best": current["path"],
                "mean_score_advantage": stats["mean"],
                "standard_error": stats["se"],
                "z": stats["z"],
                "deal_pairs": comparison["deal_pairs"],
                "evaluation_file": outcome["evaluation_file"],
            },
            "history": [
                *current.get("history", []),
                {k: current[k] for k in ("path", "promoted_at") if k in current},
            ],
        }
        _atomic_write_json(record_path, record)
    return outcome


def evaluate_and_decide(
    candidate: str,
    *,
    deals: int = 16_384,
    block_size: int = 128,
    batch: int = 256,
    device: str = "cpu",
    seed: int = 20261015,
    output: str | None = None,
    record_path: Path = BEST_CHECKPOINT_RECORD,
    min_z: float = MIN_Z,
    procs: int = 1,
    common_random_numbers: bool = False,
) -> dict[str, Any]:
    """Run a fresh paired evaluation of ``candidate`` against the current best."""
    from training.paired_checkpoint_eval import run_experiment

    best = read_best_record(record_path)["path"]
    if _same_checkpoint(candidate, best):
        raise ValueError("candidate is already the best checkpoint")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output or str(
        REPO_ROOT / "local_evaluations" / f"best_gate_{Path(candidate).stem}_{stamp}.json"
    )
    args = argparse.Namespace(
        candidate=f"candidate={candidate}",
        opponents=[f"best={best}"],
        output=output,
        device=device,
        deals=deals,
        batch=batch,
        block_size=block_size,
        bootstrap_samples=10_000,
        sign_flip_samples=20_000,
        alpha=0.05,
        seed=seed,
        no_gt=False,
        procs=procs,
        common_random_numbers=common_random_numbers,
    )
    result = run_experiment(args)
    comparison = _comparison_against_best(result, candidate, best)
    return decide(candidate, comparison, output, record_path=record_path, min_z=min_z)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a checkpoint to BEST_CHECKPOINT only if it is >= 2 SE better")
    parser.add_argument("candidate")
    parser.add_argument("--from-eval", default=None,
                        help="reuse an existing paired evaluation of candidate vs the current best")
    parser.add_argument("--deals", type=int, default=16_384)
    parser.add_argument("--procs", type=int, default=8, help="evaluation worker processes")
    parser.add_argument("--crn", dest="common_random_numbers", action="store_true",
                        help="common random numbers in the paired evaluation")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.from_eval:
        result = json.loads(Path(args.from_eval).read_text(encoding="utf-8"))
        best = read_best_record()["path"]
        comparison = _comparison_against_best(result, args.candidate, best)
        outcome = decide(args.candidate, comparison, args.from_eval)
    else:
        outcome = evaluate_and_decide(
            args.candidate, deals=args.deals, block_size=args.block_size,
            batch=args.batch, device=args.device, seed=args.seed, output=args.output,
            procs=args.procs, common_random_numbers=args.common_random_numbers,
        )
    verdict = "PROMOTED" if outcome["promoted"] else "kept current best"
    print(
        f"{verdict}: {outcome['candidate']} vs {outcome['best_before']}: "
        f"{outcome['mean']:+.2f} ± {outcome['se']:.2f} points/round "
        f"(z = {outcome['z']:.2f}, need {outcome['min_z']})"
    )


if __name__ == "__main__":
    main()
