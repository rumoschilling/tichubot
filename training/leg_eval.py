"""Post-fetch evaluation drill for a Kaggle training leg.

Runs the standard checks on a freshly fetched run:
  1. eval: new checkpoint vs the previous leg's checkpoint (score diff primary)
  2. eval: new checkpoint vs the embed-700k fixed baseline
  3. GT head health check (declare rate + selectivity, see training.gt_check)

Run/prev discovery: kaggle_runs/*/ dirs that contain
tichubot_outputs/checkpoints/*/latest.pt, sorted by name (dates sort
lexicographically) — newest is the run under eval, second-newest is "prev".

Usage:
    kaggle/run.sh --fetch standard          # first, fetch the finished kernel
    python -m training.leg_eval             # then run the drill
    python -m training.leg_eval --games 1000
    python -m training.leg_eval --run kaggle_runs/2026-07-11_103232_standard
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASELINE = "checkpoints/embed/ckpt_00700032.pt"


def find_run_checkpoints(runs_dir: str = "kaggle_runs") -> list[str]:
    """All run latest.pt paths, sorted oldest -> newest by run dir name."""
    pattern = os.path.join(runs_dir, "*", "tichubot_outputs", "checkpoints",
                           "*", "latest.pt")
    return sorted(glob.glob(pattern))


def run_checkpoint(run_dir: str) -> str:
    """The latest.pt inside one run dir (errors if missing or ambiguous)."""
    hits = glob.glob(os.path.join(run_dir, "tichubot_outputs", "checkpoints",
                                  "*", "latest.pt"))
    if len(hits) != 1:
        raise SystemExit(f"Expected exactly one latest.pt under {run_dir}, "
                         f"found {len(hits)}")
    return hits[0]


def run_eval(ckpt_a: str, ckpt_b: str, games: int, batch: int,
             device: str, extra: list[str] | None = None) -> dict | None:
    """Run training.eval A-vs-B as a subprocess, tee output, parse results."""
    cmd = [sys.executable, "-m", "training.eval", ckpt_a, ckpt_b,
           "--games", str(games), "--batch", str(batch), "--device", device]
    cmd += extra or []
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return None
    result = {}
    m = re.search(r"Mean score diff \(A - B\): ([+-][\d.]+)", proc.stdout)
    if m:
        result["score_diff"] = float(m.group(1))
    m = re.search(r"A win rate: ([\d.]+)%", proc.stdout)
    if m:
        result["win_rate_a"] = float(m.group(1))
    m = re.search(r"GT A: on\s+declared \d+ \(([\d.]+)% of rounds\), "
                  r"made ([\d.]+%|n/a), net ([+-][\d.]+) pts/round", proc.stdout)
    if m:
        result["gt_declared_pct"] = float(m.group(1))
        result["gt_made"] = m.group(2)
        result["gt_net"] = float(m.group(3))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standard post-fetch eval drill for a Kaggle leg")
    parser.add_argument("--run", type=str, default=None,
                        help="Run dir to evaluate (default: newest in kaggle_runs/)")
    parser.add_argument("--prev", type=str, default=None,
                        help="Checkpoint to compare against "
                             "(default: previous run's latest.pt)")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE,
                        help="Fixed baseline checkpoint")
    parser.add_argument("--games", type=int, default=8000,
                        help="Games per matchup (8000 ~ ±3.5/round stderr, "
                             "~4 min each on CPU)")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-gt", action="store_true",
                        help="Skip the GT head health check")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    if args.run is not None:
        new_ckpt = run_checkpoint(args.run)
        prev_ckpt = args.prev
        if prev_ckpt is None:
            ckpts = find_run_checkpoints()
            older = [c for c in ckpts if c != new_ckpt and c < new_ckpt]
            prev_ckpt = older[-1] if older else None
    else:
        ckpts = find_run_checkpoints()
        if not ckpts:
            raise SystemExit("No runs with latest.pt found under kaggle_runs/")
        new_ckpt = ckpts[-1]
        prev_ckpt = args.prev or (ckpts[-2] if len(ckpts) > 1 else None)

    print(f"New checkpoint:  {new_ckpt}")
    print(f"Prev checkpoint: {prev_ckpt}")
    print(f"Baseline:        {args.baseline}\n")

    summary: list[str] = []

    if prev_ckpt is not None:
        print("=" * 60)
        print("[1/3] New vs previous leg")
        print("=" * 60)
        r = run_eval(new_ckpt, prev_ckpt, args.games, args.batch, args.device)
        if r and "score_diff" in r:
            summary.append(f"vs prev:     {r['score_diff']:+.1f}/round  "
                           f"(win {r.get('win_rate_a', 0):.1f}%)")
    else:
        print("No previous checkpoint found — skipping [1/3]")

    print("=" * 60)
    print("[2/3] New vs fixed baseline")
    print("=" * 60)
    r = run_eval(new_ckpt, args.baseline, args.games, args.batch, args.device)
    if r and "score_diff" in r:
        summary.append(f"vs baseline: {r['score_diff']:+.1f}/round  "
                       f"(win {r.get('win_rate_a', 0):.1f}%)")

    if not args.skip_gt:
        print("=" * 60)
        print("[3/4] GT head health")
        print("=" * 60)
        from training.gt_check import gt_report
        g = gt_report(new_ckpt)
        if g is not None:
            summary.append(f"GT head:     {g['mean_prob']:.1%} mean call, "
                           f"corr {g['corr_strength']:+.2f} "
                           f"(collapsed if 0.0% / flat if corr~0)")

        if g is not None and g["mean_prob"] > 0.001:
            print("=" * 60)
            print("[4/4] GT value (new-with-GT vs new-without, same weights)")
            print("=" * 60)
            r = run_eval(new_ckpt, new_ckpt, args.games, args.batch,
                         args.device, extra=["--gt-a"])
            if r and "score_diff" in r:
                summary.append(
                    f"GT value:    {r['score_diff']:+.1f}/round  "
                    f"(declared {r.get('gt_declared_pct', 0):.1f}% of rounds, "
                    f"made {r.get('gt_made', '?')}, "
                    f"gt pts alone {r.get('gt_net', 0):+.1f}/round)")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for line in summary:
        print(f"  {line}")


if __name__ == "__main__":
    main()
