"""Training-curve comparison: ring agents vs the self-play lineage by own games.

Bins every PPO update by the model's own (self-play-equivalent) game count
and reports per-bin medians of value loss, policy entropy, approximate KL and
clip fraction, for the ring agents (pooled and per agent) and the compact
lineage (the 2026-08-13+ compact-scratch runs, which the ring starts from).

Usage: python -m training.ring_training_curves [--bin 500000]
"""

from __future__ import annotations

import argparse
import glob

import pandas as pd

LINEAGE_RUNS = [
    "2026-08-13_132440", "2026-08-14_084154", "2026-08-15_073817",
    "2026-08-20_154119", "2026-08-21_221118",
]
METRICS = ["value_loss", "entropy", "approx_kl", "clip_frac"]


def lineage() -> pd.DataFrame:
    frames = []
    for run in LINEAGE_RUNS:
        for f in glob.glob(f"kaggle_runs/{run}_compact-scratch/tichubot_outputs/logs/train_history_compact_scratch.csv"):
            frames.append(pd.read_csv(f))
    df = pd.concat(frames).drop_duplicates("games").sort_values("games")
    return df.rename(columns={"games": "own_games"})


def ring(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"agent_games": "own_games"})


def binned(df: pd.DataFrame, width: int, by: str | None = None) -> pd.DataFrame:
    df = df.assign(bin=(df["own_games"] // width) * width)
    keys = ["bin"] + ([by] if by else [])
    out = df.groupby(keys)[METRICS].median()
    out["n"] = df.groupby(keys).size()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ring-log", default="ring_runs/ring1/ring_log.csv")
    ap.add_argument("--bin", type=int, default=500_000)
    args = ap.parse_args()
    pd.set_option("display.width", 160)
    lin, rg = lineage(), ring(args.ring_log)
    lb, rb = binned(lin, args.bin), binned(rg, args.bin)
    table = lb.join(rb, lsuffix="_lin", rsuffix="_ring", how="outer")
    cols = [f"{m}_{s}" for m in METRICS for s in ("lin", "ring")] + ["n_lin", "n_ring"]
    table.index = [f"{b / 1e6:.2f}M" for b in table.index]
    print(table[cols].round(4).to_string())
    print("\nRing agents, last 1M own games (median per agent):")
    last = rg[rg["own_games"] > rg.groupby("agent")["own_games"].transform("max") - 1_000_000]
    print(last.groupby("agent")[METRICS + ["own_games"]].median().round(4).to_string())


if __name__ == "__main__":
    main()
