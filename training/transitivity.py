"""Is a lineage's progress general, or is it metagaming against itself?

Scores a ladder of one lineage's checkpoints against a panel of opponents
from outside that lineage (deal-paired, both seatings), plus optional direct
head-to-head pairs inside the ladder. Under transitive (additive) strength,
for any two ladder checkpoints A and B

    score(A vs X) - score(B vs X)  ~=  score(A vs B)     for every opponent X

A gain that shows up head-to-head but shrinks against outside opponents is
the signature of metagaming (learning to beat one's own past selves); a gain
that differs strongly between outside opponents is non-transitivity.

Results are cached per comparison file in --out, so re-runs only add new
cells. Usage:

  python -m training.transitivity --out local_evaluations/transitivity_mmd \\
      --ladder r29=A.pt r64=B.pt ... --panel hn13.5=X.pt lin5.8=Y.pt ... \\
      --pairs r116:r99 r99:r80 [--deals 16384 --procs 9]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _spec(arg: str) -> tuple[str, str]:
    label, sep, path = arg.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {arg}")
    return label, path


def _result(path: Path) -> dict[str, tuple[float, float]]:
    """opponent label (as given) -> (mean, per-deal SE) from a finished eval file."""
    data = json.loads(path.read_text())
    if not data.get("complete"):
        return {}
    out = {}
    for c in data["comparisons"]:
        se = c["paired_effect_std"] / math.sqrt(c["deal_pairs"])
        out[c["opponent"].removesuffix("-noLN")] = (c["candidate_mean_score_advantage"], se)
    return out


def evaluate(candidate: tuple[str, str], opponents: list[tuple[str, str]], out: Path, deals: int,
             procs: int) -> dict[str, tuple[float, float]]:
    """Cached run of candidate vs opponents; returns label -> (mean, se)."""
    from training.paired_checkpoint_eval import run_experiment

    target = out / f"{candidate[0]}__vs__{'+'.join(label for label, _ in opponents)}.json"
    if target.exists() and _result(target):
        return _result(target)
    args = argparse.Namespace(
        candidate=f"{candidate[0]}={candidate[1]}", opponents=[f"{l}={p}" for l, p in opponents],
        output=str(target), device="cpu", deals=deals, batch=256,
        block_size=256 if deals % 256 == 0 else deals,
        bootstrap_samples=10_000, sign_flip_samples=20_000, alpha=0.05, seed=20261101, no_gt=False,
        procs=procs, common_random_numbers=False,
    )
    run_experiment(args)
    return _result(target)


def report(ladder, panel, cells, pairs) -> str:
    lines = ["LADDER vs OUTSIDE PANEL (points/round, +- per-deal SE)"]
    lines.append(f"{'':12s}" + "".join(f"{label:>18s}" for label, _ in panel))
    for label, _ in ladder:
        row = cells.get(label, {})
        lines.append(f"{label:12s}" + "".join(
            f"{row[p][0]:+10.2f} +-{row[p][1]:4.2f}" if p in row else f"{'-':>18s}" for p, _ in panel))
    if pairs:
        lines.append("")
        lines.append("TRANSITIVITY: direct head-to-head vs the same gap measured through each outside opponent")
        for a, b, (m, se) in pairs:
            parts = []
            for p, _ in panel:
                if p in cells.get(a, {}) and p in cells.get(b, {}):
                    (ma, sa), (mb, sb) = cells[a][p], cells[b][p]
                    parts.append(f"{p} {ma - mb:+.2f}+-{math.hypot(sa, sb):.2f}")
            lines.append(f"  {a} vs {b}: direct {m:+.2f}+-{se:.2f} | via " + ", ".join(parts))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ladder", nargs="+", type=_spec, required=True)
    ap.add_argument("--panel", nargs="+", type=_spec, required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B ladder labels for direct head-to-head")
    ap.add_argument("--deals", type=int, default=16_384)
    ap.add_argument("--procs", type=int, default=9)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ladder_paths = dict(args.ladder)
    cells = {}
    for candidate in args.ladder:
        cells[candidate[0]] = evaluate(candidate, args.panel, args.out, args.deals, args.procs)
        print(report(args.ladder, args.panel, cells, []), flush=True)
    pairs = []
    for spec in args.pairs:
        a, b = spec.split(":")
        res = evaluate((a, ladder_paths[a]), [(b, ladder_paths[b])], args.out, args.deals, args.procs)
        pairs.append((a, b, res[b]))
    text = report(args.ladder, args.panel, cells, pairs)
    (args.out / "report.txt").write_text(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
