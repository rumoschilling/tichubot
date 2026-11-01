"""Grade the GT head's teacher against reality.

The Grand Tichu head never sees game outcomes: its PPO advantage is the
value net's counterfactual V(GT=1) - V(GT=0) (rollout_gt.py). This probe
replays that exact computation on real eval round-start states and compares
the teacher's promise with what the games actually pay:

  * team A declares through its head        -> real GT=1 rounds
  * team B's head runs hypothetically only  -> hand-matched GT=0 controls
  * post-passing, every hand the head wanted to declare gets the teacher's
    V(GT=1) and V(GT=0) (own GT bit toggled, empty history — the same
    feature path the training signal uses)
  * at round end the realized normalized score diff is recorded

All values are in return units (score diff / 400); x400 converts to points.
On each side, ONE branch is gradeable against reality (V(GT=1) on declared
rounds, V(GT=0) on control rounds); the other branch is the counterfactual
the head is trained on — no game outcome ever grades it.

Usage:
    python -m training.gt_calibration CKPT [--games 8000] [--batch 64]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from tichu_env.encoding import (
    encode_pass_source_ids,
    encode_pass_target_ids,
    encode_static,
)
from training.checkpoints import load_policy_checkpoint
from training.eval import evaluate
from training.rollout_gt import _get_batch_values

RETURN_SCALE = 400.0  # rollout returns are (score_a - score_b) / 400


class GTTeacherProbe:
    """Collects (teacher prediction, realized return) pairs via eval hooks."""

    def __init__(self, value_policy, device: torch.device):
        self.value_policy = value_policy
        self.device = device
        self.pending: list[tuple[int, str, int, float, float]] = []
        # per team: list of {"v1", "v0", "realized"} in return units
        self.records: dict[str, list[dict[str, float]]] = {"a": [], "b": []}

    @torch.no_grad()
    def post_pass(self, games, gt_wanted) -> None:
        entries: list[tuple[int, str]] = []
        obs1, obs0, ps, pt = [], [], [], []
        for gi, gs in enumerate(games):
            for team in ("a", "b"):
                for p in gt_wanted[gi][team + "_players"]:
                    entries.append((gi, team, p))
                    ps.append(encode_pass_source_ids(gs, p))
                    pt.append(encode_pass_target_ids(gs, p))
                    original = gs.tichu_declarations[p]
                    gs.tichu_declarations[p] = 2
                    obs1.append(encode_static(gs, p))
                    gs.tichu_declarations[p] = 0
                    obs0.append(encode_static(gs, p))
                    gs.tichu_declarations[p] = original
        if not entries:
            self.pending = []
            return
        ps_np = np.stack(ps).astype(np.int64)
        pt_np = np.stack(pt).astype(np.int64)
        v1 = _get_batch_values(
            self.value_policy, np.stack(obs1), ps_np, pt_np, self.device)
        v0 = _get_batch_values(
            self.value_policy, np.stack(obs0), ps_np, pt_np, self.device)
        self.pending = [
            (gi, team, p, float(v1[j]), float(v0[j]))
            for j, (gi, team, p) in enumerate(entries)
        ]

    def post_round(self, games, gt_wanted) -> None:
        for gi, team, _p, v1, v0 in self.pending:
            sa, sb = games[gi].round_scores
            diff = (sa - sb) / RETURN_SCALE
            self.records[team].append({
                "v1": v1,
                "v0": v0,
                "realized": diff if team == "a" else -diff,
            })
        self.pending = []


def report(records: dict[str, list[dict[str, float]]]) -> None:
    stats = {}
    for team in ("a", "b"):
        recs = records[team]
        if not recs:
            print(f"  team {team.upper()}: no wanted hands recorded")
            return
        v1 = np.array([r["v1"] for r in recs]) * RETURN_SCALE
        v0 = np.array([r["v0"] for r in recs]) * RETURN_SCALE
        real = np.array([r["realized"] for r in recs]) * RETURN_SCALE
        stats[team] = (v1, v0, real)

    print("\nGT teacher calibration (points/round; return units x400)")
    for team, title, taken in (
        ("a", "Declared rounds — head wanted, GT declared", "v1"),
        ("b", "Control rounds — head wanted, played quietly", "v0"),
    ):
        v1, v0, real = stats[team]
        n = len(real)
        promise = v1 - v0
        sem = real.std() / np.sqrt(n)
        print(f"\n  {title} (n={n} decisions):")
        taken_pred = v1.mean() if taken == "v1" else v0.mean()
        other_pred = v0.mean() if taken == "v1" else v1.mean()
        other_lbl = "V(GT=0)" if taken == "v1" else "V(GT=1)"
        taken_lbl = "V(GT=1)" if taken == "v1" else "V(GT=0)"
        print(f"    teacher {taken_lbl}: {taken_pred:+7.1f}   <- branch reality took")
        print(f"    teacher {other_lbl}: {other_pred:+7.1f}   <- imagined branch, never graded")
        print(f"    promised gain from declaring: {promise.mean():+.1f}  "
              f"(std {promise.std():.1f}, positive for {np.mean(promise > 0):.0%})")
        print(f"    realized outcome: {real.mean():+.1f} ± {sem:.1f}")
        print(f"    level error on taken branch (pred - real): "
              f"{taken_pred - real.mean():+.1f}")

    v1a, v0a, real_a = stats["a"]
    v1b, v0b, real_b = stats["b"]
    promised_all = np.concatenate([v1a - v0a, v1b - v0b]).mean()
    causal = real_a.mean() - real_b.mean()
    causal_sem = float(np.sqrt(real_a.var() / len(real_a)
                               + real_b.var() / len(real_b)))
    print("\n  Bottom line on the decision the teacher teaches:")
    print(f"    teacher's promised value of declaring: {promised_all:+.1f} pts/round")
    print(f"    measured value (declared - control):   {causal:+.1f} ± {causal_sem:.1f} pts/round")
    print(f"    teacher bias on the declare decision:  "
          f"{promised_all - causal:+.1f} pts/round")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the GT head's value-net teacher with realized returns")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--games", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    loaded = load_policy_checkpoint(args.checkpoint, device)
    probe = GTTeacherProbe(loaded.policy, device)

    print(f"\nProbing GT teacher calibration over {args.games} games "
          f"(A declares, B is the hand-matched control):", flush=True)
    evaluate(
        args.checkpoint, args.checkpoint,
        n_games=args.games,
        batch=args.batch,
        device=device,
        gt_a=True,
        gt_b=False,
        post_pass_hook=probe.post_pass,
        post_round_hook=probe.post_round,
    )
    report(probe.records)


if __name__ == "__main__":
    main()
