"""CPU benchmark and profile of ring-style multi-agent rollouts.

Loads four checkpoints as ring agents, samples ring seat owners (trainees
(0, 1) vs frozen (2, 3), optional history share) and times
``_collect_batched_multi`` in-process. Every repeat uses the same seeds, so
repeats differ only in timing noise.

Usage:
    OMP_NUM_THREADS=1 python -m training.bench_rollout --games 512 --repeats 5
    OMP_NUM_THREADS=1 python -m training.bench_rollout --games 512 --profile prof.out
    OMP_NUM_THREADS=1 python -m training.bench_rollout --games 64 --digest   # buffer hash
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import random
import statistics
import time

import numpy as np
import torch

STAT_KEYS = (
    "legal_actions_s", "forward_s", "immediate_decisions_s", "env_step_s",
    "obs_encode_s", "history_encode_s", "action_vec_s", "specialist_heads_s",
    "initialization_s",
)


def load_agents(paths: list[str]):
    from training.rollout_history import RolloutAgent
    from training.train_ring import build_arg_parser, load_agent

    args = build_arg_parser().parse_args(["--out", "/nonexistent"])
    device = torch.device("cpu")
    agents = [load_agent(p, f"agent{k}", args, device) for k, p in enumerate(paths)]
    return [RolloutAgent(a.bundle.policy, a.bundle.gt_policy, a.bundle.pass_policy) for a in agents]


def ring_owners(n_games: int, n_agents: int, seed: int) -> list[tuple[int, int, int, int]]:
    from training.ring import sample_seat_owners_with_history

    history = list(range(4, n_agents))
    return sample_seat_owners_with_history(
        (0, 1), (2, 3), history, n_games, 0.2 if history else 0.0, random.Random(seed))


def buffer_digest(buffers: dict) -> str:
    h = hashlib.sha256()
    for learner in sorted(buffers):
        for key in sorted(buffers[learner]):
            arr = np.ascontiguousarray(buffers[learner][key])
            h.update(f"{learner}:{key}:{arr.dtype}:{arr.shape}".encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def run_once(agents, owners, seed: int, torch_seed: int):
    from training.rollout_history import _collect_batched_multi, reset_rollout_stats

    reset_rollout_stats()
    torch.manual_seed(torch_seed)
    t0 = time.perf_counter()
    buffers, stats = _collect_batched_multi(
        agents, len(owners), torch.device("cpu"), seat_owners=owners,
        learners=(0, 1), with_calling=True, seed=seed,
    )
    return time.perf_counter() - t0, buffers, dict(stats)


def main() -> None:
    from training.train_ring import DEFAULT_STARTS

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--threads", type=int, default=1, help="torch intra-op threads")
    p.add_argument("--agents", type=int, default=4, help="4 ring agents, or 6 with two snapshot slots")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warmup", type=int, default=32, help="games played once before timing")
    p.add_argument("--profile", default=None, help="write a cProfile of one run here")
    p.add_argument("--digest", action="store_true", help="print a hash of the buffers")
    p.add_argument("--json", default=None, help="append one JSON line of results here")
    p.add_argument("--label", default="")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    paths = list(DEFAULT_STARTS) + list(DEFAULT_STARTS[:2])
    agents = load_agents(paths[: args.agents])
    owners = ring_owners(args.games, args.agents, args.seed)
    if args.warmup:
        run_once(agents, ring_owners(args.warmup, args.agents, args.seed + 1), args.seed + 1, 0)

    if args.profile:
        from training.rollout_history import _collect_batched_multi

        torch.manual_seed(args.seed)
        prof = cProfile.Profile()
        t0 = time.perf_counter()
        prof.enable()
        _collect_batched_multi(agents, len(owners), torch.device("cpu"), seat_owners=owners,
                               learners=(0, 1), with_calling=True, seed=args.seed)
        prof.disable()
        print(f"profiled run: {time.perf_counter() - t0:.2f}s")
        prof.dump_stats(args.profile)
        return

    walls, digests, all_stats = [], [], []
    for _ in range(args.repeats):
        wall, buffers, stats = run_once(agents, owners, args.seed, args.seed)
        walls.append(wall)
        all_stats.append(stats)
        digests.append(buffer_digest(buffers) if args.digest else "")
    gps = [args.games / w for w in walls]
    mean_stats = {k: statistics.mean(s.get(k, 0.0) for s in all_stats) for k in STAT_KEYS}
    wall = statistics.mean(walls)
    print(f"{args.label} games={args.games} agents={args.agents} threads={args.threads} repeats={args.repeats}")
    print(f"  games/s mean {statistics.mean(gps):.2f}  sd {statistics.stdev(gps) if len(gps) > 1 else 0:.2f}  "
          f"min {min(gps):.2f}  max {max(gps):.2f}")
    print(f"  steps {all_stats[0].get('total_steps', 0):.0f}  wall {wall:.2f}s")
    for k in STAT_KEYS:
        print(f"  {k:24s} {mean_stats[k]:8.2f}s  {100 * mean_stats[k] / wall:5.1f}%")
    if args.digest:
        print(f"  digest {digests[0]}  (all repeats equal: {len(set(digests)) == 1})")
    if args.json:
        with open(args.json, "a") as fh:
            fh.write(json.dumps({
                "label": args.label, "games": args.games, "agents": args.agents,
                "threads": args.threads, "gps": gps, "walls": walls, "stats": mean_stats,
                "digest": digests[0], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }) + "\n")


if __name__ == "__main__":
    main()
