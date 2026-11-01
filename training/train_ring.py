"""Ring league trainer: four agents, one pair trains while the other is frozen.

See ``training/ring.py`` for the schedule and the parity rule. Each agent
keeps its own optimizers and its lineage's training recipe; only the
opponents differ from ordinary self-play. Every agent advances along the
lineage's entropy schedule by its own self-play-equivalent games (seats it
played / 4), so resuming from an early checkpoint does not jump the
coefficient.

Usage:
    python -m training.train_ring --out ring_runs/ring1 --device cuda --workers 8
    python -m training.train_ring --out ring_runs/ring1 --resume      # continue
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import multiprocessing as mp
import os
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from training.arena import ArenaModel, load_arena_model, play_matchup
from training.checkpoints import history_norm_flags
from training.config import TrainConfig
from training.eval import _spec_head_deciders
from training.model_gt import TichuGrandTichuPolicy
from training.model_passing import TichuPassingPolicy
from training.play_training import (
    PASS_BUFFER_KEYS,
    PlayTrainingBundle,
    _run_gt_probe_block,
    _train_passing_step,
    save_play_checkpoint,
)
from training.ppo_history import ppo_update_history
from training.ring import (
    RingState,
    entropy_coef_at,
    evaluate_matchups,
    learner_seat_counts,
    phase_split,
    draw_snapshots,
    latest_phase_checkpoints,
    sample_seat_owners_with_history,
    snapshot_weights,
    summarize,
)
from training.rollout_embed import set_action_vec_cache_enabled
from training.rollout_history import (
    RolloutAgent,
    _agent_payload,
    collect_rollouts_history,
    collect_rollouts_multi,
)
from training.train_history import (
    _build_policy,
    _encoder_cfg,
    _make_optimizer,
    _model_cfg,
    _prepare_policy_for_resume,
    load_checkpoint,
)

_COMPACT = "kaggle_runs/2026-08-13_132440_compact-scratch/tichubot_outputs/checkpoints/history_compact_scratch"
DEFAULT_STARTS = [
    f"{_COMPACT}/ckpt_00250880.pt",
    f"{_COMPACT}/ckpt_00500736.pt",
    f"{_COMPACT}/ckpt_00750592.pt",
    f"{_COMPACT}/ckpt_00800768.pt",
]
METRICS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")


@dataclass
class RingAgent:
    name: str
    cfg: TrainConfig
    bundle: PlayTrainingBundle
    enc_cfg: dict
    mdl_cfg: dict
    games: float
    checkpoint_dir: Path
    pass_buffer: list = field(default_factory=list)

    def rollout_agent(self) -> RolloutAgent:
        b = self.bundle
        return RolloutAgent(b.policy, b.gt_policy, b.pass_policy)

    def arena_model(self, device: torch.device) -> ArenaModel:
        """A frozen copy on ``device``, independent of the training modules."""
        b = self.bundle
        policy, pass_policy, gt_policy = (
            copy.deepcopy(m).to(device).eval() for m in (b.policy, b.pass_policy, b.gt_policy)
        )
        dragon, wish = _spec_head_deciders(policy, "history", device, self.name)
        return ArenaModel(self.name, policy, "history", pass_policy, gt_policy, dragon, wish)

    def save(self, label: str | None = None) -> str:
        """Write latest.pt (and a named copy); returns the named path."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        games = int(self.games)
        path = self.checkpoint_dir / (label or f"ckpt_{games:08d}.pt")
        save_play_checkpoint(self.bundle, games, self.cfg, str(path))
        save_play_checkpoint(self.bundle, games, self.cfg, str(self.checkpoint_dir / "latest.pt"))
        return str(path)


def load_agent(
    path: str, name: str, args: argparse.Namespace, device: torch.device,
    *, frozen: bool = False,
) -> RingAgent:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg: TrainConfig = copy.deepcopy(ckpt["config"])
    cfg.device = str(device)
    # Trainees get the history norms (a noLN checkpoint is upgraded on load);
    # frozen snapshots keep the architecture they were trained with.
    if frozen:
        cfg.hist_final_norm, cfg.hist_pool_norm = history_norm_flags(ckpt["model"])
    else:
        cfg.hist_final_norm = cfg.hist_pool_norm = True
    # The lineage schedule, stated explicitly: checkpoints store the decayed
    # value, not the schedule's start.
    cfg.entropy_coef = args.entropy_coef
    cfg.entropy_coef_end = args.entropy_coef_end
    cfg.entropy_decay_games = args.entropy_decay_games
    cfg.entropy_decay_start = 0
    for key in ("pass_every", "gt_probe_every", "gt_probe_games", "gt_probe_chunk"):
        if getattr(args, key, None) is not None:
            setattr(cfg, key, getattr(args, key))
    policy = _build_policy(cfg).to(device)
    _prepare_policy_for_resume(policy, path, device)
    optimizer = _make_optimizer(policy, cfg)
    pass_policy = TichuPassingPolicy(hidden_dim=cfg.pass_hidden_dim, n_layers=cfg.pass_n_layers).to(device)
    pass_optimizer = torch.optim.Adam(pass_policy.parameters(), lr=cfg.pass_lr)
    gt_policy = TichuGrandTichuPolicy(hidden_dim=cfg.gt_hidden_dim, n_layers=cfg.gt_n_layers).to(device)
    gt_optimizer = torch.optim.Adam(gt_policy.parameters(), lr=cfg.gt_lr)
    games = load_checkpoint(policy, optimizer, path, device, pass_policy, pass_optimizer, gt_policy, gt_optimizer)
    bundle = PlayTrainingBundle(policy, optimizer, pass_policy, pass_optimizer, gt_policy, gt_optimizer)
    return RingAgent(
        name=name, cfg=cfg, bundle=bundle, enc_cfg=_encoder_cfg(cfg), mdl_cfg=_model_cfg(cfg),
        games=float(games), checkpoint_dir=Path(args.out) / "agents" / name,
    )


def _release_cuda_cache() -> None:
    """Return the trainer's cached GPU memory before workers use the device."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _crossed(before: float, after: float, every: int) -> bool:
    return every > 0 and int(after // every) > int(before // every)


def _prescreen_vs_best(agent: RingAgent, device: torch.device, deals: int, seed: int) -> float:
    """Quick deal-paired mean advantage of an in-memory agent over BEST_CHECKPOINT."""
    from training.checkpoints import read_best_record

    best = load_arena_model(f"best={read_best_record()['path']}", device)
    models = [agent.arena_model(device), best]
    seeds = np.random.default_rng(seed).integers(0, 2**63, size=deals, dtype=np.int64)
    a = play_matchup(models, (0, 1, 0, 1), deals, device=device, rng=random.Random(seed), game_seeds=seeds, with_calling=True)
    b = play_matchup(models, (1, 0, 1, 0), deals, device=device, rng=random.Random(seed + 1), game_seeds=seeds, with_calling=True)
    return float(np.mean((np.asarray(a) - np.asarray(b)) / 2.0))


def train_ring(args: argparse.Namespace) -> RingState:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "ring_state.json"
    device = torch.device(args.device)
    worker_device = torch.device(args.worker_device or args.device)
    set_action_vec_cache_enabled(True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    if args.resume:
        state = RingState.load(state_path)
        print(f"Resuming ring at phase {state.phase_index}, {state.ring_games:,} ring games")
    else:
        if state_path.exists():
            raise SystemExit(f"{state_path} exists; pass --resume or choose a new --out")
        state = RingState(
            agent_names=[f"agent{k}" for k in range(4)],
            agent_checkpoints=list(args.starts),
            agent_games=[0.0] * 4,
            next_check=args.min_phase_games,
        )
    # Fresh seat assignments after a resume rather than replaying the start.
    rng = random.Random(args.seed + state.ring_games)
    agents = [
        load_agent(path, name, args, device)
        for path, name in zip(state.agent_checkpoints, state.agent_names)
    ]
    if not state.snapshots:
        # Start points plus every phase-end checkpoint so far (runs that began
        # before snapshot mixing existed get their history back-filled).
        state.snapshots = list(dict.fromkeys(
            [*args.starts, *(path for p in state.phases for path in p["checkpoints"])]
        ))
    print(f"  snapshot pool: {len(state.snapshots)} checkpoints; "
          f"{args.history_frac:.0%} of games vs {args.history_per_update} drawn per update")
    # Snapshots only ever act: keep them on CPU without optimizer state (rollout
    # workers receive CPU copies of the weights anyway) in a bounded LRU cache.
    # Caching every drawn snapshot on the GPU leaked ~6 MB per snapshot.
    snapshot_cache: collections.OrderedDict[str, RingAgent] = collections.OrderedDict()
    snapshot_device = torch.device("cpu") if args.workers > 1 else worker_device

    def snapshot(path: str) -> RingAgent:
        if path in snapshot_cache:
            snapshot_cache.move_to_end(path)
            return snapshot_cache[path]
        agent = load_agent(path, f"snap:{Path(path).stem}", args, snapshot_device, frozen=True)
        b = agent.bundle
        b.optimizer = b.pass_optimizer = b.gt_optimizer = None
        snapshot_cache[path] = agent
        while len(snapshot_cache) > args.snapshot_cache:
            snapshot_cache.popitem(last=False)
        return agent

    # Strength per snapshot for weighted drawing: phase-end screens already in
    # the state; anything else (the start points) is screened once on CPU.
    for phase in state.phases:
        for entry in phase.get("promotion", []):
            state.snapshot_strength.setdefault(entry["checkpoint"], entry["prescreen_mean"])
    if args.snapshot_temp > 0:
        for path in state.snapshots:
            if path not in state.snapshot_strength:
                state.snapshot_strength[path] = _prescreen_vs_best(
                    snapshot(path), torch.device(args.promote_device), args.prescreen_deals, args.seed + 31)
                print(f"  screened snapshot {Path(path).name}: {state.snapshot_strength[path]:+.1f}/round vs best")

    for k, agent in enumerate(agents):
        state.agent_games[k] = agent.games
        print(f"  {agent.name}: {state.agent_checkpoints[k]} ({agent.games:,.0f} games, "
              f"entropy_coef {entropy_coef_at(agent.games, args.entropy_coef, args.entropy_coef_end, args.entropy_decay_games):.4f})")
    state.save(state_path)

    pool = mp.get_context("spawn").Pool(args.workers) if args.workers > 1 else None
    log_path = out / "ring_log.csv"
    new_log = not log_path.exists()
    log_fh = open(log_path, "a", newline="")
    log = csv.DictWriter(log_fh, fieldnames=[
        "time", "ring_games", "phase", "agent", "agent_games", "entropy_coef", "transitions",
        "mean_return", *METRICS, "update_seconds", "ring_games_per_sec",
    ])
    if new_log:
        log.writeheader()

    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True
        print("\nStop requested; will checkpoint after this update.", flush=True)

    old_int = signal.signal(signal.SIGINT, request_stop)
    old_term = signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    deadline = started + args.max_hours * 3600 if args.max_hours else None

    def save_agents(indices) -> None:
        for k in indices:
            state.agent_checkpoints[k] = agents[k].save()
            state.agent_games[k] = agents[k].games
        state.save(state_path)

    try:
        while state.ring_games < args.total_games and not stop:
            trainees, frozen = phase_split(state.phase_index)
            t0 = time.perf_counter()
            drawn = draw_snapshots(
                state.snapshots,
                snapshot_weights(state.snapshots, state.snapshot_strength,
                                 latest_phase_checkpoints(state.phases), args.snapshot_temp),
                args.history_per_update, rng,
            ) if args.history_frac > 0 else []
            players = agents + [snapshot(path) for path in drawn]
            owners = sample_seat_owners_with_history(
                trainees, frozen, list(range(len(agents), len(players))),
                args.games_per_update, args.history_frac, rng,
            )
            _release_cuda_cache()
            buffers = collect_rollouts_multi(
                [a.rollout_agent() for a in players],
                [(a.enc_cfg, a.mdl_cfg) for a in players],
                owners, trainees, worker_device,
                pool=pool, n_workers=args.workers, with_calling=True,
                seed=rng.randrange(2**31),
            )
            seats = learner_seat_counts(owners, trainees)
            for t in trainees:
                agent, buf = agents[t], buffers[t]
                cfg = agent.cfg
                cfg.entropy_coef = entropy_coef_at(agent.games, args.entropy_coef, args.entropy_coef_end, args.entropy_decay_games)
                coef = cfg.entropy_coef
                stats = ppo_update_history(agent.bundle.policy, agent.bundle.optimizer, buf, cfg, device)
                before, agent.games = agent.games, agent.games + seats[t] / 4
                if "pass_train_obs" in buf:
                    agent.pass_buffer.append({k: buf[k] for k in PASS_BUFFER_KEYS})
                if _crossed(before, agent.games, cfg.pass_every):
                    _train_passing_step(agent.bundle, cfg, device, agent.pass_buffer)
                    agent.pass_buffer = []
                if cfg.train_gt and cfg.gt_probe_every > 0 and _crossed(before, agent.games, cfg.gt_probe_every):
                    _run_gt_probe_block(
                        agent.bundle, cfg, device,
                        lambda n, a=agent: (_release_cuda_cache(), collect_rollouts_history(
                            a.bundle.policy, n, worker_device, pool=pool, n_workers=args.workers,
                            encoder_cfg=a.enc_cfg, model_cfg=a.mdl_cfg, with_calling=True,
                            gt_policy=a.bundle.gt_policy, pass_policy=a.bundle.pass_policy,
                            block=True, gt_probe=True,
                        ))[1],
                        ppo_update=lambda r, a=agent: ppo_update_history(
                            a.bundle.policy, a.bundle.optimizer, r, a.cfg, device),
                        deadline=deadline,
                    )
                log.writerow({
                    "time": f"{time.monotonic() - started:.0f}",
                    "ring_games": state.ring_games + args.games_per_update,
                    "phase": state.phase_index, "agent": agent.name,
                    "agent_games": f"{agent.games:.0f}", "entropy_coef": f"{coef:.5f}",
                    "transitions": len(buf["obs"]), "mean_return": f"{float(buf['returns'].mean()):.4f}",
                    **{m: f"{stats[m]:.5f}" for m in METRICS},
                    "update_seconds": f"{time.perf_counter() - t0:.1f}",
                    "ring_games_per_sec": f"{args.games_per_update / (time.perf_counter() - t0):.1f}",
                })
            log_fh.flush()
            state.ring_games += args.games_per_update
            state.phase_games += args.games_per_update
            for k in trainees:
                state.agent_games[k] = agents[k].games
            print(
                f"[ring {state.ring_games:,}] phase {state.phase_index} "
                f"train {trainees} vs frozen {frozen}  phase games {state.phase_games:,}  "
                f"history {[Path(d).stem for d in drawn]}  "
                f"{args.games_per_update / (time.perf_counter() - t0):.1f} g/s",
                flush=True,
            )

            if _crossed(state.ring_games - args.games_per_update, state.ring_games, args.checkpoint_every):
                save_agents(trainees)

            at_cap = state.phase_games >= args.max_phase_games
            if state.phase_games >= state.next_check or at_cap:
                check_seed = args.seed + 7919 * len(state.checks)
                _release_cuda_cache()
                results = evaluate_matchups(
                    [_agent_payload(a.rollout_agent(), a.enc_cfg, a.mdl_cfg) for a in agents],
                    trainees, frozen,
                    deals=args.eval_deals, block_size=args.eval_block, seed=check_seed,
                    device=worker_device, pool=pool, n_workers=args.workers,
                )
                summary = summarize(results, min_pooled_z=args.parity_pooled_z)
                check = {"ring_games": state.ring_games, "phase": state.phase_index,
                         "phase_games": state.phase_games, "trainees": list(trainees),
                         "frozen": list(frozen), "seed": check_seed, **summary}
                state.checks.append(check)
                (out / "checks").mkdir(exist_ok=True)
                (out / "checks" / f"check_{len(state.checks):04d}.json").write_text(
                    json.dumps({**check, "raw": results}, indent=2) + "\n", encoding="utf-8")
                print(f"  parity check: pooled {summary['pooled_mean']:+.2f}/round "
                      f"(z {summary['pooled_z']:+.2f}, need {args.parity_pooled_z:g}), "
                      f"worst z {summary['worst_z']:+.2f} -> parity={summary['parity']}", flush=True)
                if summary["parity"] or at_cap:
                    finished = []
                    for k in trainees:
                        finished.append(agents[k].save(f"phase{state.phase_index:03d}_{int(agents[k].games):08d}.pt"))
                        state.agent_checkpoints[k] = finished[-1]
                    phase = {"phase": state.phase_index, "trainees": list(trainees), "frozen": list(frozen),
                             "games": state.phase_games, "ended_at_ring_games": state.ring_games,
                             "reason": "parity" if summary["parity"] else "cap",
                             "checkpoints": finished, "final_check": summary}
                    if args.promote or args.snapshot_temp > 0:
                        phase["promotion"] = []
                        for k, path in zip(trainees, finished):
                            # CPU: the rollout workers still hold their GPU memory here.
                            promote_device = torch.device(args.promote_device)
                            screen = _prescreen_vs_best(agents[k], promote_device, args.prescreen_deals, check_seed + 17)
                            state.snapshot_strength[path] = screen
                            entry = {"agent": k, "checkpoint": path, "prescreen_mean": screen}
                            if args.promote and screen > 0:
                                from training.promote_best import evaluate_and_decide
                                entry.update(evaluate_and_decide(path, device=str(promote_device)))
                            phase["promotion"].append(entry)
                            print(f"  promotion screen {agents[k].name}: {screen:+.2f}/round vs best"
                                  + (f" -> promoted={entry.get('promoted')}" if args.promote and screen > 0 else ""), flush=True)
                    state.phases.append(phase)
                    state.snapshots.extend(p for p in finished if p not in state.snapshots)
                    print(f"  phase {state.phase_index} ended ({phase['reason']}) after {state.phase_games:,} games", flush=True)
                    state.phase_index += 1
                    state.phase_games = 0
                    state.next_check = args.min_phase_games
                else:
                    state.next_check = state.phase_games + args.check_every
                # Save agents with the state so a resume never pairs advanced
                # counters with older weights.
                save_agents(trainees)

            if deadline is not None and time.monotonic() >= deadline:
                print(f"\nWall-clock cap ({args.max_hours:g}h) reached.", flush=True)
                stop = True
        save_agents(range(4))
        print(f"Saved ring state at {state.ring_games:,} ring games, phase {state.phase_index}.")
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        if pool is not None:
            pool.terminate()
            pool.join()
        log_fh.close()
    return state


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ring league: 4 agents, strict pair alternation, parity cutoff")
    p.add_argument("--out", required=True, help="run directory (state, logs, agent checkpoints)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--starts", nargs=4, default=DEFAULT_STARTS, help="four start checkpoints")
    p.add_argument("--device", default="cpu")
    p.add_argument("--worker-device", default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--games-per-update", type=int, default=4096,
                   help="ring games per update; each trainee gets ~1 seat per game, "
                        "so 4096 matches the lineage's ~1024 self-play games per update")
    p.add_argument("--total-games", type=int, default=50_000_000)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--min-phase-games", type=int, default=40_960)
    p.add_argument("--check-every", type=int, default=40_960)
    p.add_argument("--max-phase-games", type=int, default=500_000)
    p.add_argument("--parity-pooled-z", type=float, default=2.0,
                   help="pooled lead in standard errors needed to end a phase (0 = even or better)")
    p.add_argument("--eval-deals", type=int, default=512)
    p.add_argument("--eval-block", type=int, default=64)
    p.add_argument("--checkpoint-every", type=int, default=102_400)
    p.add_argument("--entropy-coef", type=float, default=0.03)
    p.add_argument("--entropy-coef-end", type=float, default=0.01)
    p.add_argument("--entropy-decay-games", type=int, default=2_000_000)
    p.add_argument("--promote", action=argparse.BooleanOptionalAction, default=True,
                   help="screen phase-end trainees against BEST_CHECKPOINT and run the promotion gate")
    p.add_argument("--prescreen-deals", type=int, default=256)
    p.add_argument("--history-frac", type=float, default=0.2,
                   help="share of games where trainees face past snapshots instead of the frozen pair")
    p.add_argument("--snapshot-temp", type=float, default=40.0,
                   help="strength softmax temperature (points/round) for drawing snapshots; 0 = uniform")
    p.add_argument("--snapshot-cache", type=int, default=32,
                   help="snapshots kept loaded (CPU, LRU)")
    p.add_argument("--history-per-update", type=int, default=2,
                   help="snapshots drawn from the pool for each update")
    p.add_argument("--promote-device", default="cpu",
                   help="device for the phase-end screen and promotion gate")
    # Lineage values come from each checkpoint's config; override for smoke tests.
    p.add_argument("--pass-every", type=int, default=None)
    p.add_argument("--gt-probe-every", type=int, default=None)
    p.add_argument("--gt-probe-games", type=int, default=None)
    p.add_argument("--gt-probe-chunk", type=int, default=None)
    p.add_argument("--seed", type=int, default=20261015)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    train_ring(args)


if __name__ == "__main__":
    main()
