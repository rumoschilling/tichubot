"""Ring league logic: phase schedule, matchups, parity rule, persistent state.

Four agents take turns in strict alternation. In each phase one pair trains
while the other pair is frozen; the trainees play as (a,b), (a,a) and (b,b)
against (c,d), (c,c) and (d,d) -- nine matchups. A phase ends at parity (the
trainees are no longer behind in any matchup) or at a game cap, and the next
phase trains a different pair. ``PHASE_PAIRS`` cycles through all six pair
splits, so every agent trains alongside every other agent, and the pair that
was frozen trains next, against the pair that just caught up.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.promote_best import paired_z

N_AGENTS = 4
# Trainee pair per phase; each split appears in both orientations.
PHASE_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))
PARITY_MAX_DEFICIT_Z = 2.0
# Pooled lead (in pooled standard errors) the trainees need to end a phase.
# 0 restores the original "even or better" rule; once the agents were nearly
# equal that rule ended phases on noise (2026-10-21: 40% at the first check).
PARITY_MIN_POOLED_Z = 2.0


def phase_split(phase_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(trainees, frozen) for a phase index; the schedule repeats every 6."""
    trainees = PHASE_PAIRS[phase_index % len(PHASE_PAIRS)]
    frozen = tuple(a for a in range(N_AGENTS) if a not in trainees)
    return trainees, frozen  # type: ignore[return-value]


def team_options(pair: tuple[int, int]) -> list[tuple[int, int]]:
    a, b = pair
    return [(a, b), (a, a), (b, b)]


def matchups(trainees: tuple[int, int], frozen: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """All nine (trainee team, frozen team) combinations."""
    return [(t, f) for t in team_options(trainees) for f in team_options(frozen)]


def seat_owners_for(team_a: tuple[int, int], team_b: tuple[int, int], swap_a: bool, swap_b: bool) -> tuple[int, int, int, int]:
    """Seats 0,2 for team A and 1,3 for team B, optionally swapping partners."""
    a0, a2 = (team_a[1], team_a[0]) if swap_a else team_a
    b1, b3 = (team_b[1], team_b[0]) if swap_b else team_b
    return (a0, b1, a2, b3)


def sample_seat_owners(
    trainees: tuple[int, int],
    frozen: tuple[int, int],
    n_games: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    """Balanced seat assignments for one rollout batch.

    The nine matchups x two team orientations are cycled as evenly as
    ``n_games`` allows, partner seat order is randomized, and the batch is
    shuffled.
    """
    cells = [(m, trainees_first) for m in matchups(trainees, frozen) for trainees_first in (True, False)]
    order = list(range(len(cells)))
    rng.shuffle(order)
    owners = []
    for k in range(n_games):
        (t_team, f_team), trainees_first = cells[order[k % len(cells)]]
        team_a, team_b = (t_team, f_team) if trainees_first else (f_team, t_team)
        owners.append(seat_owners_for(team_a, team_b, rng.random() < 0.5, rng.random() < 0.5))
    rng.shuffle(owners)
    return owners


def sample_seat_owners_with_history(
    trainees: tuple[int, int],
    frozen: tuple[int, int],
    history_slots: list[int],
    n_games: int,
    history_frac: float,
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    """Like ``sample_seat_owners``, with a share of games against past snapshots.

    ``round(history_frac * n_games)`` games put a trainee partnership against
    a team drawn from ``history_slots`` (agent indices of loaded snapshots;
    the two seats may hold the same or different snapshots). These games keep
    the trainees from forgetting how to beat earlier versions; the rest of
    the batch is the balanced trainee-vs-frozen schedule.
    """
    n_history = round(history_frac * n_games) if history_slots else 0
    owners = sample_seat_owners(trainees, frozen, n_games - n_history, rng)
    t_teams = team_options(trainees)
    for k in range(n_history):
        t_team = t_teams[k % len(t_teams)]
        h_team = (rng.choice(history_slots), rng.choice(history_slots))
        team_a, team_b = (t_team, h_team) if rng.random() < 0.5 else (h_team, t_team)
        owners.append(seat_owners_for(team_a, team_b, rng.random() < 0.5, rng.random() < 0.5))
    rng.shuffle(owners)
    return owners


def latest_phase_checkpoints(phases: list[dict[str, Any]]) -> set[str]:
    """Each agent's most recent phase-end checkpoint (a copy of a live agent)."""
    latest: dict[int, str] = {}
    for phase in phases:
        for agent, path in zip(phase["trainees"], phase["checkpoints"]):
            latest[agent] = path
    return set(latest.values())


def snapshot_weights(
    pool: list[str],
    strengths: dict[str, float],
    exclude: set[str],
    temp: float,
) -> list[float]:
    """Draw weights: softmax of strength / temp over the pool minus ``exclude``.

    A snapshot ``temp`` points per round stronger gets e (~2.7x) the weight,
    so stronger past opponents are favoured without dropping the weak ones.
    ``temp <= 0`` (or missing strengths) means uniform weights.
    """
    allowed = [p not in exclude for p in pool]
    if temp <= 0 or any(p not in strengths for p, ok in zip(pool, allowed) if ok):
        return [1.0 if ok else 0.0 for ok in allowed]
    top = max(strengths[p] for p, ok in zip(pool, allowed) if ok)
    return [math.exp((strengths[p] - top) / temp) if ok else 0.0 for p, ok in zip(pool, allowed)]


def draw_snapshots(
    pool: list[str],
    weights: list[float],
    k: int,
    rng: random.Random,
) -> list[str]:
    """Up to ``k`` distinct snapshots, drawn sequentially by weight."""
    candidates = [(p, w) for p, w in zip(pool, weights) if w > 0]
    drawn = []
    while candidates and len(drawn) < k:
        total = sum(w for _, w in candidates)
        x = rng.random() * total
        for i, (p, w) in enumerate(candidates):
            x -= w
            if x <= 0 or i == len(candidates) - 1:
                drawn.append(p)
                candidates.pop(i)
                break
    return drawn


def learner_seat_counts(seat_owners: list[tuple[int, ...]], learners: tuple[int, ...]) -> dict[int, int]:
    return {a: sum(o == a for owners in seat_owners for o in owners) for a in learners}


def entropy_coef_at(games: float, start: float, end: float | None, decay_games: int | None, decay_start: int = 0) -> float:
    """The lineage's linear entropy schedule at an agent's own game count."""
    if end is None or not decay_games:
        return start
    progress = min(1.0, max(0.0, (games - decay_start) / decay_games))
    return start + progress * (end - start)


# ── Parity evaluation ────────────────────────────────────────────────────


def _play_orientation(args: tuple) -> np.ndarray:
    """Worker job: one matchup orientation over fixed deals (team-A diffs)."""
    from training.arena import ArenaModel, play_matchup
    from training.eval import _spec_head_deciders
    from training.rollout_history import _build_worker_agent

    payloads, seating, deal_seeds, torch_seed, rng_seed, device_str, n_threads, batch = args
    if n_threads:
        torch.set_num_threads(n_threads)
    device = torch.device(device_str)
    models = []
    for slot, (weights, gt_w, pass_w, enc_cfg, mdl_cfg) in enumerate(payloads):
        policy, gt_policy, pass_policy = _build_worker_agent(
            weights, gt_w, pass_w, enc_cfg, mdl_cfg, device, slot=("eval", slot))
        dragon, wish = _spec_head_deciders(policy, "history", device, f"agent{slot}")
        models.append(ArenaModel(f"agent{slot}", policy, "history", pass_policy, gt_policy, dragon, wish))
    torch.manual_seed(torch_seed)
    return np.asarray(play_matchup(
        models, seating, len(deal_seeds), batch=batch, device=device,
        rng=random.Random(rng_seed), game_seeds=deal_seeds, with_calling=True,
    ))


def evaluate_matchups(
    payloads: list[tuple],
    trainees: tuple[int, int],
    frozen: tuple[int, int],
    *,
    deals: int,
    block_size: int,
    seed: int,
    device: torch.device,
    batch: int = 256,
    pool=None,
    n_workers: int = 1,
) -> list[dict[str, Any]]:
    """Deal-paired score advantage of the trainees in each of the 9 matchups.

    ``payloads[k]`` is agent ``k``'s dispatched weights
    (``rollout_history._agent_payload``). Each deal is played with the
    trainee team on seats 0,2 and again on seats 1,3; the two sign-corrected
    outcomes are averaged into one paired effect. The 18 orientation jobs run
    on ``pool`` when given, with results identical to the serial path.
    """
    if deals % block_size or deals // block_size < 2:
        raise ValueError("deals must be a multiple of block_size with >= 2 blocks")
    pairs = matchups(trainees, frozen)
    n_threads = max(1, (os.cpu_count() or 1) // max(1, n_workers)) if pool is not None else 0
    jobs = []
    for index, (t_team, f_team) in enumerate(pairs):
        deal_seeds = np.random.default_rng(seed + 1000 * index).integers(0, 2**63, size=deals, dtype=np.int64)
        for orientation, seating in enumerate((
            seat_owners_for(t_team, f_team, False, False),
            seat_owners_for(f_team, t_team, False, False),
        )):
            base = seed + 1000 * index + 1 + 2 * orientation
            jobs.append((payloads, seating, deal_seeds, base, base + 1, str(device), n_threads, batch))
    diffs = pool.map(_play_orientation, jobs) if pool is not None else [_play_orientation(j) for j in jobs]

    results = []
    for index, (t_team, f_team) in enumerate(pairs):
        trainees_a, frozen_a = diffs[2 * index], diffs[2 * index + 1]
        effects = (trainees_a - frozen_a) / 2.0
        block_means = effects.reshape(-1, block_size).mean(axis=1)
        results.append({
            "trainee_team": list(t_team),
            "frozen_team": list(f_team),
            "deals": deals,
            "block_means": block_means.tolist(),
            # Deals are independent, so the per-deal paired effects give a far
            # steadier standard error than a handful of block means.
            **paired_z(effects),
        })
    return results


def pooled_stats(results: list[dict[str, Any]]) -> tuple[float, float]:
    """Mean advantage over the matchups and its standard error.

    Matchups use independent deal sets, so their variances add.
    """
    n = len(results)
    mean = float(np.mean([r["mean"] for r in results]))
    se = float(np.sqrt(sum(r["se"] ** 2 for r in results)) / n)
    return mean, se


def parity_reached(
    results: list[dict[str, Any]],
    max_deficit_z: float = PARITY_MAX_DEFICIT_Z,
    min_pooled_z: float = PARITY_MIN_POOLED_Z,
) -> bool:
    """Trainees caught up: clearly ahead overall, and not clearly behind anywhere.

    The pooled advantage over all nine matchups must be at least
    ``min_pooled_z`` pooled standard errors (with 0: merely >= 0), and no
    single matchup may be ``max_deficit_z`` standard errors or more behind.
    """
    mean, se = pooled_stats(results)
    ahead = mean >= min_pooled_z * se if min_pooled_z > 0 else mean >= 0.0
    return ahead and all(r["z"] > -max_deficit_z for r in results)


def summarize(results: list[dict[str, Any]], min_pooled_z: float = PARITY_MIN_POOLED_Z) -> dict[str, Any]:
    mean, se = pooled_stats(results)
    return {
        "pooled_mean": mean,
        "pooled_se": se,
        "pooled_z": mean / se if se > 0 else 0.0,
        "min_pooled_z": min_pooled_z,
        "worst_z": float(min(r["z"] for r in results)),
        "best_z": float(max(r["z"] for r in results)),
        "parity": parity_reached(results, min_pooled_z=min_pooled_z),
        "matchups": [
            {k: r[k] for k in ("trainee_team", "frozen_team", "mean", "se", "z")} for r in results
        ],
    }


# ── Persistent state ─────────────────────────────────────────────────────


@dataclass
class RingState:
    """Everything needed to resume a ring run exactly at a phase boundary or mid-phase."""

    agent_names: list[str]
    agent_checkpoints: list[str]          # latest checkpoint per agent
    agent_games: list[float]              # self-play-equivalent games per agent
    phase_index: int = 0
    phase_games: int = 0                  # ring games played in the current phase
    ring_games: int = 0                   # ring games played in total
    next_check: int = 0                   # phase_games at which parity is next checked
    phases: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    # Past checkpoints mixed in as opponents (start points + phase ends).
    snapshots: list[str] = field(default_factory=list)
    # Points/round vs BEST_CHECKPOINT per snapshot (phase-end screens).
    snapshot_strength: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "RingState":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

