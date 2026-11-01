"""Shared play-policy training loop for embed/history trainers.

This module keeps the outer PPO self-play loop independent from:
  - how rollouts are collected
  - how a main policy update is applied
  - whether rollout collection is synchronous or pipelined

That separation is the key prerequisite for future arena-style training where
the rollout side chooses opponents from a pool rather than always mirroring the
latest self-play policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import multiprocessing as mp
import os
import signal
import time
from typing import Any, Callable

import numpy as np
import torch

from training.config import TrainConfig
from training.ppo_gt import gt_probe_update, ppo_update_gt
from training.ppo_passing import ppo_update_passing
from training.stats import get_system_stats

GT_BUFFER_KEYS = ("gt_h8", "gt_declare", "gt_log_probs", "gt_returns")
PASS_BUFFER_KEYS = (
    "pass_train_obs",
    "pass_train_hand_cards",
    "pass_train_choices",
    "pass_train_log_probs",
    "pass_train_returns",
)


@dataclass
class PlayTrainingBundle:
    policy: torch.nn.Module
    optimizer: torch.optim.Optimizer
    pass_policy: torch.nn.Module | None = None
    pass_optimizer: torch.optim.Optimizer | None = None
    gt_policy: torch.nn.Module | None = None
    gt_optimizer: torch.optim.Optimizer | None = None


@dataclass
class PlayTrainingHooks:
    start_rollout: Callable[[bool], Any]
    finish_rollout: Callable[[Any], dict[str, np.ndarray]]
    ppo_update: Callable[[dict[str, np.ndarray]], dict[str, float]]
    save_checkpoint: Callable[[int, str], None]
    metric_fields: tuple[str, ...]
    supports_async: bool = False
    show_cycle_timings: bool = False
    log_system_stats: bool = False
    # Blocking collection of n probe-group game-equivalents (gt_probe mode,
    # n % 5 == 0). Required for the probe/adapt duality (cfg.gt_probe_every > 0).
    collect_gt_probe: Callable[[int], dict[str, np.ndarray]] | None = None
    # Called after every main PPO update with (games_played, loss_stats);
    # the MMD controller uses it to move the magnet.
    after_update: Callable[[int, dict[str, float]], None] | None = None


def save_play_checkpoint(
    bundle: PlayTrainingBundle,
    games_played: int,
    cfg: TrainConfig,
    path: str,
) -> None:
    data = {
        "games_played": games_played,
        "model": bundle.policy.state_dict(),
        "optimizer": bundle.optimizer.state_dict(),
        "config": cfg,
    }
    if bundle.pass_policy is not None:
        data["pass_model"] = bundle.pass_policy.state_dict()
    if bundle.pass_optimizer is not None:
        data["pass_optimizer"] = bundle.pass_optimizer.state_dict()
    if bundle.gt_policy is not None:
        data["gt_model"] = bundle.gt_policy.state_dict()
    if bundle.gt_optimizer is not None:
        data["gt_optimizer"] = bundle.gt_optimizer.state_dict()
    torch.save(data, path)


def _status_line(
    games_played: int,
    mean_return: float,
    win_rate: float,
    loss_stats: dict[str, float],
    gps: float,
    sys_stats: dict[str, float] | None,
) -> str:
    parts = [
        f"games={games_played:>8,}",
        f"return={mean_return:+.3f}",
        f"win={win_rate:.2%}",
    ]
    if "policy_loss" in loss_stats:
        parts.append(f"ploss={loss_stats['policy_loss']:+.4f}")
    if "value_loss" in loss_stats:
        parts.append(f"vloss={loss_stats['value_loss']:.4f}")
    if "entropy" in loss_stats:
        parts.append(f"ent={loss_stats['entropy']:.3f}")
    if "approx_kl" in loss_stats:
        parts.append(f"kl={loss_stats['approx_kl']:.4f}")
    if "clip_frac" in loss_stats:
        parts.append(f"clip={loss_stats['clip_frac']:.3f}")
    parts.append(f"{gps:.0f} g/s")
    if sys_stats is not None:
        parts.extend(
            [
                f"cpu={sys_stats['cpu_load_1m']}",
                f"gpu={sys_stats['gpu_util_pct']}%",
                f"vram={sys_stats['vram_used_mb']}MB",
            ]
        )
    return "  ".join(parts)


def _train_passing_step(
    bundle: PlayTrainingBundle,
    cfg: TrainConfig,
    device: torch.device,
    pass_buffer: list[dict[str, np.ndarray]],
) -> None:
    """Update the pass net on decisions harvested from self-play rollouts.

    Realized-return signal (like the GT head): every pass decision in the
    adapt stream is credited with the passer's realized team round return,
    advantage = return − b(pass_obs). The old dedicated V(post)−V(pre)
    rollout pipeline is retired — it was blind to where the cards went.
    """
    if bundle.pass_policy is None or bundle.pass_optimizer is None or not pass_buffer:
        return

    t_pass = time.perf_counter()
    pass_rollout = {
        k: np.concatenate([b[k] for b in pass_buffer], axis=0)
        for k in PASS_BUFFER_KEYS
    }
    pass_stats = ppo_update_passing(
        bundle.pass_policy, bundle.pass_optimizer,
        {
            "pass_obs": pass_rollout["pass_train_obs"],
            "hand_cards": pass_rollout["pass_train_hand_cards"],
            "choices": pass_rollout["pass_train_choices"],
            "log_probs": pass_rollout["pass_train_log_probs"],
            "returns": pass_rollout["pass_train_returns"],
        },
        cfg, device,
    )
    dt_pass = time.perf_counter() - t_pass
    n = len(pass_rollout["pass_train_obs"])
    print(
        f"  pass: {n} decisions  "
        f"ploss={pass_stats['pass_policy_loss']:+.4f}  "
        f"bloss={pass_stats['pass_baseline_loss']:.4f}  "
        f"ent={pass_stats['pass_entropy']:.3f}  "
        f"kl={pass_stats['pass_approx_kl']:.4f}  "
        f"{dt_pass:.1f}s"
    )


def _train_gt_step(
    bundle: PlayTrainingBundle,
    cfg: TrainConfig,
    device: torch.device,
    gt_buffer: list[dict[str, np.ndarray]],
) -> None:
    """Update the GT head on decisions harvested from self-play rollouts.

    The head is only updated here, so everything accumulated since the last
    call was sampled by the current head (+ explore eps) — the stored
    behavior log-probs make the PPO ratio exact.
    """
    if bundle.gt_policy is None or bundle.gt_optimizer is None or not gt_buffer:
        return

    t_gt = time.perf_counter()
    gt_rollout = {
        k: np.concatenate([b[k] for b in gt_buffer], axis=0)
        for k in GT_BUFFER_KEYS
    }
    gt_stats = ppo_update_gt(
        bundle.gt_policy, bundle.gt_optimizer, gt_rollout, cfg, device
    )
    dt_gt = time.perf_counter() - t_gt
    n = len(gt_rollout["gt_h8"])
    declare_rate = float(gt_rollout["gt_declare"].mean())
    print(
        f"  gt:   {n} decisions (declared {declare_rate:.1%})  "
        f"ploss={gt_stats['gt_policy_loss']:+.4f}  "
        f"bloss={gt_stats['gt_baseline_loss']:.4f}  "
        f"ent={gt_stats['gt_entropy']:.3f}  "
        f"kl={gt_stats['gt_approx_kl']:.4f}  "
        f"{dt_gt:.1f}s"
    )


def _release_cuda_cache() -> None:
    """Return the trainer's cached GPU memory before rollout workers need it.

    On a 10 GB card the PPO update's cached blocks plus four GPU rollout
    workers overflowed (ring run, 2026-10-17); releasing between phases
    costs nothing measurable and changes no results.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_gt_probe_block(
    bundle: PlayTrainingBundle,
    cfg: TrainConfig,
    device: torch.device,
    collect_gt_probe: Callable[[int], dict[str, np.ndarray]],
    ppo_update: Callable[[dict[str, np.ndarray]], dict[str, float]] | None = None,
    deadline: float | None = None,
) -> None:
    """One probe block of the probe/adapt duality.

    Each chunk of shared-control probe groups (1 all-quiet control + 4
    sole-declarer branches per deal) trains the GT head on the four
    control-differenced diffs per group. With cfg.gt_probe_practice, the same
    chunk then feeds the standard play PPO update: the sole-declarer branches
    are the play/value nets' only real source of declared-round practice once
    declares are on-policy-rare (4 of every 5 probe rounds), and the control
    keeps a quiet slice in the update. The play policy only steps between
    chunks, so all branches of every group are still collected under
    identical play weights and the diffs stay causally clean.
    """
    if bundle.gt_policy is None or bundle.gt_optimizer is None:
        return

    practice = cfg.gt_probe_practice and ppo_update is not None
    t0 = time.perf_counter()
    n_groups_total = cfg.gt_probe_games // 5
    print(
        f"  gt-probe: block start — {n_groups_total} deals x (1 control + 4 "
        f"sole declarers) ({cfg.gt_probe_games} game-equivalents), "
        f"{'play practices on probe rounds' if practice else 'play policy frozen'}",
        flush=True,
    )
    total = 0
    while total < cfg.gt_probe_games:
        if deadline is not None and time.monotonic() >= deadline:
            print("  gt-probe: wall-clock cap reached — block cut short",
                  flush=True)
            break
        chunk = min(cfg.gt_probe_chunk, cfg.gt_probe_games - total)
        chunk -= chunk % 5
        if chunk == 0:
            break
        _release_cuda_cache()
        buf = collect_gt_probe(chunk)
        stats = gt_probe_update(
            bundle.gt_policy, bundle.gt_optimizer, buf, cfg, device
        )
        play_note = ""
        if practice:
            play_stats = ppo_update(buf)
            play_note = (
                f"ploss={play_stats['policy_loss']:+.4f}  "
                f"pkl={play_stats['approx_kl']:.4f}  "
            )
            # Pass-net practice: realized control-differenced team returns
            # from the declared branches (see rollout_history harvest) — the
            # densest declared-round signal. The paired quiet control is a
            # sharper baseline than b(pass_obs), so these rewards enter as
            # ready-made advantages and skip the baseline entirely.
            if (bundle.pass_policy is not None
                    and bundle.pass_optimizer is not None
                    and "pass_train_obs" in buf):
                pass_stats = ppo_update_passing(
                    bundle.pass_policy, bundle.pass_optimizer,
                    {
                        "pass_obs": buf["pass_train_obs"],
                        "hand_cards": buf["pass_train_hand_cards"],
                        "choices": buf["pass_train_choices"],
                        "log_probs": buf["pass_train_log_probs"],
                        "advantages": buf["pass_train_rewards"],
                    },
                    cfg, device,
                )
                play_note += f"passkl={pass_stats['pass_approx_kl']:.4f}  "
        total += chunk
        print(
            f"  gt-probe: {total}/{cfg.gt_probe_games}  "
            f"probes={len(buf['gt_probe_diff'])}  "
            f"diff={400 * stats['gt_diff_mean']:+.1f}"
            f"±{400 * stats['gt_diff_std']:.0f} pts  "
            f"pos={stats['gt_diff_pos_frac']:.0%}  "
            f"p={stats['gt_mean_p']:.3f}  "
            f"ent={stats['gt_entropy']:.3f}  "
            f"dloss={stats['gt_adv_loss']:.4f}  "
            f"bloss={stats['gt_baseline_loss']:.4f}  "
            f"{play_note}"
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )


def run_play_training(
    cfg: TrainConfig,
    bundle: PlayTrainingBundle,
    hooks: PlayTrainingHooks,
    device: torch.device,
    *,
    pool: mp.pool.Pool | None = None,
    initial_games_played: int = 0,
) -> None:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_dir = os.path.dirname(cfg.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    log_exists = os.path.exists(cfg.log_file) and os.path.getsize(cfg.log_file) > 0
    log_fh = open(cfg.log_file, "a", newline="")
    fields = [
        "games",
        "transitions",
        "mean_return",
        "win_rate",
        *hooks.metric_fields,
        "games_per_sec",
    ]
    if hooks.log_system_stats:
        fields.extend(["cpu_load", "gpu_util", "ram_gb", "vram_mb"])
    if log_exists:
        # Resuming into a log written by an older version: keep its columns
        # (new metrics are dropped, missing ones left blank) so rows stay aligned.
        with open(cfg.log_file, newline="") as fh:
            header = next(csv.reader(fh), None)
        if header:
            fields = header
    writer = csv.DictWriter(log_fh, fieldnames=fields, extrasaction="ignore")
    if not log_exists:
        writer.writeheader()
        log_fh.flush()

    transitions_total = 0
    games_played = initial_games_played
    stop_requested = False
    cfg._entropy_coef_start = cfg.entropy_coef
    gt_buffer: list[dict[str, np.ndarray]] = []
    pass_buffer: list[dict[str, np.ndarray]] = []

    def _request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("\nInterrupt received; will checkpoint and stop after this update.")

    old_sigint = signal.signal(signal.SIGINT, _request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, _request_stop)

    use_async = hooks.supports_async and not cfg.synchronous
    gt_probe_mode = (
        cfg.train_gt
        and cfg.gt_probe_every > 0
        and hooks.collect_gt_probe is not None
    )
    rollout_job = None
    if use_async and games_played < cfg.total_games:
        rollout_job = hooks.start_rollout(False)
    train_start = time.monotonic()

    try:
        while games_played < cfg.total_games:
            t0 = time.perf_counter()
            if rollout_job is None:
                # Synchronous mode collects at the start of the cycle, after
                # every policy/auxiliary update from the preceding cycle.
                _release_cuda_cache()
                rollout_job = hooks.start_rollout(True)
            rollout = hooks.finish_rollout(rollout_job)
            rollout_job = None
            t_rollout = time.perf_counter() - t0

            if (
                cfg.max_hours is not None
                and not stop_requested
                and time.monotonic() - train_start >= cfg.max_hours * 3600
            ):
                stop_requested = True
                print(
                    f"\nWall-clock cap ({cfg.max_hours:g}h) reached; "
                    "will checkpoint and stop after this update."
                )

            transitions_total += len(rollout["obs"])
            if cfg.train_gt and not gt_probe_mode and "gt_h8" in rollout:
                gt_buffer.append({k: rollout[k] for k in GT_BUFFER_KEYS})
            if "pass_train_obs" in rollout:
                pass_buffer.append({k: rollout[k] for k in PASS_BUFFER_KEYS})
            next_games = games_played + cfg.games_per_update
            probe_due = (
                gt_probe_mode
                and not stop_requested
                and next_games % cfg.gt_probe_every < cfg.games_per_update
            )
            should_prefetch = (
                use_async
                and not probe_due
                and not stop_requested
                and next_games < cfg.total_games
            )
            if should_prefetch:
                rollout_job = hooks.start_rollout(False)

            if cfg.entropy_coef_end is not None and cfg.entropy_decay_games:
                anchor = (
                    cfg.entropy_decay_start
                    if cfg.entropy_decay_start is not None
                    else initial_games_played
                )
                progress = min(1.0, max(0.0, (games_played - anchor) / cfg.entropy_decay_games))
                cfg.entropy_coef = cfg._entropy_coef_start + progress * (cfg.entropy_coef_end - cfg._entropy_coef_start)

            t_ppo_start = time.perf_counter()
            loss_stats = hooks.ppo_update(rollout)
            t_ppo = time.perf_counter() - t_ppo_start

            games_played = next_games
            elapsed = time.perf_counter() - t0
            gps = cfg.games_per_update / elapsed

            if hooks.show_cycle_timings:
                print(
                    f"  [cycle] rollout: {t_rollout:.1f}s | "
                    f"ppo: {t_ppo:.1f}s | total: {elapsed:.1f}s",
                    flush=True,
                )

            mean_return = float(rollout["returns"].mean())
            win_rate = float((rollout["returns"] > 0).mean())

            if games_played % cfg.log_every < cfg.games_per_update:
                sys_stats = get_system_stats(device) if hooks.log_system_stats else None
                row = {
                    "games": games_played,
                    "transitions": transitions_total,
                    "mean_return": f"{mean_return:.4f}",
                    "win_rate": f"{win_rate:.3f}",
                    "games_per_sec": f"{gps:.1f}",
                }
                for key in hooks.metric_fields:
                    row[key] = f"{loss_stats[key]:.5f}"
                if sys_stats is not None:
                    row.update(
                        {
                            "cpu_load": sys_stats["cpu_load_1m"],
                            "gpu_util": sys_stats["gpu_util_pct"],
                            "ram_gb": sys_stats["ram_avail_gb"],
                            "vram_mb": sys_stats["vram_used_mb"],
                        }
                    )
                writer.writerow(row)
                log_fh.flush()
                print(_status_line(games_played, mean_return, win_rate, loss_stats, gps, sys_stats))

            if games_played % cfg.checkpoint_every < cfg.games_per_update:
                path = os.path.join(cfg.checkpoint_dir, f"ckpt_{games_played:08d}.pt")
                hooks.save_checkpoint(games_played, path)
                hooks.save_checkpoint(games_played, os.path.join(cfg.checkpoint_dir, "latest.pt"))
                print(f"  -> checkpoint: {path}")

            if hooks.after_update is not None and not stop_requested:
                hooks.after_update(games_played, loss_stats)

            if games_played % cfg.pass_every < cfg.games_per_update:
                _train_passing_step(bundle, cfg, device, pass_buffer)
                pass_buffer = []

            if cfg.train_gt and not gt_probe_mode and games_played % cfg.gt_every < cfg.games_per_update:
                _train_gt_step(bundle, cfg, device, gt_buffer)
                gt_buffer = []

            if probe_due:
                _release_cuda_cache()
                _run_gt_probe_block(
                    bundle, cfg, device, hooks.collect_gt_probe,
                    ppo_update=hooks.ppo_update,
                    deadline=(
                        train_start + cfg.max_hours * 3600
                        if cfg.max_hours is not None else None
                    ),
                )

            # Resume async collection only after probe practice. Synchronous
            # mode naturally collects at the top of the next cycle.
            if (
                use_async
                and rollout_job is None
                and not stop_requested
                and next_games < cfg.total_games
            ):
                rollout_job = hooks.start_rollout(False)

            if stop_requested:
                interrupted = os.path.join(cfg.checkpoint_dir, "latest.pt")
                hooks.save_checkpoint(games_played, interrupted)
                print(f"  -> interrupt checkpoint: {interrupted}")
                break

        if games_played >= cfg.total_games:
            print(f"\nDone. {games_played:,} games trained.")
            hooks.save_checkpoint(games_played, os.path.join(cfg.checkpoint_dir, "final.pt"))

    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if pool is not None:
            pool.terminate()
            pool.join()
        log_fh.close()
