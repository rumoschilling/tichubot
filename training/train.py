"""PPO self-play training for Tichu.

Usage:
    python -m training.train
    python -m training.train --games 500000 --lr 1e-4
    python -m training.train --workers 4
    python -m training.train --resume checkpoints/fixed/latest.pt
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time

import torch

from training.config import TrainConfig
from training.model import TichuPolicy
from training.ppo import ppo_update
from training.rollout import collect_rollouts
from training.stats import get_system_stats


def save_checkpoint(policy, optimizer, games_played: int, cfg: TrainConfig, path: str):
    torch.save(
        {
            "games_played": games_played,
            "model":        policy.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "config":       cfg,
        },
        path,
    )


def load_checkpoint(policy, optimizer, path: str, device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["games_played"])


def train(cfg: TrainConfig, resume: str | None = None):
    device = torch.device(cfg.device)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_dir = os.path.dirname(cfg.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    policy    = TichuPolicy(hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    games_played = 0
    if resume:
        games_played = load_checkpoint(policy, optimizer, resume, device)
        print(f"Resumed from {resume} at {games_played:,} games")

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {n_params:,} parameters")
    print(f"Device: {device}  |  workers: {cfg.n_workers} (CPU count: {os.cpu_count()})")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Games/update: {cfg.games_per_update}  |  "
          f"batch: {cfg.batch_size}  |  "
          f"epochs: {cfg.n_epochs}  |  "
          f"lr: {cfg.lr}")
    print(f"Target: {cfg.total_games:,} games\n")

    # CSV log
    log_exists = os.path.exists(cfg.log_file)
    log_fh = open(cfg.log_file, "a", newline="")
    fields = ["games", "transitions", "mean_return", "win_rate",
              "policy_loss", "value_loss", "entropy", "approx_kl",
              "games_per_sec", "cpu_load", "ram_gb", "vram_mb"]
    writer = csv.DictWriter(log_fh, fieldnames=fields)
    if not log_exists:
        writer.writeheader()
        log_fh.flush()

    # Worker pool: created once, lives for the full training run.
    # 'spawn' avoids PyTorch multiprocessing pitfalls on Linux/Mac/Windows.
    pool = None
    if cfg.n_workers > 1:
        pool = mp.get_context("spawn").Pool(cfg.n_workers)
        print(f"Worker pool ready ({cfg.n_workers} processes)\n")

    transitions_total = 0

    try:
        while games_played < cfg.total_games:
            t0 = time.perf_counter()

            rollout = collect_rollouts(
                policy, cfg.games_per_update, device,
                pool=pool, n_workers=cfg.n_workers,
            )

            transitions_total += len(rollout["obs"])

            loss_stats = ppo_update(policy, optimizer, rollout, cfg, device)

            games_played += cfg.games_per_update
            elapsed = time.perf_counter() - t0
            gps     = cfg.games_per_update / elapsed

            mean_return = float(rollout["returns"].mean())
            win_rate    = float((rollout["returns"] > 0).mean())

            if games_played % cfg.log_every < cfg.games_per_update:
                sys_stats = get_system_stats(device)
                row = {
                    "games":         games_played,
                    "transitions":   transitions_total,
                    "mean_return":   f"{mean_return:.4f}",
                    "win_rate":      f"{win_rate:.3f}",
                    "games_per_sec": f"{gps:.1f}",
                    "cpu_load":      sys_stats["cpu_load_1m"],
                    "ram_gb":        sys_stats["ram_avail_gb"],
                    "vram_mb":       sys_stats["vram_used_mb"],
                    **{k: f"{v:.5f}" for k, v in loss_stats.items()},
                }
                writer.writerow(row)
                log_fh.flush()
                print(
                    f"games={games_played:>8,}  "
                    f"return={mean_return:+.3f}  "
                    f"win={win_rate:.2%}  "
                    f"ploss={loss_stats['policy_loss']:+.4f}  "
                    f"vloss={loss_stats['value_loss']:.4f}  "
                    f"ent={loss_stats['entropy']:.3f}  "
                    f"kl={loss_stats['approx_kl']:.4f}  "
                    f"{gps:.0f} g/s  "
                    f"cpu={sys_stats['cpu_load_1m']}  "
                    f"vram={sys_stats['vram_used_mb']}MB"
                )

            if games_played % cfg.checkpoint_every < cfg.games_per_update:
                path = os.path.join(cfg.checkpoint_dir, f"ckpt_{games_played:08d}.pt")
                save_checkpoint(policy, optimizer, games_played, cfg, path)
                save_checkpoint(policy, optimizer, games_played, cfg,
                                os.path.join(cfg.checkpoint_dir, "latest.pt"))
                print(f"  → checkpoint: {path}")

        print(f"\nDone. {games_played:,} games trained.")
        save_checkpoint(policy, optimizer, games_played, cfg,
                        os.path.join(cfg.checkpoint_dir, "final.pt"))

    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        log_fh.close()


def main():
    parser = argparse.ArgumentParser(description="Train Tichu PPO agent")
    parser.add_argument("--resume",           type=str,   default=None)
    parser.add_argument("--games",            type=int,   default=None)
    parser.add_argument("--lr",               type=float, default=None)
    parser.add_argument("--hidden",           type=int,   default=None)
    parser.add_argument("--games-per-update", type=int,   default=None)
    parser.add_argument("--batch-size",       type=int,   default=None)
    parser.add_argument("--n-epochs",         type=int,   default=None)
    parser.add_argument("--checkpoint-dir",   type=str,   default=None)
    parser.add_argument("--log-file",         type=str,   default=None)
    parser.add_argument("--workers",          type=int,   default=None,
                        help="worker processes for game collection (0 = all CPU cores)")
    parser.add_argument("--device",           type=str,   default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TrainConfig(device=args.device)
    if args.games:            cfg.total_games      = args.games
    if args.lr:               cfg.lr               = args.lr
    if args.hidden:           cfg.hidden_dim       = args.hidden
    if args.games_per_update: cfg.games_per_update = args.games_per_update
    if args.batch_size:       cfg.batch_size       = args.batch_size
    if args.n_epochs:         cfg.n_epochs         = args.n_epochs
    if args.checkpoint_dir:   cfg.checkpoint_dir   = args.checkpoint_dir
    if args.log_file:         cfg.log_file         = args.log_file
    if args.workers is not None:
        cfg.n_workers = os.cpu_count() if args.workers == 0 else args.workers

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
