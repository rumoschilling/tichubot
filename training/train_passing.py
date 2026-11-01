"""Train the passing network against a frozen play-policy value function.

Typical usage:
    python -m training.train_passing \
        --value-checkpoint checkpoints/embed/ckpt_00700032.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch

from training.config import TrainConfig
from training.model import TichuPolicy
from training.model_embed import TichuPolicyEmbed
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder
from training.model_passing import TichuPassingPolicy
from training.ppo_passing import ppo_update_passing
from training.rollout_passing import collect_passing_rollouts


def _detect_kind(keys: set[str]) -> str:
    if "history_encoder.input_proj.weight" in keys:
        return "history"
    if "action_embed.weight" in keys:
        return "embed"
    return "fixed"


def _load_gt_policy(ckpt: dict, device: torch.device):
    """Build the GT head bundled in the value checkpoint, or None."""
    gt_state = ckpt.get("gt_model")
    if gt_state is None:
        print("No GT model in value checkpoint; passing states carry no declarations")
        return None
    from training.model_gt import TichuGrandTichuPolicy
    cfg = ckpt.get("config")
    gt_policy = TichuGrandTichuPolicy(
        hidden_dim=getattr(cfg, "gt_hidden_dim", 64) if cfg is not None else 64,
        n_layers=getattr(cfg, "gt_n_layers", 2) if cfg is not None else 2,
    ).to(device)
    gt_policy.load_state_dict(gt_state)
    gt_policy.eval()
    print("Loaded GT head from value checkpoint (declarations in passing states)")
    return gt_policy


def _load_value_policy(path: str, device: torch.device):
    from training.checkpoints import load_policy_checkpoint

    loaded = load_policy_checkpoint(path, device=device)
    ckpt = loaded.checkpoint
    policy = loaded.policy
    kind = loaded.kind
    for param in policy.parameters():
        param.requires_grad_(False)

    games = ckpt.get("games_played", "?")
    label = f"{games:,}" if isinstance(games, int) else str(games)
    print(f"Loaded value policy {path}  ({label} games, {kind})")
    return policy, kind, _load_gt_policy(ckpt, device)


def save_checkpoint(
    pass_policy,
    pass_optimizer,
    value_checkpoint: str,
    deals_trained: int,
    cfg: TrainConfig,
    path: str,
):
    torch.save(
        {
            "value_checkpoint": value_checkpoint,
            "deals_trained": deals_trained,
            "pass_model": pass_policy.state_dict(),
            "pass_optimizer": pass_optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    pass_policy,
    pass_optimizer,
    path: str,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    pass_policy.load_state_dict(ckpt["pass_model"])
    from training.checkpoints import migrate_legacy_pass_optimizer
    migrate_legacy_pass_optimizer(
        ckpt["pass_optimizer"],
        expected_params=len(tuple(pass_policy.parameters())),
    )
    pass_optimizer.load_state_dict(ckpt["pass_optimizer"])
    return int(ckpt.get("deals_trained", 0))


def train(
    value_checkpoint: str,
    cfg: TrainConfig,
    rounds: int,
    resume: str | None = None,
    n_workers: int = 1,
):
    device = torch.device(cfg.device)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_dir = os.path.dirname(cfg.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    value_policy, kind, gt_policy = _load_value_policy(value_checkpoint, device)

    pass_policy = TichuPassingPolicy(
        hidden_dim=cfg.pass_hidden_dim,
        n_layers=cfg.pass_n_layers,
    ).to(device)
    pass_optimizer = torch.optim.Adam(pass_policy.parameters(), lr=cfg.pass_lr)

    deals_trained = 0
    if resume:
        deals_trained = load_checkpoint(
            pass_policy, pass_optimizer, resume, device
        )
        print(f"Resumed pass policy from {resume} at {deals_trained:,} deals")

    n_params = sum(p.numel() for p in pass_policy.parameters())
    print(f"Passing network: {n_params:,} parameters")
    print(
        f"Value policy kind: {kind}  |  pass deals/update: {cfg.pass_deals}  |  "
        f"epochs: {cfg.n_epochs}  |  batch: {cfg.batch_size}  |  lr: {cfg.pass_lr}"
    )
    print(f"Target: {rounds} rounds ({rounds * cfg.pass_deals:,} deals)")
    print(f"Parallel workers: {n_workers}\n")

    log_exists = os.path.exists(cfg.log_file)
    log_fh = open(cfg.log_file, "a", newline="")
    writer = csv.DictWriter(
        log_fh,
        fieldnames=[
            "deals",
            "mean_abs_reward",
            "pass_policy_loss",
            "pass_entropy",
            "pass_approx_kl",
            "deals_per_sec",
        ],
    )
    if not log_exists:
        writer.writeheader()
        log_fh.flush()

    pool = None
    if n_workers > 1:
        import multiprocessing as mp
        pool = mp.Pool(n_workers)

    try:
        for round_idx in range(rounds):
            t0 = time.perf_counter()
            rollout = collect_passing_rollouts(
                pass_policy, value_policy, cfg.pass_deals, device,
                n_workers=n_workers, pool=pool, cfg=cfg, gt_policy=gt_policy
            )
            # The team-signed V-delta reward is already baselined (the
            # V(pre) terms are action-independent), so it enters as a
            # ready-made advantage.
            stats = ppo_update_passing(
                pass_policy, pass_optimizer,
                {**rollout, "advantages": rollout["rewards"]},
                cfg, device,
            )
            elapsed = time.perf_counter() - t0
            deals_trained += cfg.pass_deals
            dps = cfg.pass_deals / elapsed
            # Team-signed rewards sum to zero across the four seats of every
            # deal, so the plain mean is 0 by construction; the mean |R|
            # tracks how much the exchange actually moves hand values.
            mean_reward = float(np.abs(rollout["rewards"]).mean())

            row = {
                "deals": deals_trained,
                "mean_abs_reward": f"{mean_reward:.5f}",
                "pass_policy_loss": f"{stats['pass_policy_loss']:.5f}",
                "pass_entropy": f"{stats['pass_entropy']:.5f}",
                "pass_approx_kl": f"{stats['pass_approx_kl']:.5f}",
                "deals_per_sec": f"{dps:.1f}",
            }
            writer.writerow(row)
            log_fh.flush()
            print(
                f"round={round_idx + 1:>3}/{rounds}  "
                f"deals={deals_trained:>7,}  "
                f"|R|={mean_reward:.4f}  "
                f"ploss={stats['pass_policy_loss']:+.4f}  "
                f"ent={stats['pass_entropy']:.3f}  "
                f"kl={stats['pass_approx_kl']:.4f}  "
                f"{dps:.0f} deals/s"
            )

            save_checkpoint(
                pass_policy,
                pass_optimizer,
                value_checkpoint,
                deals_trained,
                cfg,
                os.path.join(cfg.checkpoint_dir, "latest.pt"),
            )

        save_checkpoint(
            pass_policy,
            pass_optimizer,
            value_checkpoint,
            deals_trained,
            cfg,
            os.path.join(cfg.checkpoint_dir, "final.pt"),
        )
        print(f"\nDone. Trained passing network for {deals_trained:,} deals.")
    finally:
        if pool:
            pool.close()
            pool.join()
        log_fh.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train Tichu passing network against a frozen value policy"
    )
    parser.add_argument(
        "--value-checkpoint",
        type=str,
        required=True,
        help="Main play-policy checkpoint used for V(post)-V(pre) rewards",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--pass-deals", type=int, default=None)
    parser.add_argument("--pass-lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/passing",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/train_passing.csv",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = TrainConfig(
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        device=args.device,
    )
    if args.pass_deals:
        cfg.pass_deals = args.pass_deals
    if args.pass_lr:
        cfg.pass_lr = args.pass_lr
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.n_epochs:
        cfg.n_epochs = args.n_epochs

    train(
        value_checkpoint=args.value_checkpoint,
        cfg=cfg,
        rounds=args.rounds,
        resume=args.resume,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
