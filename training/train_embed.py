"""PPO self-play training for Tichu using the action-embedding policy."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import torch

from tichu_env.encoding import (
    COUNTED_TRICK_POINTS_OBS_SIZE,
    STATIC_OBS_SIZE,
)

from training.checkpoints import (
    migrate_legacy_action_dim,
    migrate_legacy_action_optimizer,
    migrate_legacy_obs_size,
    migrate_legacy_optimizer_obs_size,
    migrate_legacy_pass_optimizer,
    revive_dead_pass_provenance,
)
from training.config import TrainConfig
from training.model_embed import TichuPolicyEmbed
from training.model_gt import TichuGrandTichuPolicy
from training.model_passing import TichuPassingPolicy
from training.play_training import (
    PlayTrainingBundle,
    PlayTrainingHooks,
    run_play_training,
    save_play_checkpoint,
)
from training.ppo_embed import ppo_update_embed
from training.rollout_embed import (
    collect_rollouts_embed,
    set_action_vec_cache_enabled,
)


def load_checkpoint(
    policy,
    optimizer,
    path: str,
    device: torch.device,
    pass_policy=None,
    pass_optimizer=None,
    gt_policy=None,
    gt_optimizer=None,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    migrate_legacy_obs_size(ckpt["model"], "embed training checkpoint")
    migrate_legacy_action_dim(ckpt["model"], "embed training checkpoint")
    migrate_legacy_optimizer_obs_size(ckpt.get("optimizer", {}))
    policy.load_state_dict(ckpt["model"], strict=False)
    migrate_legacy_action_optimizer(policy, optimizer, ckpt["optimizer"])
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except ValueError:
        print("  Warning: Optimizer state mismatch. Re-initializing optimizer.")
    revive_dead_pass_provenance(
        policy,
        optimizer,
        context="embed training checkpoint",
    )

    if pass_policy is not None and "pass_model" in ckpt:
        pass_policy.load_state_dict(ckpt["pass_model"])
        print("  Loaded passing network from checkpoint")
    if pass_optimizer is not None and "pass_optimizer" in ckpt:
        migrate_legacy_pass_optimizer(
            ckpt["pass_optimizer"],
            expected_params=len(tuple(pass_policy.parameters())),
        )
        try:
            pass_optimizer.load_state_dict(ckpt["pass_optimizer"])
        except ValueError as exc:
            print(
                "  Warning: Passing optimizer state mismatch. "
                f"Re-initializing optimizer: {exc}"
            )

    if gt_policy is not None and "gt_model" in ckpt:
        gt_policy.load_state_dict(ckpt["gt_model"])
        print("  Loaded Grand Tichu network from checkpoint")
    if gt_optimizer is not None and "gt_optimizer" in ckpt:
        if getattr(gt_policy, "migrated_from_classifier", False):
            print("  Skipping GT optimizer state (classifier-head checkpoint)")
        else:
            try:
                gt_optimizer.load_state_dict(ckpt["gt_optimizer"])
            except ValueError:
                pass

    return int(ckpt["games_played"])


def load_pass_checkpoint(
    pass_policy,
    pass_optimizer,
    path: str,
    device: torch.device,
) -> None:
    """Attach a passing model independently from the embed play checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    pass_state = ckpt.get("pass_model")
    if pass_state is None:
        raise ValueError(f"No pass_model found in {path}")

    pass_policy.load_state_dict(pass_state)
    if "pass_optimizer" in ckpt:
        migrate_legacy_pass_optimizer(
            ckpt["pass_optimizer"],
            expected_params=len(tuple(pass_policy.parameters())),
        )
        try:
            pass_optimizer.load_state_dict(ckpt["pass_optimizer"])
        except ValueError as exc:
            print(
                "  Warning: Attached passing optimizer state mismatch. "
                f"Re-initializing optimizer: {exc}"
            )
    print(f"  Attached passing network from {path}")


def load_gt_checkpoint(
    gt_policy,
    gt_optimizer,
    path: str,
    device: torch.device,
    *,
    load_optimizer: bool,
) -> None:
    """Attach a declaration policy independently from the play checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    gt_state = ckpt.get("gt_model")
    if gt_state is None:
        raise ValueError(f"No gt_model found in {path}")

    gt_policy.load_state_dict(gt_state)
    if load_optimizer and "gt_optimizer" in ckpt:
        if getattr(gt_policy, "migrated_from_classifier", False):
            print("  Skipping GT optimizer state (classifier-head checkpoint)")
        else:
            try:
                gt_optimizer.load_state_dict(ckpt["gt_optimizer"])
            except ValueError as exc:
                print(
                    "  Warning: Attached GT optimizer state mismatch. "
                    f"Re-initializing optimizer: {exc}"
                )
    mode = "trainable" if load_optimizer else "frozen"
    print(f"  Attached {mode} Grand Tichu network from {path}")


def train(
    cfg: TrainConfig,
    resume: str | None = None,
    train_gt: bool | None = None,
    pass_checkpoint: str | None = None,
    gt_checkpoint: str | None = None,
    freeze_play: bool = False,
):
    device = torch.device(cfg.device)
    set_action_vec_cache_enabled(cfg.use_action_vec_cache)
    if train_gt is not None:
        cfg.train_gt = train_gt
    games_played = 0
    obs_size = (
        COUNTED_TRICK_POINTS_OBS_SIZE
        if cfg.counted_trick_points
        else STATIC_OBS_SIZE
    )

    policy = TichuPolicyEmbed(
        obs_size=obs_size,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        embed_dim=cfg.embed_dim,
        action_hidden_dim=cfg.action_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    pass_policy = TichuPassingPolicy(
        hidden_dim=cfg.pass_hidden_dim,
        n_layers=cfg.pass_n_layers,
    ).to(device)
    pass_optimizer = torch.optim.Adam(pass_policy.parameters(), lr=cfg.pass_lr)

    gt_policy = None
    gt_optimizer = None
    if cfg.train_gt or gt_checkpoint is not None:
        gt_policy = TichuGrandTichuPolicy(
            hidden_dim=cfg.gt_hidden_dim,
            n_layers=cfg.gt_n_layers,
        ).to(device)
        from training.init_gt import init_gt_smartly
        init_gt_smartly(gt_policy)
        gt_optimizer = torch.optim.Adam(gt_policy.parameters(), lr=cfg.gt_lr)

    if resume:
        games_played = load_checkpoint(
            policy,
            optimizer,
            resume,
            device,
            pass_policy,
            pass_optimizer,
            gt_policy,
            gt_optimizer,
        )
        print(f"Resumed from {resume} at {games_played:,} games")
    if pass_checkpoint is not None:
        load_pass_checkpoint(
            pass_policy, pass_optimizer, pass_checkpoint, device
        )
    if gt_checkpoint is not None:
        load_gt_checkpoint(
            gt_policy,
            gt_optimizer,
            gt_checkpoint,
            device,
            load_optimizer=cfg.train_gt,
        )

    n_params = sum(p.numel() for p in policy.parameters())
    n_pass = sum(p.numel() for p in pass_policy.parameters())
    print(
        f"Policy (embed): {n_params:,} parameters  embed_dim={cfg.embed_dim}  "
        f"action_mlp={cfg.action_hidden_dim}"
    )
    if cfg.counted_trick_points:
        print(
            "Observation: 196 static features + 4 relative captured-point totals"
        )
    print(f"Passing network: {n_pass:,} parameters")
    if gt_policy is not None:
        n_gt = sum(p.numel() for p in gt_policy.parameters())
        print(f"Grand Tichu network: {n_gt:,} parameters")

    print(f"Device: {device}  |  workers: {cfg.n_workers}")
    if freeze_play:
        updating = "passing and GT" if cfg.train_gt else "passing"
        print(f"Play policy frozen; only {updating} updates")
    print(
        f"Games/update: {cfg.games_per_update}  |  "
        f"batch: {cfg.batch_size}  |  "
        f"epochs: {cfg.n_epochs}  |  "
        f"lr: {cfg.lr}"
    )
    print(f"Target: {cfg.total_games:,} games\n")

    pool = None
    if cfg.n_workers > 1:
        pool = mp.get_context("spawn").Pool(cfg.n_workers)
        print(f"Worker pool ready ({cfg.n_workers} processes)\n")

    mdl_cfg = {
        "obs_size": obs_size,
        "hidden_dim": cfg.hidden_dim,
        "n_layers": cfg.n_layers,
        "embed_dim": cfg.embed_dim,
        "action_hidden_dim": cfg.action_hidden_dim,
    }
    pass_mdl_cfg = {
        "hidden_dim": cfg.pass_hidden_dim,
        "n_layers": cfg.pass_n_layers,
    }
    gt_mdl_cfg = {
        "hidden_dim": cfg.gt_hidden_dim,
        "n_layers": cfg.gt_n_layers,
    }

    bundle = PlayTrainingBundle(
        policy=policy,
        optimizer=optimizer,
        pass_policy=pass_policy,
        pass_optimizer=pass_optimizer,
        gt_policy=gt_policy,
        gt_optimizer=gt_optimizer,
    )

    def _start_rollout(_block: bool):
        return collect_rollouts_embed(
            policy,
            cfg.games_per_update,
            device,
            pool=pool,
            n_workers=cfg.n_workers,
            with_calling=gt_policy is not None,
            gt_policy=gt_policy,
            pass_policy=pass_policy,
            model_cfg=mdl_cfg,
            pass_model_cfg=pass_mdl_cfg,
            gt_model_cfg=gt_mdl_cfg,
            counted_trick_points=cfg.counted_trick_points,
        )

    def _finish_rollout(rollout):
        return rollout

    def _ppo_update(rollout):
        if freeze_play:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
            }
        return ppo_update_embed(policy, optimizer, rollout, cfg, device)

    def _save_checkpoint(games_played: int, path: str):
        save_play_checkpoint(bundle, games_played, cfg, path)

    hooks = PlayTrainingHooks(
        start_rollout=_start_rollout,
        finish_rollout=_finish_rollout,
        ppo_update=_ppo_update,
        save_checkpoint=_save_checkpoint,
        metric_fields=("policy_loss", "value_loss", "entropy", "approx_kl"),
        supports_async=False,
        show_cycle_timings=False,
        log_system_stats=False,
    )

    run_play_training(
        cfg,
        bundle,
        hooks,
        device,
        pool=pool,
        initial_games_played=games_played,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Tichu PPO agent (action embedding)"
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--pass-checkpoint",
        type=str,
        default=None,
        help="Attach pass_model from a separate checkpoint",
    )
    parser.add_argument(
        "--freeze-play",
        action="store_true",
        help="Keep the embed play/value policy fixed while training passing",
    )
    parser.add_argument(
        "--gt-checkpoint",
        type=str,
        default=None,
        help="Attach a frozen gt_model from a separate checkpoint",
    )
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--action-hidden", type=int, default=None)
    parser.add_argument(
        "--counted-trick-points",
        action="store_true",
        help="Append each relative player's captured trick-point total",
    )
    parser.add_argument("--games-per-update", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--clip-eps", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/embed")
    parser.add_argument("--log-file", type=str, default="logs/train_embed.csv")
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--pass-hidden", type=int, default=None)
    parser.add_argument("--pass-n-layers", type=int, default=None)
    parser.add_argument("--pass-lr", type=float, default=None)
    parser.add_argument("--pass-every", type=int, default=None)
    parser.add_argument("--pass-deals", type=int, default=None)
    parser.add_argument("--disable-rollout-bucketing", action="store_true")
    parser.add_argument("--disable-ppo-bucketing", action="store_true")
    parser.add_argument("--disable-action-vec-cache", action="store_true")
    parser.add_argument(
        "--train-gt", action="store_true", help="Enable Grand Tichu training"
    )
    parser.add_argument("--gt-every", type=int, default=None)
    parser.add_argument("--gt-deals", type=int, default=None)
    parser.add_argument("--gt-lr", type=float, default=None)
    parser.add_argument("--gt-hidden", type=int, default=None)
    parser.add_argument("--gt-n-layers", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        device=args.device,
    )
    cfg.train_gt = args.train_gt
    cfg.counted_trick_points = args.counted_trick_points
    if args.games is not None:
        cfg.total_games = args.games
    if args.lr is not None:
        cfg.lr = args.lr
    if args.hidden is not None:
        cfg.hidden_dim = args.hidden
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.embed_dim is not None:
        cfg.embed_dim = args.embed_dim
    if args.action_hidden is not None:
        cfg.action_hidden_dim = args.action_hidden
    if args.games_per_update is not None:
        cfg.games_per_update = args.games_per_update
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.n_epochs is not None:
        cfg.n_epochs = args.n_epochs
    if args.clip_eps is not None:
        cfg.clip_eps = args.clip_eps
    if args.value_coef is not None:
        cfg.value_coef = args.value_coef
    if args.entropy_coef is not None:
        cfg.entropy_coef = args.entropy_coef
    if args.max_grad_norm is not None:
        cfg.max_grad_norm = args.max_grad_norm
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.workers is not None:
        cfg.n_workers = os.cpu_count() if args.workers == 0 else args.workers
    if args.max_hours is not None:
        cfg.max_hours = args.max_hours
    if args.pass_hidden is not None:
        cfg.pass_hidden_dim = args.pass_hidden
    if args.pass_n_layers is not None:
        cfg.pass_n_layers = args.pass_n_layers
    if args.pass_lr is not None:
        cfg.pass_lr = args.pass_lr
    if args.pass_every is not None:
        cfg.pass_every = args.pass_every
    if args.pass_deals is not None:
        cfg.pass_deals = args.pass_deals
    if args.disable_rollout_bucketing:
        cfg.use_rollout_bucketing = False
    if args.disable_ppo_bucketing:
        cfg.use_ppo_bucketing = False
    if args.disable_action_vec_cache:
        cfg.use_action_vec_cache = False
    if args.gt_every is not None:
        cfg.gt_every = args.gt_every
    if args.gt_deals is not None:
        cfg.gt_deals = args.gt_deals
    if args.gt_lr is not None:
        cfg.gt_lr = args.gt_lr
    if args.gt_hidden is not None:
        cfg.gt_hidden_dim = args.gt_hidden
    if args.gt_n_layers is not None:
        cfg.gt_n_layers = args.gt_n_layers

    train(
        cfg,
        resume=args.resume,
        pass_checkpoint=args.pass_checkpoint,
        gt_checkpoint=args.gt_checkpoint,
        freeze_play=args.freeze_play,
    )


if __name__ == "__main__":
    main()
