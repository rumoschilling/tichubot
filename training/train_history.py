"""PPO self-play training for Tichu — history-aware variant.

Uses TichuPolicyHistory (model_history.py) with a TransformerHistoryEncoder.
The static trunk + action embedder are the same as the embed model; a
transformer processes the play history and fuses it with the static path.

Usage:
    python -m training.train_history
    python -m training.train_history --workers 4 --games 500000
    python -m training.train_history --resume checkpoints/history/latest.pt
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import torch

from training.checkpoints import (
    LEGACY_HISTORY_TAG,
    history_norm_flags,
    is_legacy_history,
    migrate_legacy_action_dim,
    migrate_legacy_action_optimizer,
    migrate_legacy_obs_size,
    migrate_legacy_optimizer_obs_size,
    migrate_legacy_pass_optimizer,
    revive_dead_pass_provenance,
)
from training.config import TrainConfig
from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder
from training.model_gt import TichuGrandTichuPolicy
from training.model_passing import TichuPassingPolicy
from training.play_training import (
    PlayTrainingBundle,
    PlayTrainingHooks,
    run_play_training,
    save_play_checkpoint,
)
from training.ppo_history import ppo_update_history
from training.rollout_history import collect_rollouts_history, finish_rollout
from training.rollout_embed import set_action_vec_cache_enabled


def _encoder_cfg(cfg: TrainConfig) -> dict:
    """Extract history encoder kwargs from config."""
    return dict(
        d_model=cfg.hist_d_model,
        nhead=cfg.hist_nhead,
        n_layers=cfg.hist_n_layers,
        ffn_dim=cfg.hist_ffn_dim,
        norm_first=cfg.hist_norm_first,
        activation=cfg.hist_activation,
        use_checkpointing=cfg.hist_checkpointing,
        final_norm=cfg.hist_final_norm,
    )


def _model_cfg(cfg: TrainConfig) -> dict:
    """Extract policy constructor kwargs (excluding history_encoder)."""
    return dict(
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        embed_dim=cfg.embed_dim,
        action_hidden_dim=cfg.action_hidden_dim,
        history_proj_gain=cfg.hist_proj_gain,
        trunk_dims=cfg.trunk_dims,
        mlp_activation=cfg.mlp_activation,
        mlp_layer_norm=cfg.mlp_layer_norm,
        split_actor_critic_history=cfg.split_actor_critic_history,
        static_attention=cfg.static_attention,
        history_pool_norm=cfg.hist_pool_norm,
    )


def _build_policy(cfg: TrainConfig):
    encoder = TransformerHistoryEncoder(**_encoder_cfg(cfg))
    return TichuPolicyHistory(encoder, **_model_cfg(cfg))


def _select_worker_device(cfg: TrainConfig, device: torch.device) -> tuple[torch.device, str | None]:
    explicit = getattr(cfg, "worker_device", None)
    if explicit is not None:
        return torch.device(explicit), None

    if device.type != "cuda":
        return torch.device(device.type), None

    if cfg.synchronous:
        return torch.device(device.type), None

    if torch.cuda.device_count() > 1:
        return torch.device(device.type), None

    return torch.device("cpu"), (
        "Single-GPU async training defaults rollout workers to CPU to avoid "
        "contention with PPO on the same device. Override with "
        "--worker-device cuda if you explicitly want GPU collection."
    )


def _resize_wish_head_for_state_dict(
    policy,
    model_state: dict[str, torch.Tensor],
    device: torch.device,
    context: str,
) -> bool:
    if "wish_head.4.weight" not in model_state:
        return False

    ckpt_out_dim = model_state["wish_head.4.weight"].shape[0]
    curr_out_dim = policy.wish_head[4].weight.shape[0]
    if ckpt_out_dim == curr_out_dim:
        return False

    print(
        f"  [compat] Resizing wish_head from {curr_out_dim} to {ckpt_out_dim} "
        f"to match {context}."
    )
    in_features = policy.wish_head[4].in_features
    policy.wish_head[4] = torch.nn.Linear(in_features, ckpt_out_dim).to(device)
    return True


def _prepare_policy_for_resume(policy, path: str, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    _resize_wish_head_for_state_dict(policy, ckpt["model"], device, path)


def _is_history_param(name: str) -> bool:
    return (
        name.startswith("history_encoder.")
        or name.startswith("history_fuse.")
        or name.startswith("history_pool_norm.")
        or name.startswith("critic_history_pool_norm.")
        or name.startswith("critic_pool_")
        or name.startswith("critic_history_fuse.")
        or name.startswith("critic_history_static_query.")
    )


def _make_optimizer(policy, cfg: TrainConfig):
    history_params = []
    static_params = []

    for name, param in policy.named_parameters():
        if name.startswith("trunk."):
            if cfg.freeze_static_trunk:
                param.requires_grad_(False)
                continue
            static_params.append(param)
        elif _is_history_param(name):
            history_params.append(param)
        else:
            static_params.append(param)

    param_groups = []
    if static_params:
        param_groups.append({"params": static_params, "lr": cfg.lr})
    if history_params:
        param_groups.append(
            {"params": history_params, "lr": cfg.lr * cfg.hist_lr_mult}
        )
    return torch.optim.Adam(param_groups, lr=cfg.lr)

def _optimizer_param_names(policy, optimizer) -> list[str]:
    """Parameter names in the optimizer's flat (state-index) order."""
    names = {id(param): name for name, param in policy.named_parameters()}
    return [names[id(p)] for group in optimizer.param_groups for p in group["params"]]


def _remap_optimizer_state_by_name(ckpt: dict, policy, optimizer) -> dict | None:
    """Re-index a saved optimizer state onto a model with added parameters.

    Adam state is keyed by position, so new modules (e.g. history norms)
    shift every index after them. Rebuild the checkpoint's own architecture,
    group it with the same rule as the current optimizer, map saved index ->
    parameter name -> current index, and leave new parameters without state.
    Returns None when the saved groups do not match that reconstruction.
    """
    from training.checkpoints import _build_policy_for_checkpoint

    old_policy, kind = _build_policy_for_checkpoint(ckpt)
    if kind != "history":
        return None
    current = {name for name in _optimizer_param_names(policy, optimizer)}
    keep_trunk = any(name.startswith("trunk.") for name in current)
    old_groups: list[list[str]] = [[], []]
    for name, _ in old_policy.named_parameters():
        if name.startswith("trunk.") and not keep_trunk:
            continue
        old_groups[1 if _is_history_param(name) else 0].append(name)
    old_groups = [g for g in old_groups if g]
    saved = ckpt["optimizer"]
    if [len(g["params"]) for g in saved["param_groups"]] != [len(g) for g in old_groups]:
        return None
    old_names = [name for group in old_groups for name in group]
    new_index = {name: i for i, name in enumerate(_optimizer_param_names(policy, optimizer))}
    state = {}
    for old_i, name in enumerate(old_names):
        if old_i in saved["state"] and name in new_index:
            state[new_index[name]] = saved["state"][old_i]
    groups, start = [], 0
    for saved_group, group in zip(saved["param_groups"], optimizer.param_groups):
        n = len(group["params"])
        groups.append({**saved_group, "params": list(range(start, start + n))})
        start += n
    added = sorted(set(new_index) - set(old_names))
    print(f"  [compat] Remapped optimizer state by name; new parameters without state: {added}")
    return {"state": state, "param_groups": groups}


def reset_history_fuse(policy, optimizer) -> list[str]:
    """Zero the fusion output layers so the history delta starts at exactly 0.

    With SiLU/Tanh after the zeroed Linear, fused = static features at step 0:
    the policy plays as the checkpoint minus its history contribution, and
    the history branch re-learns from there. Their Adam moments are dropped.
    """
    reset = []
    fuses = [("history_fuse", policy.history_fuse)]
    if getattr(policy, "split_actor_critic_history", False):
        fuses.append(("critic_history_fuse", policy.critic_history_fuse))
    for prefix, fuse in fuses:
        for idx, module in enumerate(fuse):
            if isinstance(module, torch.nn.Linear):
                with torch.no_grad():
                    module.weight.zero_()
                    module.bias.zero_()
                for param in (module.weight, module.bias):
                    optimizer.state.pop(param, None)
                reset.append(f"{prefix}.{idx}")
    return reset


def _load_optimizer_keep_lr(optimizer, state: dict, label: str) -> None:
    """Load optimizer state but keep the learning rates the run was built with.

    ``Optimizer.load_state_dict`` restores the saved hyperparameters, so a
    changed --lr / --pass-lr / --gt-lr would otherwise be ignored on resume.
    """
    intended = [group["lr"] for group in optimizer.param_groups]
    optimizer.load_state_dict(state)
    restored = [group["lr"] for group in optimizer.param_groups]
    for group, lr in zip(optimizer.param_groups, intended):
        group["lr"] = lr
    if restored != intended:
        print(f"  {label} learning rate {restored} from checkpoint -> {intended} (config)")


def load_checkpoint(
    policy, optimizer, path: str, device: torch.device,
    pass_policy=None, pass_optimizer=None,
    gt_policy=None, gt_optimizer=None,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_state = ckpt["model"]
    _resize_wish_head_for_state_dict(policy, model_state, device, path)
    migrate_legacy_obs_size(model_state, path)
    migrate_legacy_action_dim(model_state, path)
    migrate_legacy_optimizer_obs_size(ckpt.get("optimizer", {}))
    policy.load_state_dict(model_state, strict=False)
    migrate_legacy_action_optimizer(policy, optimizer, ckpt["optimizer"])
    try:
        _load_optimizer_keep_lr(optimizer, ckpt["optimizer"], "policy")
    except ValueError as exc:
        remapped = _remap_optimizer_state_by_name(ckpt, policy, optimizer)
        if remapped is not None:
            _load_optimizer_keep_lr(optimizer, remapped, "policy")
        else:
            print(
                "  Skipping optimizer state from checkpoint due to parameter-group "
                f"mismatch: {exc}"
            )
    revive_dead_pass_provenance(policy, optimizer, context=path)
    if is_legacy_history(ckpt) and any(
        history_norm_flags(dict(policy.named_parameters()))
    ):
        # Upgrading a pre-fix checkpoint: the fusion layers were trained on
        # the unnormalized history, so restart them from zero (delta = 0).
        reset = reset_history_fuse(policy, optimizer)
        print(
            f"  Upgraded {LEGACY_HISTORY_TAG} checkpoint to the history norms; "
            f"reset fusion output layers to zero: {reset}"
        )
    if pass_policy is not None and "pass_model" in ckpt:
        migrate_legacy_obs_size(ckpt["pass_model"])
        pass_policy.load_state_dict(ckpt["pass_model"])
        print("  Loaded passing network from checkpoint")
    if pass_optimizer is not None and "pass_optimizer" in ckpt:
        migrate_legacy_optimizer_obs_size(ckpt["pass_optimizer"])
        migrate_legacy_pass_optimizer(
            ckpt["pass_optimizer"],
            expected_params=len(tuple(pass_policy.parameters())),
        )
        try:
            _load_optimizer_keep_lr(pass_optimizer, ckpt["pass_optimizer"], "passing")
        except ValueError as exc:
            print(
                "  Skipping passing optimizer state from checkpoint due to "
                f"parameter-group mismatch: {exc}"
            )
    
    if gt_policy is not None and "gt_model" in ckpt:
        gt_policy.load_state_dict(ckpt["gt_model"])
        print("  Loaded Grand Tichu network from checkpoint")
    if gt_optimizer is not None and "gt_optimizer" in ckpt:
        if getattr(gt_policy, "migrated_from_classifier", False):
            # The legacy classifier head and the fresh adv_head have
            # identical shapes, so the stale Adam moments would load
            # silently onto the wrong parameter.
            print("  Skipping GT optimizer state (classifier-head checkpoint)")
        else:
            try:
                _load_optimizer_keep_lr(gt_optimizer, ckpt["gt_optimizer"], "GT")
            except ValueError as exc:
                print(
                    "  Skipping GT optimizer state from checkpoint due to "
                    f"parameter-group mismatch: {exc}"
                )
    return int(ckpt["games_played"])


def init_from_embed_checkpoint(
    policy,
    path: str,
    device: torch.device,
) -> int:
    """Initialize shared policy weights from an embed checkpoint.

    Copies the static trunk, query head, action embedder, and value head.
    The history encoder/projection stay freshly initialized.
    Returns the source checkpoint's games_played for logging only.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    src = ckpt["model"]
    migrate_legacy_obs_size(src, path)
    migrate_legacy_action_dim(src, path)
    dst = policy.state_dict()

    shared_prefixes = (
        "trunk.",
        "query_head.",
        "action_embed.",
        "action_encoder.",
        "value_head.",
    )
    shared_keys = [key for key in src if key.startswith(shared_prefixes)]
    for key in shared_keys:
        if key not in src:
            raise KeyError(f"Embed checkpoint missing key: {key}")
        if key not in dst:
            raise KeyError(f"History policy missing key: {key}")
        if src[key].shape != dst[key].shape:
            raise ValueError(
                f"Shape mismatch for {key}: {tuple(src[key].shape)} != {tuple(dst[key].shape)}"
            )
        dst[key] = src[key]

    policy.load_state_dict(dst)
    return int(ckpt.get("games_played", 0))


def train(
    cfg: TrainConfig,
    resume: str | None = None,
    init_from_embed: str | None = None,
    reinit_gt: bool = False,
    reset_fuse: bool = False,
):
    device = torch.device(cfg.device)
    set_action_vec_cache_enabled(cfg.use_action_vec_cache)

    policy = _build_policy(cfg).to(device)
    if resume:
        _prepare_policy_for_resume(policy, resume, device)
    optimizer = _make_optimizer(policy, cfg)

    # Passing network
    pass_policy = TichuPassingPolicy(
        hidden_dim=cfg.pass_hidden_dim,
        n_layers=cfg.pass_n_layers,
    ).to(device)
    pass_optimizer = torch.optim.Adam(pass_policy.parameters(), lr=cfg.pass_lr)

    # Grand Tichu network
    gt_policy = TichuGrandTichuPolicy(
        hidden_dim=cfg.gt_hidden_dim,
        n_layers=cfg.gt_n_layers,
    ).to(device)
    
    # SMART INIT: Force 20% GT call probability
    from training.init_gt import init_gt_smartly
    init_gt_smartly(gt_policy)
    
    gt_optimizer = torch.optim.Adam(gt_policy.parameters(), lr=cfg.gt_lr)

    games_played = 0
    if resume:
        games_played = load_checkpoint(
            policy, optimizer, resume, device,
            pass_policy, pass_optimizer,
            gt_policy, gt_optimizer,
        )
        print(f"Resumed from {resume} at {games_played:,} games")
        if reset_fuse:
            reset = reset_history_fuse(policy, optimizer)
            print(f"Reset history fusion output layers to zero: {reset}")
        if reinit_gt:
            init_gt_smartly(gt_policy)
            gt_optimizer = torch.optim.Adam(gt_policy.parameters(), lr=cfg.gt_lr)
            print(
                "Reinitialized Grand Tichu head (smart init), discarding the "
                "checkpoint's GT weights and optimizer state"
            )
    elif init_from_embed:
        src_games = init_from_embed_checkpoint(policy, init_from_embed, device)
        print(
            f"Initialized history policy from embed checkpoint {init_from_embed} "
            f"({src_games:,} source games)"
        )

    n_params = sum(p.numel() for p in policy.parameters())
    n_hist = sum(
        p.numel()
        for n, p in policy.named_parameters()
        if "history" in n
    )
    n_pass = sum(p.numel() for p in pass_policy.parameters())
    print(
        f"Policy (history): {n_params:,} parameters  "
        f"(history encoder+proj: {n_hist:,})"
    )
    print(f"Passing network:  {n_pass:,} parameters")
    if gt_policy:
        n_gt = sum(p.numel() for p in gt_policy.parameters())
        print(f"Grand Tichu network: {n_gt:,} parameters")
    print(
        f"  trunk: hidden={cfg.hidden_dim} layers={cfg.n_layers}  "
        f"dims={cfg.trunk_dims or 'uniform'}  "
        f"activation={cfg.mlp_activation} layer_norm={cfg.mlp_layer_norm}  "
        f"embed_dim={cfg.embed_dim}  action_mlp={cfg.action_hidden_dim}"
    )
    print(
        f"  transformer: d_model={cfg.hist_d_model} nhead={cfg.hist_nhead} "
        f"layers={cfg.hist_n_layers} ffn={cfg.hist_ffn_dim} "
        f"proj_gain={cfg.hist_proj_gain} norm_first={cfg.hist_norm_first} final_norm={cfg.hist_final_norm} pool_norm={cfg.hist_pool_norm} "
        f"activation={cfg.hist_activation} "
        f"checkpointing={cfg.hist_checkpointing}"
    )
    print(
        f"  history_lr_mult={cfg.hist_lr_mult}  "
        f"freeze_static_trunk={cfg.freeze_static_trunk}  "
        f"split_actor_critic={cfg.split_actor_critic_history}  "
        f"static_attention={cfg.static_attention}"
    )
    print(f"Device: {device}  |  workers: {cfg.n_workers} (CPU count: {os.cpu_count()})")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Games/update: {cfg.games_per_update}  |  "
        f"memory batch: {cfg.batch_size}  |  "
        f"optimizer batch: {cfg.optimizer_batch_size or 'full rollout'}  |  "
        f"epochs: {cfg.n_epochs}  |  "
        f"lr: {cfg.lr}"
    )
    print(f"Target: {cfg.total_games:,} games\n")

    pool = None
    if cfg.n_workers > 1:
        pool = mp.get_context("spawn").Pool(cfg.n_workers)
        print(f"Worker pool ready ({cfg.n_workers} processes)\n")

    enc_cfg = _encoder_cfg(cfg)
    mdl_cfg = _model_cfg(cfg)

    worker_device, worker_note = _select_worker_device(cfg, device)
    print(f"Main training device: {device} | Worker collection device: {worker_device}")
    if worker_note:
        print(worker_note)

    bundle = PlayTrainingBundle(
        policy=policy,
        optimizer=optimizer,
        pass_policy=pass_policy,
        pass_optimizer=pass_optimizer,
        gt_policy=gt_policy,
        gt_optimizer=gt_optimizer,
    )

    gt_probe_mode = cfg.train_gt and cfg.gt_probe_every > 0

    def _start_rollout(block: bool):
        return collect_rollouts_history(
            policy,
            cfg.games_per_update,
            worker_device,
            pool=pool,
            n_workers=cfg.n_workers,
            encoder_cfg=enc_cfg,
            model_cfg=mdl_cfg,
            use_rollout_bucketing=cfg.use_rollout_bucketing,
            with_calling=True,
            gt_policy=gt_policy,
            pass_policy=pass_policy,
            block=block,
            # Eps-declares in the adapt stream (2026-07-21): with the head
            # honestly at p~0 the pure-on-policy adapt stream had zero
            # declared rounds; GT_EXPLORE_EPS supplies them at no rollout
            # cost. The head still never trains on these (probe-only).
            # --no-gt-explore turns them off once the head calls on its own.
            disable_gt_explore=not cfg.gt_explore,
        )

    def _collect_gt_probe(n_games: int):
        return collect_rollouts_history(
            policy,
            n_games,
            worker_device,
            pool=pool,
            n_workers=cfg.n_workers,
            encoder_cfg=enc_cfg,
            model_cfg=mdl_cfg,
            use_rollout_bucketing=cfg.use_rollout_bucketing,
            with_calling=True,
            gt_policy=gt_policy,
            pass_policy=pass_policy,
            block=True,
            gt_probe=True,
        )

    def _finish_rollout(result):
        return finish_rollout(result, device)

    def _save_checkpoint(games_played: int, path: str):
        save_play_checkpoint(bundle, games_played, cfg, path)

    mmd = None
    if cfg.mmd_alpha > 0:
        from training.mmd import MMDController

        def _rollback(path: str) -> None:
            load_checkpoint(policy, optimizer, path, device, pass_policy, pass_optimizer,
                            gt_policy, gt_optimizer)

        mmd = MMDController(cfg, device, _save_checkpoint, _rollback, policy)
        mmd.restore(games_played)

    def _ppo_update(rollout):
        return ppo_update_history(policy, optimizer, rollout, cfg, device, mmd=mmd)

    hooks = PlayTrainingHooks(
        start_rollout=_start_rollout,
        finish_rollout=_finish_rollout,
        ppo_update=_ppo_update,
        save_checkpoint=_save_checkpoint,
        metric_fields=(
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_frac",
            *(("magnet_kl", "mmd_round") if mmd is not None else ()),
            "grad_norm",
            "grad_norm_max",
            "grad_clipped_frac",
            "dragon_loss",
            "dragon_entropy",
            "wish_loss",
            "wish_entropy",
            "optimizer_steps",
            "transitions_per_optimizer_step",
            "ppo_seconds",
            "peak_allocated_mb",
            "peak_reserved_mb",
        ),
        supports_async=True,
        show_cycle_timings=True,
        log_system_stats=True,
        collect_gt_probe=_collect_gt_probe if gt_probe_mode else None,
        after_update=mmd.after_update if mmd is not None else None,
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
        description="Train Tichu PPO agent (history-aware)"
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--reinit-gt", action="store_true",
        help="On resume, discard the checkpoint's Grand Tichu head and "
             "re-apply the smart init (use once to recover a collapsed head)")
    parser.add_argument("--init-from-embed", type=str, default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=None, help="Wall-clock cap in hours; checkpoints and stops gracefully when reached")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--action-hidden", type=int, default=None)
    parser.add_argument(
        "--trunk-dims", type=str, default=None,
        help="Comma-separated tapered static trunk widths, ending at --hidden",
    )
    parser.add_argument(
        "--mlp-activation", choices=("tanh", "silu"), default=None
    )
    parser.add_argument("--mlp-layer-norm", action="store_true")
    parser.add_argument("--games-per-update", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--optimizer-batch-size",
        type=int,
        default=None,
        help="Transitions per main-policy optimizer step (default: full rollout)",
    )
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--clip-eps", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--entropy-coef-end", type=float, default=None)
    parser.add_argument("--entropy-decay-games", type=int, default=None)
    parser.add_argument("--entropy-decay-start", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/history"
    )
    parser.add_argument(
        "--log-file", type=str, default="logs/train_history.csv"
    )
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    # History encoder args
    parser.add_argument("--hist-d-model", type=int, default=None)
    parser.add_argument("--hist-nhead", type=int, default=None)
    parser.add_argument("--hist-n-layers", type=int, default=None)
    parser.add_argument("--hist-ffn-dim", type=int, default=None)
    parser.add_argument("--hist-norm-first", action="store_true")
    # History norms are on by default (config.py); the two flags are kept so
    # older launch scripts still parse. --legacy-history-norms reproduces the
    # pre-2026-10-25 ("noLN") architecture for controlled comparisons only.
    parser.add_argument("--hist-final-norm", action="store_true",
                        help="closing LayerNorm on the history encoder output (default on)")
    parser.add_argument("--hist-pool-norm", action="store_true",
                        help="LayerNorm on each pooled history vector before fusion (default on)")
    parser.add_argument("--mmd-alpha", type=float, default=None,
                        help="Magnetic Mirror Descent strength for the play policy (0 = entropy bonus)")
    parser.add_argument("--mmd-epsilon", type=float, default=None, help="uniform share of the magnet")
    parser.add_argument("--mmd-window-updates", type=int, default=None)
    parser.add_argument("--mmd-min-round-games", type=int, default=None)
    parser.add_argument("--mmd-max-round-games", type=int, default=None)
    parser.add_argument("--mmd-eval-deals", type=int, default=None)
    parser.add_argument("--mmd-eval-procs", type=int, default=None)
    parser.add_argument("--mmd-reference", type=str, default=None,
                        help="fixed checkpoint every round end is also scored against")
    parser.add_argument("--legacy-history-norms", action="store_true",
                        help="build the old noLN history branch (no encoder final norm, no pool norm)")
    parser.add_argument(
        "--reset-history-fuse", action="store_true",
        help="On resume, zero the history fusion output layers (history delta = 0). "
             "Upgrading a noLN checkpoint does this automatically.")
    parser.add_argument(
        "--hist-activation", choices=("relu", "gelu"), default=None
    )
    parser.add_argument("--hist-checkpointing", action="store_true")
    parser.add_argument("--split-actor-critic-history", action="store_true")
    parser.add_argument("--static-attention", action="store_true")
    parser.add_argument("--hist-proj-gain", type=float, default=None)
    parser.add_argument("--history-lr-mult", type=float, default=None)
    parser.add_argument("--freeze-static-trunk", action="store_true")
    parser.add_argument("--disable-rollout-bucketing", action="store_true")
    parser.add_argument("--disable-ppo-bucketing", action="store_true")
    parser.add_argument("--disable-action-vec-cache", action="store_true")
    parser.add_argument("--pass-hidden", type=int, default=None)
    parser.add_argument("--pass-n-layers", type=int, default=None)
    parser.add_argument("--pass-lr", type=float, default=None)
    parser.add_argument("--pass-every", type=int, default=None)
    parser.add_argument("--pass-deals", type=int, default=None)
    parser.add_argument("--train-gt", action="store_true", help="Enable Grand Tichu head training")
    parser.add_argument("--disable-gt", action="store_true", help="Disable Grand Tichu training")
    parser.add_argument("--gt-every", type=int, default=None, help="Train Grand Tichu head every N play games")
    parser.add_argument("--gt-deals", type=int, default=None)
    parser.add_argument("--gt-lr", type=float, default=None)
    parser.add_argument("--gt-hidden", type=int, default=None)
    parser.add_argument("--gt-n-layers", type=int, default=None)
    parser.add_argument(
        "--gt-probe-every", type=int, default=None,
        help="Enable probe/adapt duality: every N self-play games, pause and "
             "train the GT head on paired frozen-play rollouts (0 = off)")
    parser.add_argument(
        "--gt-probe-games", type=int, default=None,
        help="Game-equivalents per probe block (2 per paired deal)")
    parser.add_argument(
        "--gt-probe-chunk", type=int, default=None,
        help="Games collected per GT head update inside a probe block")
    parser.add_argument(
        "--gt-probe-practice", type=int, default=None, choices=[0, 1],
        help="1 (default): play/value/spec nets run a PPO update on each "
             "probe chunk (declared-round practice); 0: play frozen in blocks")
    parser.add_argument(
        "--no-gt-explore", action="store_true",
        help="no random eps-declares in the main self-play stream (sensible declarations only)")
    parser.add_argument("--use-amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--synchronous", action="store_true", help="Disable pipelining (Synchronous PPO)")
    parser.add_argument("--use-lazy-gpu-loading", action="store_true", help="Keep large buffers on CPU to prevent OOM")
    parser.add_argument("--worker-device", type=str, default=None, help="Force workers to use specific device (e.g. cpu)")
    args = parser.parse_args()

    if args.train_gt and args.disable_gt:
        parser.error("--train-gt and --disable-gt are mutually exclusive")

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device == "cuda":
        try:
            # Test a small operation to catch "incompatible hardware" errors early.
            # We must perform a compute operation to force a kernel launch.
            (torch.ones(1).cuda() + 1).cpu()
        except Exception as e:
            print(f"CUDA initialization failed (hardware might be incompatible): {e}")
            print("Falling back to CPU...")
            args.device = "cpu"

    cfg = TrainConfig(
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        device=args.device,
        use_amp=args.use_amp,
    )

    if args.games is not None:
        cfg.total_games = args.games
    if args.max_hours is not None:
        cfg.max_hours = args.max_hours
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
    if args.trunk_dims is not None:
        try:
            cfg.trunk_dims = tuple(
                int(value.strip()) for value in args.trunk_dims.split(",")
                if value.strip()
            )
        except ValueError as exc:
            parser.error(f"invalid --trunk-dims: {exc}")
        if not cfg.trunk_dims or any(value <= 0 for value in cfg.trunk_dims):
            parser.error("--trunk-dims must contain positive integers")
        if cfg.trunk_dims[-1] != cfg.hidden_dim:
            parser.error(
                f"--trunk-dims must end at --hidden ({cfg.hidden_dim})"
            )
        cfg.n_layers = len(cfg.trunk_dims)
    if args.mlp_activation is not None:
        cfg.mlp_activation = args.mlp_activation
    if args.mlp_layer_norm:
        cfg.mlp_layer_norm = True
    if args.games_per_update is not None:
        cfg.games_per_update = args.games_per_update
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.optimizer_batch_size is not None:
        if args.optimizer_batch_size <= 0:
            parser.error("--optimizer-batch-size must be positive")
        cfg.optimizer_batch_size = args.optimizer_batch_size
    if args.n_epochs is not None:
        cfg.n_epochs = args.n_epochs
    if args.clip_eps is not None:
        cfg.clip_eps = args.clip_eps
    if args.value_coef is not None:
        cfg.value_coef = args.value_coef
    if args.entropy_coef is not None:
        cfg.entropy_coef = args.entropy_coef
    if args.entropy_coef_end is not None:
        cfg.entropy_coef_end = args.entropy_coef_end
    if args.entropy_decay_games is not None:
        cfg.entropy_decay_games = args.entropy_decay_games
    if args.entropy_decay_start is not None:
        cfg.entropy_decay_start = args.entropy_decay_start
    if args.max_grad_norm is not None:
        cfg.max_grad_norm = args.max_grad_norm
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.checkpoint_every is not None:
        cfg.checkpoint_every = args.checkpoint_every
    if args.workers is not None:
        cfg.n_workers = (
            os.cpu_count() if args.workers == 0 else args.workers
        )
    if args.hist_d_model is not None:
        cfg.hist_d_model = args.hist_d_model
    if args.hist_nhead is not None:
        cfg.hist_nhead = args.hist_nhead
    if args.hist_n_layers is not None:
        cfg.hist_n_layers = args.hist_n_layers
    if args.hist_ffn_dim is not None:
        cfg.hist_ffn_dim = args.hist_ffn_dim
    if args.hist_norm_first:
        cfg.hist_norm_first = True
    if args.hist_final_norm:
        cfg.hist_final_norm = True
    if args.hist_pool_norm:
        cfg.hist_pool_norm = True
    if args.legacy_history_norms:
        cfg.hist_final_norm = cfg.hist_pool_norm = False
    for key in ("mmd_alpha", "mmd_epsilon", "mmd_window_updates", "mmd_min_round_games",
                "mmd_max_round_games", "mmd_eval_deals", "mmd_eval_procs", "mmd_reference"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    if args.hist_activation is not None:
        cfg.hist_activation = args.hist_activation
    if args.hist_checkpointing:
        cfg.hist_checkpointing = True
    if args.split_actor_critic_history:
        cfg.split_actor_critic_history = True
    if args.static_attention:
        cfg.static_attention = True
    if args.hist_proj_gain is not None:
        cfg.hist_proj_gain = args.hist_proj_gain
    if args.history_lr_mult is not None:
        cfg.hist_lr_mult = args.history_lr_mult
    if args.freeze_static_trunk:
        cfg.freeze_static_trunk = True
    if args.disable_rollout_bucketing:
        cfg.use_rollout_bucketing = False
    if args.disable_ppo_bucketing:
        cfg.use_ppo_bucketing = False
    if args.disable_action_vec_cache:
        cfg.use_action_vec_cache = False

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
    if args.train_gt:
        cfg.train_gt = True
    if args.disable_gt:
        cfg.train_gt = False
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
    if args.gt_probe_every is not None:
        cfg.gt_probe_every = args.gt_probe_every
    if args.gt_probe_games is not None:
        cfg.gt_probe_games = args.gt_probe_games
    if args.gt_probe_chunk is not None:
        cfg.gt_probe_chunk = args.gt_probe_chunk
    if args.gt_probe_practice is not None:
        cfg.gt_probe_practice = bool(args.gt_probe_practice)
    if args.no_gt_explore:
        cfg.gt_explore = False
    if args.synchronous:
        cfg.synchronous = True
    if args.use_lazy_gpu_loading:
        cfg.use_lazy_gpu_loading = True
    if args.worker_device:
        cfg.worker_device = args.worker_device

    train(cfg, resume=args.resume, init_from_embed=args.init_from_embed,
          reinit_gt=args.reinit_gt, reset_fuse=args.reset_history_fuse)


if __name__ == "__main__":
    main()
