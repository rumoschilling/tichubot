"""Checkpoint discovery and model loading helpers.

This module is intentionally small and dependency-light outside of PyTorch.
Training and evaluation code should use these helpers instead of hard-coding
checkpoint paths or duplicating model-version detection logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

STANDARD_CHECKPOINT = (
    "kaggle_runs/2026-08-01_110222_standard/tichubot_outputs/"
    "checkpoints/history_standard_gpu/latest.pt"
)

# The strongest known checkpoint. It is recorded, with the evidence that
# promoted it, in training/best_checkpoint.json (shipped as package data) and
# changes only through training.promote_best, which requires a candidate to
# beat the current best by at least two standard errors on deal-paired rounds.
BEST_CHECKPOINT_RECORD = Path(__file__).resolve().parent / "best_checkpoint.json"


def read_best_record(path: Path = BEST_CHECKPOINT_RECORD) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record.get("path"), str) or not record["path"]:
        raise ValueError(f"{path} has no checkpoint path")
    return record


BEST_CHECKPOINT: str = read_best_record()["path"]
DEFAULT_PLAY_CHECKPOINT = BEST_CHECKPOINT


MODEL_ALIASES: dict[str, str] = {
    "best": BEST_CHECKPOINT,
    "play": BEST_CHECKPOINT,
    "standard": STANDARD_CHECKPOINT,
    "embed": "checkpoints/embed/latest.pt",
    "embed-700k": "checkpoints/embed/ckpt_00700032.pt",
    "embed-1m": "checkpoints/embed_1M/latest.pt",
    "embed-tiny": "checkpoints/embed_tiny/latest.pt",
    "history-tiny": "checkpoints/tiny_history/latest.pt",
    "tiny-history": "checkpoints/tiny_history/latest.pt",
    "tiny-old": "checkpoints/tiny_history/latest.pt",
    "history-tiny-fresh": "checkpoints/tiny_history_fresh/latest.pt",
    "tiny-history-fresh": "checkpoints/tiny_history_fresh/latest.pt",
    "tiny-fresh": "checkpoints/tiny_history_fresh/latest.pt",
    "tiny-kaggle": "kagglecheckpoints/tiny1.4.pt",
    "history-scratch": "checkpoints/history_scratch_500k_opt2/latest.pt",
    "history-from-embed-1m": "checkpoints/history_from_embed_1M_experiment/latest.pt",
    "v11mil": "kagglecheckpoints/v11mil.pt",
    "v4async-900k": "kagglecheckpoints/v4async900k.pt",
    "v4sync-1m": "kagglecheckpoints/v4sync1M.pt",
    "v4sync-1.4k": "kagglecheckpoints/v4sync1.4k.pt",
    "v4sync-2.8m": "kagglecheckpoints/v4sync2.8M.pt",
}


@dataclass(frozen=True)
class CheckpointInfo:
    ref: str
    path: Path
    kind: str
    games_played: int | str
    size_mb: float
    has_pass_model: bool
    has_gt_model: bool
    config: dict[str, Any]
    legacy_history: bool = False


@dataclass
class LoadedPolicy:
    policy: torch.nn.Module
    kind: str
    checkpoint: dict[str, Any]
    path: Path


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _append_pt(path: Path) -> Path:
    if path.suffix == ".pt":
        return path
    return Path(str(path) + ".pt")


def resolve_checkpoint(ref: str | Path) -> Path:
    """Resolve an alias, file path, run directory, or bare checkpoint name."""
    ref_s = str(ref)
    if ref_s in MODEL_ALIASES:
        return _repo_path(MODEL_ALIASES[ref_s])

    direct = _repo_path(ref_s)
    if direct.is_file():
        return direct
    if direct.is_dir() and (direct / "latest.pt").is_file():
        return direct / "latest.pt"

    in_checkpoints = REPO_ROOT / "checkpoints" / ref_s
    if in_checkpoints.is_dir() and (in_checkpoints / "latest.pt").is_file():
        return in_checkpoints / "latest.pt"
    if _append_pt(in_checkpoints).is_file():
        return _append_pt(in_checkpoints)

    in_kaggle = REPO_ROOT / "kagglecheckpoints" / ref_s
    if in_kaggle.is_dir() and (in_kaggle / "latest.pt").is_file():
        return in_kaggle / "latest.pt"
    if _append_pt(in_kaggle).is_file():
        return _append_pt(in_kaggle)

    known = ", ".join(sorted(MODEL_ALIASES))
    raise FileNotFoundError(
        f"Could not resolve checkpoint '{ref_s}'. Known aliases: {known}"
    )


def detect_kind_from_keys(keys: set[str]) -> str:
    if "history_encoder.input_proj.weight" in keys:
        return "history"
    if "action_embed.weight" in keys:
        return "embed"
    return "fixed"


def detect_kind_from_checkpoint(ckpt: dict[str, Any]) -> str:
    return detect_kind_from_keys(set(ckpt["model"].keys()))


# Fields whose TrainConfig default changed to True after checkpoints were
# pickled without them: a pickled config lacks them in its instance dict, and
# getattr would silently fall back to the new class default.
_INSTANCE_ONLY_FIELDS = frozenset({"hist_final_norm", "hist_pool_norm"})

LEGACY_HISTORY_TAG = "noLN"


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if key in _INSTANCE_ONLY_FIELDS:
        return vars(cfg).get(key, False if default is None else default)
    return getattr(cfg, key, default)


def history_norm_flags(model_state: dict) -> tuple[bool, bool]:
    """(final_norm, pool_norm) as present in a history model's weights."""
    return (
        "history_encoder.encoder.norm.weight" in model_state,
        "history_pool_norm.weight" in model_state,
    )


def is_legacy_history(ckpt: dict) -> bool:
    """True for history checkpoints trained before the history norms (2026-10-25)."""
    return detect_kind_from_checkpoint(ckpt) == "history" and not any(
        history_norm_flags(ckpt["model"])
    )


def tag_legacy_label(label: str, ckpt: dict) -> str:
    """Append "-noLN" to labels of pre-fix history checkpoints (idempotent)."""
    if is_legacy_history(ckpt) and not label.endswith(f"-{LEGACY_HISTORY_TAG}"):
        return f"{label}-{LEGACY_HISTORY_TAG}"
    return label


def _config_summary(cfg: Any) -> dict[str, Any]:
    fields = (
        "hidden_dim",
        "n_layers",
        "embed_dim",
        "action_hidden_dim",
        "counted_trick_points",
        "trunk_dims",
        "mlp_activation",
        "mlp_layer_norm",
        "hist_d_model",
        "hist_nhead",
        "hist_n_layers",
        "hist_ffn_dim",
        "hist_proj_gain",
        "hist_lr_mult",
        "hist_norm_first",
        "hist_final_norm",
        "hist_pool_norm",
        "hist_activation",
        "hist_checkpointing",
        "split_actor_critic_history",
        "static_attention",
        "lr",
        "entropy_coef",
        "clip_eps",
        "batch_size",
        "optimizer_batch_size",
        "n_epochs",
        "games_per_update",
        "n_workers",
        "synchronous",
        "use_rollout_bucketing",
        "use_ppo_bucketing",
        "use_action_vec_cache",
        "train_gt",
        "gt_every",
        "worker_device",
    )
    return {
        field: _cfg_value(cfg, field)
        for field in fields
        if _cfg_value(cfg, field) is not None
    }


def load_raw_checkpoint(ref: str | Path, device: torch.device | str = "cpu") -> tuple[dict[str, Any], Path]:
    path = resolve_checkpoint(ref)
    ckpt = torch.load(path, map_location=torch.device(device), weights_only=False)
    return ckpt, path


def checkpoint_info(ref: str | Path, device: torch.device | str = "cpu") -> CheckpointInfo:
    ckpt, path = load_raw_checkpoint(ref, device)
    return CheckpointInfo(
        ref=str(ref),
        path=path,
        kind=detect_kind_from_checkpoint(ckpt),
        games_played=ckpt.get("games_played", "?"),
        size_mb=path.stat().st_size / (1024 * 1024),
        has_pass_model="pass_model" in ckpt,
        has_gt_model="gt_model" in ckpt,
        config=_config_summary(ckpt.get("config")),
        legacy_history=is_legacy_history(ckpt),
    )


def discover_checkpoints(
    roots: tuple[str, ...] = ("checkpoints", "kagglecheckpoints"),
    *,
    include_all: bool = False,
) -> list[Path]:
    """Return checkpoint files worth listing.

    By default this lists run representatives (`latest.pt`/`final.pt`) plus
    top-level Kaggle checkpoint files. Use include_all=True for every *.pt file.
    """
    paths: list[Path] = []
    for root_s in roots:
        root = _repo_path(root_s)
        if root.is_file() and root.suffix == ".pt":
            paths.append(root)
            continue
        if not root.is_dir():
            continue
        if include_all:
            paths.extend(sorted(root.rglob("*.pt")))
            continue
        paths.extend(sorted(root.glob("*.pt")))
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for name in ("latest.pt", "final.pt"):
                candidate = run_dir / name
                if candidate.is_file():
                    paths.append(candidate)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _build_policy_for_checkpoint(ckpt: dict[str, Any]) -> tuple[torch.nn.Module, str]:
    cfg = ckpt.get("config")
    kind = detect_kind_from_checkpoint(ckpt)

    if kind == "history":
        from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder

        enc = TransformerHistoryEncoder(
            d_model=_cfg_value(cfg, "hist_d_model", 64),
            nhead=_cfg_value(cfg, "hist_nhead", 4),
            n_layers=_cfg_value(cfg, "hist_n_layers", 2),
            ffn_dim=_cfg_value(cfg, "hist_ffn_dim", 128),
            norm_first=_cfg_value(cfg, "hist_norm_first", False),
            activation=_cfg_value(cfg, "hist_activation", "relu"),
            use_checkpointing=_cfg_value(cfg, "hist_checkpointing", False),
            final_norm=history_norm_flags(ckpt["model"])[0],
        )
        policy = TichuPolicyHistory(
            enc,
            hidden_dim=_cfg_value(cfg, "hidden_dim", 256),
            n_layers=_cfg_value(cfg, "n_layers", 2),
            embed_dim=_cfg_value(cfg, "embed_dim", 64),
            action_hidden_dim=_cfg_value(cfg, "action_hidden_dim", 128),
            history_proj_gain=_cfg_value(cfg, "hist_proj_gain", 0.1),
            trunk_dims=_cfg_value(cfg, "trunk_dims"),
            mlp_activation=_cfg_value(cfg, "mlp_activation", "tanh"),
            mlp_layer_norm=_cfg_value(cfg, "mlp_layer_norm", False),
            split_actor_critic_history=_cfg_value(
                cfg, "split_actor_critic_history", False
            ),
            static_attention=_cfg_value(cfg, "static_attention", False),
            history_pool_norm=history_norm_flags(ckpt["model"])[1],
        )
        return policy, kind

    if kind == "embed":
        from training.model_embed import TichuPolicyEmbed
        from tichu_env.encoding import (
            COUNTED_TRICK_POINTS_OBS_SIZE,
            STATIC_OBS_SIZE,
        )

        obs_size = (
            COUNTED_TRICK_POINTS_OBS_SIZE
            if _cfg_value(cfg, "counted_trick_points", False)
            else STATIC_OBS_SIZE
        )

        policy = TichuPolicyEmbed(
            obs_size=obs_size,
            hidden_dim=_cfg_value(cfg, "hidden_dim", 256),
            n_layers=_cfg_value(cfg, "n_layers", 2),
            embed_dim=_cfg_value(cfg, "embed_dim", 64),
            action_hidden_dim=_cfg_value(cfg, "action_hidden_dim", 128),
        )
        return policy, kind

    from training.model import TichuPolicy

    policy = TichuPolicy(
        hidden_dim=_cfg_value(cfg, "hidden_dim", 256),
        n_layers=_cfg_value(cfg, "n_layers", 2),
    )
    return policy, kind


_LEGACY_OBS_SIZE = 195


def migrate_legacy_obs_size(state: dict, context: str = "") -> bool:
    """Zero-pad obs-consuming weights from the 195-dim era to STATIC_OBS_SIZE.

    Checkpoints trained before the partner same-weights bit (obs[195]) have
    first-layer weights of shape (H, 195). Padding with a zero column gives
    the new input no influence, so a migrated model behaves exactly like the
    original until training assigns the bit meaning. Also works on the inner
    dicts of an Adam state (exp_avg / exp_avg_sq mirror parameter shapes).
    """
    from tichu_env.encoding import STATIC_OBS_SIZE

    changed = False
    for key, val in state.items():
        if (
            torch.is_tensor(val)
            and val.dim() == 2
            and val.shape[1] == _LEGACY_OBS_SIZE
        ):
            pad = val.new_zeros(val.shape[0], STATIC_OBS_SIZE - _LEGACY_OBS_SIZE)
            state[key] = torch.cat([val, pad], dim=1)
            changed = True
    if changed and context:
        print(
            f"  [compat] Zero-padded obs weights {_LEGACY_OBS_SIZE} -> "
            f"{STATIC_OBS_SIZE} ({context})",
            flush=True,
        )
    return changed


def migrate_legacy_optimizer_obs_size(optim_state: dict) -> bool:
    """Apply the obs-size migration to an optimizer state dict."""
    changed = False
    for param_state in optim_state.get("state", {}).values():
        changed |= migrate_legacy_obs_size(param_state)
    return changed


def migrate_legacy_action_dim(state: dict, context: str = "") -> bool:
    """Zero-pad play-action weights for the appended Phoenix-choice bit."""
    from training.model_embed import ACTION_DIM, LEGACY_ACTION_DIM

    changed = False
    for key in ("action_embed.weight", "action_encoder.residual.0.weight"):
        value = state.get(key)
        if (
            torch.is_tensor(value)
            and value.ndim == 2
            and value.shape[1] == LEGACY_ACTION_DIM
        ):
            state[key] = torch.cat(
                (
                    value,
                    value.new_zeros(
                        value.shape[0], ACTION_DIM - LEGACY_ACTION_DIM
                    ),
                ),
                dim=1,
            )
            changed = True
    if changed and context:
        print(
            f"  [compat] Zero-padded action weights "
            f"{LEGACY_ACTION_DIM} -> {ACTION_DIM} ({context})",
            flush=True,
        )
    return changed


def migrate_legacy_action_optimizer(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optim_state: dict,
) -> bool:
    """Upgrade old Adam state for the nonlinear 58-wide action encoder.

    The encoder is registered after all legacy modules, and both trainable play
    optimizers retain it as the tail of one parameter group.  Extending that
    saved group preserves every legacy parameter-to-moment mapping while
    leaving the new parameters without state until their first update.
    """
    from training.model_embed import ACTION_DIM, LEGACY_ACTION_DIM

    changed = False
    for parameter_state in optim_state.get("state", {}).values():
        for key, value in tuple(parameter_state.items()):
            if (
                torch.is_tensor(value)
                and value.ndim == 2
                and value.shape[1] == LEGACY_ACTION_DIM
            ):
                pad = value.new_zeros(
                    value.shape[0], ACTION_DIM - LEGACY_ACTION_DIM
                )
                parameter_state[key] = torch.cat((value, pad), dim=1)
                changed = True

    encoder = getattr(policy, "action_encoder", None)
    if encoder is None:
        return changed

    encoder_params = set(encoder.parameters())
    containing_groups = []
    for index, group in enumerate(optimizer.param_groups):
        params = group["params"]
        if any(parameter in encoder_params for parameter in params):
            containing_groups.append(index)
    if len(containing_groups) != 1:
        return changed

    group_index = containing_groups[0]
    current_params = optimizer.param_groups[group_index]["params"]
    encoder_count = len(encoder_params)
    if (
        encoder_count == 0
        or len(current_params) < encoder_count
        or any(
            parameter not in encoder_params
            for parameter in current_params[-encoder_count:]
        )
    ):
        return changed

    saved_groups = optim_state.get("param_groups", [])
    if len(saved_groups) != len(optimizer.param_groups):
        return changed

    # A checkpoint that already contains the nonlinear encoder needs only the
    # 57 -> 58 moment padding above; its parameter ids already match.
    if all(
        len(saved.get("params", [])) == len(current["params"])
        for saved, current in zip(saved_groups, optimizer.param_groups)
    ):
        return changed

    for index, (saved, current) in enumerate(
        zip(saved_groups, optimizer.param_groups)
    ):
        expected = len(current["params"])
        actual = len(saved.get("params", []))
        missing = encoder_count if index == group_index else 0
        if actual != expected - missing:
            return changed

    used_ids = [
        parameter_id
        for group in saved_groups
        for parameter_id in group.get("params", [])
    ]
    next_id = max(used_ids, default=-1) + 1
    saved_groups[group_index]["params"].extend(
        range(next_id, next_id + encoder_count)
    )
    print(
        "  [compat] Added nonlinear action-encoder parameter slots to optimizer",
        flush=True,
    )
    return True


def revive_dead_pass_provenance(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    context: str = "",
) -> tuple[str, ...]:
    """Revive legacy provenance branches whose two weight layers are all zero.

    Historical embed/history policies zero-initialized both the card embedding
    and its projection. That preserved outputs but also made both gradients
    identically zero forever. Reinitializing only the embedding leaves the
    zero-projection output exactly unchanged while allowing the projection to
    learn on the next update.

    When an optimizer has already restored checkpoint state, discard moments
    only for the revived embedding/projection weights. Their historical
    gradients were zero, and retaining a large Adam step counter would distort
    their first real update. Learned projection biases remain untouched.
    """
    from training.model_embed import initialize_pass_provenance_embedding_

    revived: list[str] = []
    named_parameters = dict(policy.named_parameters())
    for branch in ("pass_source", "pass_target"):
        embedding = getattr(policy, f"{branch}_embed", None)
        projection = getattr(policy, f"{branch}_proj", None)
        if embedding is None or projection is None:
            continue
        if torch.count_nonzero(embedding.weight).item() != 0:
            continue
        if torch.count_nonzero(projection.weight).item() != 0:
            continue

        initialize_pass_provenance_embedding_(embedding)
        revived.append(branch)

        if optimizer is not None:
            for suffix in ("embed.weight", "proj.weight"):
                parameter = named_parameters[f"{branch}_{suffix}"]
                optimizer.state.pop(parameter, None)

    if revived:
        where = f" ({context})" if context else ""
        print(
            "  [compat] Revived behavior-neutral pass provenance embeddings: "
            f"{', '.join(revived)}{where}",
            flush=True,
        )
    return tuple(revived)


def migrate_legacy_pass_optimizer(
    optim_state: dict,
    expected_params: int | None = None,
) -> bool:
    """Migrate passing-net Adam state for observation and scorer upgrades.

    Only param index 0 (trunk.0.weight, the first registered parameter) takes
    the pass obs; picked_proj.weight is also (H, 56) and must stay untouched,
    so input padding is keyed by index rather than by shape.

    Candidate-embedding parameters were registered after every legacy
    parameter. Appending their ids to the single Adam parameter group
    preserves all old moments; PyTorch initializes the new states lazily on
    the first optimizer step.
    """
    from training.model_passing import GT_DECL_DIM, NUM_CARDS

    changed = False
    trunk_state = optim_state.get("state", {}).get(0, {})
    for key, val in trunk_state.items():
        if torch.is_tensor(val) and val.dim() == 2 and val.shape[1] == NUM_CARDS:
            pad = val.new_zeros(val.shape[0], GT_DECL_DIM)
            trunk_state[key] = torch.cat([val, pad], dim=1)
            changed = True
    if changed:
        print(
            "  [compat] Zero-padded passing-net optimizer moments "
            f"{NUM_CARDS} -> {NUM_CARDS + GT_DECL_DIM}",
            flush=True,
        )

    groups = optim_state.get("param_groups", [])
    if expected_params is not None and len(groups) == 1:
        params = groups[0].get("params", [])
        if len(params) < expected_params:
            next_id = max(params, default=-1) + 1
            params.extend(
                range(next_id, next_id + expected_params - len(params))
            )
            changed = True
            print(
                "  [compat] Added candidate-embedding parameter slots to "
                "passing-net optimizer",
                flush=True,
            )
    return changed


def _resize_wish_head_for_state_dict(
    policy: torch.nn.Module,
    model_state: dict[str, torch.Tensor],
    device: torch.device,
    context: str,
) -> None:
    if not hasattr(policy, "wish_head") or "wish_head.4.weight" not in model_state:
        return

    wish_head = getattr(policy, "wish_head")
    ckpt_out_dim = model_state["wish_head.4.weight"].shape[0]
    curr_out_dim = wish_head[4].weight.shape[0]
    if ckpt_out_dim == curr_out_dim:
        return

    print(
        f"  [compat] Resizing wish_head from {curr_out_dim} to {ckpt_out_dim} "
        f"to match {context}.",
        flush=True,
    )
    in_features = wish_head[4].in_features
    wish_head[4] = torch.nn.Linear(in_features, ckpt_out_dim).to(device)


def load_policy_checkpoint(
    ref: str | Path,
    device: torch.device | str = "cpu",
    *,
    strict: bool = False,
) -> LoadedPolicy:
    """Load a fixed, embed, or history policy from a checkpoint reference."""
    device_t = torch.device(device)
    ckpt, path = load_raw_checkpoint(ref, device_t)
    policy, kind = _build_policy_for_checkpoint(ckpt)
    model_state = ckpt["model"]
    _resize_wish_head_for_state_dict(policy, model_state, device_t, _display_path(path))
    migrate_legacy_obs_size(model_state, _display_path(path))
    migrate_legacy_action_dim(model_state, _display_path(path))
    policy.load_state_dict(model_state, strict=strict)
    revive_dead_pass_provenance(
        policy,
        context=_display_path(path),
    )
    policy.to(device_t)
    policy.eval()
    return LoadedPolicy(policy=policy, kind=kind, checkpoint=ckpt, path=path)


def format_info(info: CheckpointInfo) -> str:
    games = info.games_played
    games_s = f"{games:,}" if isinstance(games, int) else str(games)
    flags = []
    if info.has_pass_model:
        flags.append("pass")
    if info.has_gt_model:
        flags.append("gt")
    flag_s = ",".join(flags) if flags else "-"
    cfg = info.config
    arch = f"h={cfg.get('hidden_dim', '?')} l={cfg.get('n_layers', '?')}"
    if info.kind in {"embed", "history"}:
        arch += f" e={cfg.get('embed_dim', '?')}"
    if info.kind == "history":
        arch += (
            f" hist={cfg.get('hist_d_model', '?')}/"
            f"{cfg.get('hist_nhead', '?')}/"
            f"{cfg.get('hist_n_layers', '?')}"
        )
    sync_cfg = cfg.get("synchronous")
    mode = "sync" if sync_cfg is True else "async" if sync_cfg is False else "?"
    return (
        f"{_display_path(info.path):55s}  {info.kind:7s}  "
        f"{games_s:>10s} games  {info.size_mb:6.1f}MB  "
        f"{arch:28s}  {mode:5s}  aux={flag_s}"
    )
