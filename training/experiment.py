"""Experiment CLI for repeatable training and evaluation commands.

Examples
--------
List known model references:
    python -m training.experiment aliases

Inspect checkpoints:
    python -m training.experiment inspect tiny-history tiny-kaggle

Run an eval without remembering paths:
    python -m training.experiment eval tiny-kaggle tiny-history-fresh --games 500

Build a reproducible training command:
    python -m training.experiment train-command history-tiny --name htiny_sync \
        --games 500000 --device cuda --workers 2 --sync
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys
from typing import Any

import torch

from training.checkpoints import (
    MODEL_ALIASES,
    REPO_ROOT,
    checkpoint_info,
    discover_checkpoints,
    format_info,
    resolve_checkpoint,
)


@dataclass(frozen=True)
class TrainPreset:
    module: str
    args: tuple[str, ...]
    description: str
    supports_history_init: bool = False
    supports_sync: bool = False
    supports_disable_gt: bool = False


TRAIN_PRESETS: dict[str, TrainPreset] = {
    "embed-tiny": TrainPreset(
        module="training.train_embed",
        args=("--hidden", "128", "--n-layers", "2", "--embed-dim", "64"),
        description="Small static/action-embedding policy.",
    ),
    "embed-standard": TrainPreset(
        module="training.train_embed",
        args=("--hidden", "256", "--n-layers", "2", "--embed-dim", "64"),
        description="Original action-embedding policy size.",
    ),
    "history-tiny": TrainPreset(
        module="training.train_history",
        args=(
            "--hidden", "128",
            "--n-layers", "2",
            "--embed-dim", "64",
            "--hist-d-model", "64",
            "--hist-nhead", "2",
            "--hist-n-layers", "2",
            "--hist-ffn-dim", "128",
        ),
        description="Small history transformer policy for quick experiments.",
        supports_history_init=True,
        supports_sync=True,
        supports_disable_gt=True,
    ),
    "history-standard": TrainPreset(
        module="training.train_history",
        args=(
            "--hidden", "256",
            "--n-layers", "4",
            "--embed-dim", "64",
            "--hist-d-model", "128",
            "--hist-nhead", "4",
            "--hist-n-layers", "2",
            "--hist-ffn-dim", "256",
        ),
        description="Current standard history model used on Kaggle T4 runs.",
        supports_history_init=True,
        supports_sync=True,
        supports_disable_gt=True,
    ),
    "history-monster": TrainPreset(
        module="training.train_history",
        args=(
            "--hidden", "512",
            "--n-layers", "4",
            "--embed-dim", "64",
            "--hist-d-model", "256",
            "--hist-nhead", "8",
            "--hist-n-layers", "2",
            "--hist-ffn-dim", "512",
        ),
        description="Large history model; expensive but versioned consistently.",
        supports_history_init=True,
        supports_sync=True,
        supports_disable_gt=True,
    ),
}


FIELD_FLAGS: dict[str, str] = {
    "total_games": "--games",
    "games": "--games",
    "lr": "--lr",
    "clip_eps": "--clip-eps",
    "value_coef": "--value-coef",
    "entropy_coef": "--entropy-coef",
    "max_grad_norm": "--max-grad-norm",
    "hidden_dim": "--hidden",
    "n_layers": "--n-layers",
    "embed_dim": "--embed-dim",
    "games_per_update": "--games-per-update",
    "batch_size": "--batch-size",
    "optimizer_batch_size": "--optimizer-batch-size",
    "n_epochs": "--n-epochs",
    "checkpoint_every": "--checkpoint-every",
    "log_every": "--log-every",
    "hist_d_model": "--hist-d-model",
    "hist_nhead": "--hist-nhead",
    "hist_n_layers": "--hist-n-layers",
    "hist_ffn_dim": "--hist-ffn-dim",
    "hist_proj_gain": "--hist-proj-gain",
    "hist_lr_mult": "--history-lr-mult",
    "history_lr_mult": "--history-lr-mult",
    "pass_every": "--pass-every",
    "pass_deals": "--pass-deals",
    "pass_lr": "--pass-lr",
    "pass_hidden_dim": "--pass-hidden",
    "pass_n_layers": "--pass-n-layers",
    "gt_every": "--gt-every",
    "gt_deals": "--gt-deals",
    "gt_lr": "--gt-lr",
    "gt_hidden_dim": "--gt-hidden",
    "gt_n_layers": "--gt-n-layers",
}


BOOL_TRUE_FLAGS: dict[str, str] = {
    "train_gt": "--train-gt",
    "freeze_static_trunk": "--freeze-static-trunk",
    "use_amp": "--use-amp",
    "synchronous": "--synchronous",
    "use_lazy_gpu_loading": "--use-lazy-gpu-loading",
}


BOOL_FALSE_FLAGS: dict[str, str] = {
    "use_rollout_bucketing": "--disable-rollout-bucketing",
    "use_ppo_bucketing": "--disable-ppo-bucketing",
    "use_action_vec_cache": "--disable-action-vec-cache",
    "train_gt": "--disable-gt",
}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    return raw


def _append_set_arg(cmd: list[str], preset: TrainPreset, item: str) -> None:
    if "=" not in item:
        raise ValueError(f"--set expects key=value, got '{item}'")
    key, raw = item.split("=", 1)
    key = key.strip().replace("-", "_")
    value = _parse_value(raw.strip())

    if isinstance(value, bool):
        if value:
            flag = BOOL_TRUE_FLAGS.get(key)
            if flag is not None:
                cmd.append(flag)
            elif key not in BOOL_FALSE_FLAGS:
                raise ValueError(f"No true-boolean flag is known for '{key}'")
            return
        flag = BOOL_FALSE_FLAGS.get(key)
        if flag is None:
            raise ValueError(f"No false-boolean flag is known for '{key}'")
        if flag == "--disable-gt" and not preset.supports_disable_gt:
            return
        cmd.append(flag)
        return

    flag = FIELD_FLAGS.get(key)
    if flag is None:
        known = ", ".join(sorted([*FIELD_FLAGS, *BOOL_TRUE_FLAGS, *BOOL_FALSE_FLAGS]))
        raise ValueError(f"Unknown config field '{key}'. Known fields: {known}")
    cmd.extend([flag, value])


def _resolved_arg(ref: str) -> str:
    return _display_path(resolve_checkpoint(ref))


def _build_train_command(args: argparse.Namespace) -> list[str]:
    preset = TRAIN_PRESETS[args.preset]
    if args.train_gt and args.disable_gt:
        raise ValueError("--train-gt and --disable-gt are mutually exclusive")
    run_name = args.name or args.preset.replace("-", "_")
    cmd = ["python", "-m", preset.module]
    cmd.extend(preset.args)
    cmd.extend(["--checkpoint-dir", f"checkpoints/{run_name}"])
    cmd.extend(["--log-file", f"logs/train_{run_name}.csv"])

    if args.games is not None:
        cmd.extend(["--games", str(args.games)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])
    if args.games_per_update is not None:
        cmd.extend(["--games-per-update", str(args.games_per_update)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
    if args.entropy_coef is not None:
        cmd.extend(["--entropy-coef", str(args.entropy_coef)])
    if args.gt_every is not None:
        cmd.extend(["--gt-every", str(args.gt_every)])
    if args.train_gt:
        cmd.append("--train-gt")
    if args.disable_gt and preset.supports_disable_gt:
        cmd.append("--disable-gt")
    if args.sync:
        if not preset.supports_sync:
            raise ValueError(f"Preset '{args.preset}' does not support --sync")
        cmd.append("--synchronous")
    if args.async_mode and not preset.supports_sync:
        raise ValueError(f"Preset '{args.preset}' does not have async/sync modes")
    if args.resume:
        cmd.extend(["--resume", _resolved_arg(args.resume)])
    if args.init_from:
        if not preset.supports_history_init:
            raise ValueError(f"Preset '{args.preset}' does not support --init-from")
        cmd.extend(["--init-from-embed", _resolved_arg(args.init_from)])
    for item in args.set:
        _append_set_arg(cmd, preset, item)
    return cmd


def cmd_aliases(_args: argparse.Namespace) -> int:
    for alias, target in sorted(MODEL_ALIASES.items()):
        path = REPO_ROOT / target
        status = "ok" if path.exists() else "missing"
        print(f"{alias:24s}  {target:55s}  {status}")
    return 0


def cmd_presets(_args: argparse.Namespace) -> int:
    for name, preset in TRAIN_PRESETS.items():
        print(f"{name:18s}  {preset.module:24s}  {preset.description}")
        print(f"{'':18s}  args: {shlex.join(list(preset.args))}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    roots = tuple(args.roots)
    for path in discover_checkpoints(roots, include_all=args.all):
        try:
            print(format_info(checkpoint_info(path, device=args.device)))
        except Exception as exc:
            print(f"{_display_path(path):55s}  error: {exc}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    for ref in args.refs:
        info = checkpoint_info(ref, device=args.device)
        print(format_info(info))
        if args.verbose:
            for key, value in sorted(info.config.items()):
                print(f"  {key}: {value}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from training.eval import evaluate

    print(f"\nEvaluating over {args.games} games:", flush=True)
    results = evaluate(
        args.policy_a,
        args.policy_b,
        pass_path_a=args.pass_a,
        pass_path_b=args.pass_b,
        n_games=args.games,
        batch=args.batch,
        device=torch.device(args.device),
        progress_every=args.progress_every,
    )
    print(f"\nResults ({results['games']} games):", flush=True)
    print(f"  A win rate: {results['win_rate_a']:.1%}  ({args.policy_a})", flush=True)
    print(f"  B win rate: {results['win_rate_b']:.1%}  ({args.policy_b})", flush=True)
    print(f"  Draws:      {results['draws']:.1%}", flush=True)
    print(f"  Mean score diff (A - B): {results['mean_score_diff']:+.1f}", flush=True)
    return 0


def cmd_train_command(args: argparse.Namespace) -> int:
    cmd = _build_train_command(args)
    print(shlex.join(cmd))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Tichu training experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    aliases = sub.add_parser("aliases", help="List stable checkpoint aliases")
    aliases.set_defaults(func=cmd_aliases)

    presets = sub.add_parser("presets", help="List training presets")
    presets.set_defaults(func=cmd_presets)

    list_p = sub.add_parser("list", help="List representative checkpoint files")
    list_p.add_argument("--roots", nargs="+", default=["checkpoints", "kagglecheckpoints"])
    list_p.add_argument("--all", action="store_true", help="List every *.pt file")
    list_p.add_argument("--device", default="cpu")
    list_p.set_defaults(func=cmd_list)

    inspect_p = sub.add_parser("inspect", help="Inspect checkpoint metadata")
    inspect_p.add_argument("refs", nargs="+")
    inspect_p.add_argument("--device", default="cpu")
    inspect_p.add_argument("--verbose", "-v", action="store_true")
    inspect_p.set_defaults(func=cmd_inspect)

    eval_p = sub.add_parser("eval", help="Evaluate two checkpoint refs")
    eval_p.add_argument("policy_a")
    eval_p.add_argument("policy_b")
    eval_p.add_argument("--pass-a", default=None)
    eval_p.add_argument("--pass-b", default=None)
    eval_p.add_argument("--games", type=int, default=500)
    eval_p.add_argument("--batch", type=int, default=64)
    eval_p.add_argument("--device", default="cpu")
    eval_p.add_argument("--progress-every", type=int, default=256)
    eval_p.set_defaults(func=cmd_eval)

    train_p = sub.add_parser("train-command", help="Print a reproducible train command")
    train_p.add_argument("preset", choices=sorted(TRAIN_PRESETS))
    train_p.add_argument("--name", default=None, help="Run name under checkpoints/ and logs/")
    train_p.add_argument("--games", type=int, default=None)
    train_p.add_argument("--device", default=None)
    train_p.add_argument("--workers", type=int, default=None)
    train_p.add_argument("--games-per-update", type=int, default=None)
    train_p.add_argument("--batch-size", type=int, default=None)
    train_p.add_argument("--lr", type=float, default=None)
    train_p.add_argument("--entropy-coef", type=float, default=None)
    train_p.add_argument("--gt-every", type=int, default=None)
    train_p.add_argument("--train-gt", action="store_true")
    train_p.add_argument("--disable-gt", action="store_true")
    mode = train_p.add_mutually_exclusive_group()
    mode.add_argument("--sync", action="store_true", help="Disable async PPO pipelining")
    mode.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Use async PPO pipelining when the trainer supports it",
    )
    train_p.add_argument("--resume", default=None, help="Checkpoint ref or path")
    train_p.add_argument("--init-from", default=None, help="Embed checkpoint ref or path")
    train_p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any supported TrainConfig CLI field",
    )
    train_p.set_defaults(func=cmd_train_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
