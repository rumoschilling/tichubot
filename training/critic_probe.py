"""Critic-only self-play probe on a frozen action-embedding policy.

This compares value networks that read the raw static observation directly.
The source actor, passing policy, and Grand Tichu policy only generate data;
no optimizer owns any of their parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tichu_env.encoding import COUNTED_TRICK_POINTS_OBS_SIZE
from training.checkpoints import load_policy_checkpoint
from training.model_gt import TichuGrandTichuPolicy
from training.model_passing import TichuPassingPolicy
from training.rollout_embed import collect_rollouts_embed


CRITIC_SPECS = {
    "linear": {"activation": "linear", "raw_residual": False},
    "tanh": {"activation": "tanh", "raw_residual": False},
    "silu": {"activation": "silu", "raw_residual": False},
    "silu_residual": {"activation": "silu", "raw_residual": True},
    "tanh_hetero": {
        "activation": "tanh",
        "raw_residual": False,
        "heteroscedastic": True,
    },
    "tanh_hetero_anchor": {
        "activation": "tanh",
        "raw_residual": False,
        "heteroscedastic": True,
        "mse_anchor_alpha": 0.2,
    },
}


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class RawObservationCritic(nn.Module):
    """Value network that bypasses the actor's shared representation."""

    def __init__(
        self,
        obs_size: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        activation: str = "silu",
        raw_residual: bool = False,
        heteroscedastic: bool = False,
        mse_anchor_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.hidden_dim = hidden_dim
        self.n_layers = 0 if activation == "linear" else n_layers
        self.activation = activation
        self.raw_residual = raw_residual
        self.heteroscedastic = heteroscedastic
        if not 0.0 <= mse_anchor_alpha <= 1.0:
            raise ValueError("mse_anchor_alpha must be between 0 and 1")
        if mse_anchor_alpha and not heteroscedastic:
            raise ValueError("mse_anchor_alpha requires a heteroscedastic critic")
        self.mse_anchor_alpha = mse_anchor_alpha

        if activation == "linear":
            if raw_residual:
                raise ValueError("linear critic already is a raw linear path")
            self.trunk = nn.Identity()
            feature_dim = obs_size
        else:
            activation_cls = {"tanh": nn.Tanh, "silu": nn.SiLU}.get(activation)
            if activation_cls is None:
                raise ValueError(f"Unsupported activation: {activation}")
            layers: list[nn.Module] = []
            in_dim = obs_size
            for _ in range(n_layers):
                layers.extend((nn.Linear(in_dim, hidden_dim), activation_cls()))
                in_dim = hidden_dim
            self.trunk = nn.Sequential(*layers)
            feature_dim = in_dim

        self.value_head = nn.Linear(feature_dim, 1)
        self.log_variance_head = (
            nn.Linear(feature_dim, 1) if heteroscedastic else None
        )
        self.raw_skip = nn.Linear(obs_size, 1, bias=False) if raw_residual else None
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        if self.log_variance_head is not None:
            nn.init.zeros_(self.log_variance_head.weight)
            # softplus(0.5413) ~= 1, so its initial mean gradient matches MSE.
            nn.init.constant_(self.log_variance_head.bias, 0.5413249)
        if self.raw_skip is not None:
            nn.init.zeros_(self.raw_skip.weight)

    def distribution(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return value mean and optional log variance in normalized units."""
        features = self.trunk(obs)
        value = self.value_head(features).squeeze(-1)
        if self.raw_skip is not None:
            value = value + self.raw_skip(obs).squeeze(-1)
        if self.log_variance_head is None:
            return value, None
        variance = F.softplus(
            self.log_variance_head(features).squeeze(-1)
        ) + 1e-6
        return value, variance.log()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.distribution(obs)[0]


def critic_regression_loss(
    critic: RawObservationCritic,
    obs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """MSE, Gaussian NLL, or NLL with a floor on the mean gradient.

    The anchored surrogate has the same forward value and variance gradient as
    Gaussian NLL. Only the value-mean gradient changes: inverse-variance
    weighting cannot fall below ``mse_anchor_alpha``. Detaching the two paths
    prevents the mean and uncertainty heads from changing each other's target.
    """
    value, log_variance = critic.distribution(obs)
    squared_error = (value - targets) ** 2
    if log_variance is None:
        return squared_error.mean()
    inverse_variance = torch.exp(-log_variance)
    alpha = critic.mse_anchor_alpha
    if alpha == 0.0:
        return 0.5 * (
            squared_error * inverse_variance + log_variance
        ).mean()

    mean_weight = alpha + (1.0 - alpha) * inverse_variance.detach()
    mean_term = 0.5 * mean_weight * squared_error
    variance_term = 0.5 * (
        squared_error.detach() * inverse_variance + log_variance
    )
    # Forward value remains conventional NLL; gradients follow the separated
    # anchored mean and variance objectives above.
    return (variance_term + mean_term - mean_term.detach()).mean()


def model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(array.tobytes())
    return digest.hexdigest()


def train_critic_batch(
    critics: dict[str, RawObservationCritic],
    optimizers: dict[str, torch.optim.Optimizer],
    obs_np: np.ndarray,
    targets_np: np.ndarray | dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    n_epochs: int,
    max_grad_norm: float,
) -> dict[str, float]:
    """Fit every critic on identical shuffled minibatches."""
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    if isinstance(targets_np, dict):
        target_arrays = targets_np
    else:
        target_arrays = {name: targets_np for name in critics}
    if target_arrays.keys() != critics.keys():
        raise ValueError("critic target names must match critic names")
    targets = {
        name: torch.as_tensor(values, dtype=torch.float32, device=device)
        for name, values in target_arrays.items()
    }
    indices = np.arange(len(obs_np))
    losses = {name: [] for name in critics}

    for critic in critics.values():
        critic.train()
    for _ in range(n_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = torch.as_tensor(
                indices[start : start + batch_size],
                dtype=torch.int64,
                device=device,
            )
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            batch_losses = {
                name: critic_regression_loss(
                    critic, obs[batch], targets[name][batch]
                )
                for name, critic in critics.items()
            }
            torch.stack(tuple(batch_losses.values())).sum().backward()
            for name, critic in critics.items():
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                optimizers[name].step()
                losses[name].append(float(batch_losses[name].detach()))

    return {name: float(np.mean(values)) for name, values in losses.items()}


COHORTS = {
    "all": lambda data: np.ones(len(data["returns"]), dtype=np.bool_),
    "terminal": lambda data: data["steps_to_end"] == 1,
    "step_5": lambda data: data["steps_to_end"] == 5,
    "step_10": lambda data: data["steps_to_end"] == 10,
    "step_20": lambda data: data["steps_to_end"] == 20,
    # The 65th decision is the fixed-from-start counterpart to the validation
    # set's mean t-10 position (64.21 zero-based). It never uses future length.
    "start_65": lambda data: data["steps_from_start"] == 65,
    "forced_terminal": lambda data: (
        (data["steps_to_end"] == 1) & (data["legal_lens"] == 1)
    ),
    "forced_double": lambda data: (
        (data["steps_to_end"] == 1)
        & (data["legal_lens"] == 1)
        & data["double_win"]
    ),
}


def cohort_mask(data: dict[str, np.ndarray], cohort: str) -> np.ndarray:
    try:
        return COHORTS[cohort](data)
    except KeyError as exc:
        known = ", ".join(COHORTS)
        raise ValueError(f"Unknown critic cohort '{cohort}'; choose from {known}") from exc


def forced_double_score_oracle(obs: np.ndarray) -> np.ndarray:
    """Return the exact team-relative score for pre-action double-win states.

    In this cohort the observer is about to finish second, so their partner is
    necessarily first. The base team margin is +200; declarations by self
    fail, while declarations by the partner succeed and opponent declarations
    fail for the opposing team.
    """
    declarations = obs[:, 168:172] * 100.0 + obs[:, 172:176] * 200.0
    relative_margin = (
        200.0
        - declarations[:, 0]
        + declarations[:, 1]
        + declarations[:, 2]
        + declarations[:, 3]
    )
    return relative_margin / 400.0


@torch.no_grad()
def build_lambda_targets(
    critics: dict[str, RawObservationCritic],
    data: dict[str, np.ndarray],
    td_lambda: float,
    *,
    device: torch.device,
    batch_size: int = 16384,
) -> dict[str, np.ndarray]:
    """Build fitted TD(lambda) targets with team-relative perspective changes.

    There are no intermediate round rewards. Each episode is contiguous and
    ``steps_to_end == 1`` marks its terminal transition. Values at consecutive
    decisions use different observers, so opponent-team bootstraps are negated.
    """
    if not 0.0 <= td_lambda <= 1.0:
        raise ValueError("td_lambda must be between 0 and 1")

    obs = data["obs"]
    players = data["players"]
    steps_to_end = data["steps_to_end"]
    returns = data["returns"]
    nonterminal = steps_to_end != 1
    if np.any(steps_to_end[:-1][nonterminal[:-1]] - 1 != steps_to_end[1:][nonterminal[:-1]]):
        raise ValueError("rollout transitions are not contiguous by episode")

    perspective_sign = np.ones(len(obs) - 1, dtype=np.float32)
    perspective_sign[(players[:-1] - players[1:]) % 2 != 0] = -1.0
    result: dict[str, np.ndarray] = {}
    for name, critic in critics.items():
        critic.eval()
        prediction_chunks = []
        for start in range(0, len(obs), batch_size):
            obs_t = torch.as_tensor(
                obs[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            prediction_chunks.append(critic(obs_t).cpu().numpy())
        predictions = np.concatenate(prediction_chunks)

        targets = returns.copy()
        for index in range(len(targets) - 2, -1, -1):
            if nonterminal[index]:
                continuation = (
                    (1.0 - td_lambda) * predictions[index + 1]
                    + td_lambda * targets[index + 1]
                )
                targets[index] = perspective_sign[index] * continuation
        result[name] = targets
    return result


@torch.no_grad()
def evaluate_critics(
    critics: dict[str, RawObservationCritic],
    data: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 16384,
) -> dict[str, dict[str, dict[str, float]]]:
    predictions: dict[str, np.ndarray] = {"original": data["values"]}
    log_variances: dict[str, np.ndarray] = {}
    obs = data["obs"]
    for name, critic in critics.items():
        critic.eval()
        chunks = []
        log_variance_chunks = []
        for start in range(0, len(obs), batch_size):
            obs_t = torch.as_tensor(
                obs[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            values, log_variance = critic.distribution(obs_t)
            chunks.append(values.cpu().numpy())
            if log_variance is not None:
                log_variance_chunks.append(log_variance.cpu().numpy())
        predictions[name] = np.concatenate(chunks)
        if log_variance_chunks:
            log_variances[name] = np.concatenate(log_variance_chunks)

    report: dict[str, dict[str, dict[str, float]]] = {}
    for cohort, make_mask in COHORTS.items():
        mask = make_mask(data)
        report[cohort] = {}
        if not mask.any():
            continue
        target = data["returns"][mask] * 400.0
        cohort_predictions = dict(predictions)
        if cohort == "forced_double":
            cohort_predictions["score_oracle"] = forced_double_score_oracle(obs)
        for name, prediction in cohort_predictions.items():
            error = prediction[mask] * 400.0 - target
            report[cohort][name] = {
                "n": int(mask.sum()),
                "bias": float(error.mean()),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.mean(error * error))),
            }
            if name in log_variances:
                log_variance = log_variances[name][mask]
                normalized_error = error / 400.0
                sigma = np.exp(0.5 * log_variance)
                abs_error = np.abs(normalized_error)
                if len(error) > 1 and np.std(sigma) > 0.0:
                    uncertainty_error_corr = float(
                        np.corrcoef(sigma, abs_error)[0, 1]
                    )
                else:
                    uncertainty_error_corr = float("nan")
                report[cohort][name].update(
                    nll=float(
                        np.mean(
                            0.5
                            * (
                                normalized_error**2 * np.exp(-log_variance)
                                + log_variance
                                + np.log(2.0 * np.pi)
                            )
                        )
                    ),
                    sigma=float(sigma.mean() * 400.0),
                    coverage_1sigma=float(np.mean(abs_error <= sigma)),
                    uncertainty_error_corr=uncertainty_error_corr,
                )
    return report


def _critic_data(rollout: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = (
        "obs",
        "returns",
        "values",
        "steps_to_end",
        "steps_from_start",
        "legal_lens",
        "double_win",
        "players",
    )
    return {key: rollout[key] for key in keys}


def _load_auxiliary_policies(checkpoint, device: torch.device):
    cfg = checkpoint.get("config", {})
    pass_policy = None
    if "pass_model" in checkpoint:
        pass_policy = TichuPassingPolicy(
            hidden_dim=_cfg_value(cfg, "pass_hidden_dim", 128),
            n_layers=_cfg_value(cfg, "pass_n_layers", 2),
        ).to(device)
        pass_policy.load_state_dict(checkpoint["pass_model"])
        pass_policy.requires_grad_(False).eval()

    gt_policy = None
    if "gt_model" in checkpoint:
        gt_policy = TichuGrandTichuPolicy(
            hidden_dim=_cfg_value(cfg, "gt_hidden_dim", 64),
            n_layers=_cfg_value(cfg, "gt_n_layers", 2),
        ).to(device)
        gt_policy.load_state_dict(checkpoint["gt_model"])
        gt_policy.requires_grad_(False).eval()
    return pass_policy, gt_policy


def _atomic_save(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def _save_probe(
    path: Path,
    *,
    source_path: Path,
    source_fingerprint: str,
    games_trained: int,
    critics: dict[str, RawObservationCritic],
    optimizers: dict[str, torch.optim.Optimizer],
    critic_configs: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    report: dict[str, Any],
) -> None:
    _atomic_save(
        {
            "format": "raw_critic_probe_v1",
            "source_checkpoint": str(source_path),
            "source_fingerprint": source_fingerprint,
            "games_trained": games_trained,
            "critic_configs": critic_configs,
            "critics": {name: model.state_dict() for name, model in critics.items()},
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in optimizers.items()
            },
            "args": vars(args),
            "validation": report,
        },
        path,
    )


def _print_report(games: int, report: dict[str, Any]) -> None:
    print(f"\nValidation after {games:,} critic-training games (error in points)")
    header = f"{'cohort':<18} {'model':<15} {'n':>7} {'bias':>8} {'MAE':>8} {'RMSE':>8}"
    print(header)
    print("-" * len(header))
    for cohort, models in report.items():
        for model, stats in models.items():
            print(
                f"{cohort:<18} {model:<15} {stats['n']:7d} "
                f"{stats['bias']:+8.1f} {stats['mae']:8.1f} {stats['rmse']:8.1f}"
            )
            if "nll" in stats:
                print(
                    f"{'':<18} {'uncertainty':<15} {'':>7} "
                    f"sigma={stats['sigma']:6.1f}  "
                    f"cover={stats['coverage_1sigma'] * 100:5.1f}%  "
                    f"corr={stats['uncertainty_error_corr']:+.3f}  "
                    f"nll={stats['nll']:+.3f}"
                )


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    loaded = load_policy_checkpoint(args.checkpoint, device)
    if loaded.kind != "embed":
        raise ValueError("critic probe currently requires an embed/static checkpoint")
    policy = loaded.policy.requires_grad_(False).eval()
    if policy.obs_size != COUNTED_TRICK_POINTS_OBS_SIZE:
        raise ValueError(
            "expected counted-static observation size "
            f"{COUNTED_TRICK_POINTS_OBS_SIZE}, got {policy.obs_size}"
        )
    source_fingerprint = model_fingerprint(policy)
    pass_policy, gt_policy = _load_auxiliary_policies(loaded.checkpoint, device)
    cfg = loaded.checkpoint.get("config", {})

    critic_specs = {name: CRITIC_SPECS[name] for name in args.critics}
    critic_configs = {
        name: {
            "obs_size": policy.obs_size,
            "hidden_dim": args.hidden,
            "n_layers": args.layers,
            **spec,
        }
        for name, spec in critic_specs.items()
    }
    critics = {
        name: RawObservationCritic(**config).to(device)
        for name, config in critic_configs.items()
    }
    # When requested together, all Tanh critics start with identical trunks
    # and value means. Heteroscedastic variants also start with identical
    # variance heads, leaving only their regression objective different.
    if "tanh" in critics:
        ordinary_state = critics["tanh"].state_dict()
        for name in ("tanh_hetero", "tanh_hetero_anchor"):
            if name not in critics:
                continue
            variant_state = critics[name].state_dict()
            for key, value in ordinary_state.items():
                if key in variant_state and variant_state[key].shape == value.shape:
                    variant_state[key] = value.detach().clone()
            critics[name].load_state_dict(variant_state)
    if "tanh_hetero" in critics and "tanh_hetero_anchor" in critics:
        anchored_state = critics["tanh_hetero_anchor"].state_dict()
        for key, value in critics["tanh_hetero"].state_dict().items():
            if key in anchored_state and anchored_state[key].shape == value.shape:
                anchored_state[key] = value.detach().clone()
        critics["tanh_hetero_anchor"].load_state_dict(anchored_state)
    optimizers = {
        name: torch.optim.Adam(critic.parameters(), lr=args.lr)
        for name, critic in critics.items()
    }
    games_trained = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume["source_fingerprint"] != source_fingerprint:
            raise ValueError("resume probe was trained with a different source actor")
        if resume["critic_configs"] != critic_configs:
            raise ValueError("resume critic architecture does not match CLI arguments")
        resumed_cohort = resume.get("args", {}).get("train_cohort", "all")
        if resumed_cohort != args.train_cohort:
            raise ValueError(
                "resume training cohort does not match CLI arguments: "
                f"{resumed_cohort} != {args.train_cohort}"
            )
        resumed_lambda = resume.get("args", {}).get("td_lambda")
        if resumed_lambda != args.td_lambda:
            raise ValueError(
                "resume TD(lambda) does not match CLI arguments: "
                f"{resumed_lambda} != {args.td_lambda}"
            )
        for name in critics:
            critics[name].load_state_dict(resume["critics"][name])
            optimizers[name].load_state_dict(resume["optimizers"][name])
            for parameter_group in optimizers[name].param_groups:
                parameter_group["lr"] = args.lr
        games_trained = int(resume["games_trained"])
        print(f"Resumed critic probe at {games_trained:,} games")

    model_cfg = {
        "obs_size": policy.obs_size,
        "hidden_dim": _cfg_value(cfg, "hidden_dim", 256),
        "n_layers": _cfg_value(cfg, "n_layers", 2),
        "embed_dim": _cfg_value(cfg, "embed_dim", 64),
        "action_hidden_dim": _cfg_value(cfg, "action_hidden_dim", 128),
    }
    pass_model_cfg = {
        "hidden_dim": _cfg_value(cfg, "pass_hidden_dim", 128),
        "n_layers": _cfg_value(cfg, "pass_n_layers", 2),
    }
    gt_model_cfg = {
        "hidden_dim": _cfg_value(cfg, "gt_hidden_dim", 64),
        "n_layers": _cfg_value(cfg, "gt_n_layers", 2),
    }
    pool = None
    if args.workers > 1:
        if device.type != "cpu":
            raise ValueError("multiprocess critic rollouts currently require --device cpu")
        pool = mp.get_context("spawn").Pool(args.workers)

    def collect(n_games: int) -> dict[str, np.ndarray]:
        rollout = collect_rollouts_embed(
            policy,
            n_games,
            device,
            pool=pool,
            n_workers=args.workers,
            with_calling=gt_policy is not None,
            gt_policy=gt_policy,
            pass_policy=pass_policy,
            model_cfg=model_cfg,
            pass_model_cfg=pass_model_cfg,
            gt_model_cfg=gt_model_cfg,
            counted_trick_points=True,
        )
        return _critic_data(rollout)

    print(
        f"Frozen actor: {loaded.path} ({loaded.checkpoint.get('games_played', '?')} games)\n"
        f"Critics: {', '.join(critics)} | trainable parameters: "
        f"{sum(p.numel() for critic in critics.values() for p in critic.parameters()):,}\n"
        f"Training cohort: {args.train_cohort}\n"
        f"Value target: {'Monte Carlo' if args.td_lambda is None else f'TD(lambda={args.td_lambda:g})'}\n"
        f"Device: {device} | workers: {args.workers} | source sha256: "
        f"{source_fingerprint[:12]}"
    )
    print(f"Collecting {args.validation_games:,} fixed validation rounds...")
    validation = collect(args.validation_games)
    report = evaluate_critics(critics, validation, device)
    _print_report(games_trained, report)

    log_path = output_dir / "train.csv"
    log_exists = log_path.exists()
    log_file = log_path.open("a", newline="")
    writer = csv.DictWriter(
        log_file,
        fieldnames=["games", "transitions", "games_per_sec", *critics],
    )
    if not log_exists:
        writer.writeheader()

    started = time.monotonic()
    last_eval = games_trained
    try:
        while games_trained < args.games:
            cycle_start = time.monotonic()
            n_games = min(args.games_per_update, args.games - games_trained)
            train_data = collect(n_games)
            train_mask = cohort_mask(train_data, args.train_cohort)
            train_samples = int(train_mask.sum())
            if train_samples == 0:
                raise RuntimeError(
                    f"rollout contained no samples for cohort {args.train_cohort}"
                )
            if args.td_lambda is None:
                train_targets: np.ndarray | dict[str, np.ndarray] = train_data[
                    "returns"
                ][train_mask]
            else:
                lambda_targets = build_lambda_targets(
                    critics,
                    train_data,
                    args.td_lambda,
                    device=device,
                )
                train_targets = {
                    name: targets[train_mask]
                    for name, targets in lambda_targets.items()
                }
            losses = train_critic_batch(
                critics,
                optimizers,
                train_data["obs"][train_mask],
                train_targets,
                device=device,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
                max_grad_norm=args.max_grad_norm,
            )
            games_trained += n_games
            if model_fingerprint(policy) != source_fingerprint:
                raise RuntimeError("frozen actor parameters changed")
            elapsed = time.monotonic() - cycle_start
            gps = n_games / elapsed
            writer.writerow(
                {
                    "games": games_trained,
                    "transitions": len(train_data["obs"]),
                    "games_per_sec": f"{gps:.2f}",
                    **{name: f"{losses[name]:.7f}" for name in critics},
                }
            )
            log_file.flush()
            print(
                f"games={games_trained:>9,} transitions={len(train_data['obs']):>7,} "
                f"train={train_samples:>5,} "
                f"{gps:5.1f} g/s  "
                + "  ".join(f"{name}={losses[name]:.4f}" for name in critics),
                flush=True,
            )

            eval_due = games_trained - last_eval >= args.eval_every
            final = games_trained >= args.games
            if eval_due or final:
                report = evaluate_critics(critics, validation, device)
                _print_report(games_trained, report)
                last_eval = games_trained
            if eval_due or final or games_trained % args.checkpoint_every < n_games:
                _save_probe(
                    latest_path,
                    source_path=loaded.path,
                    source_fingerprint=source_fingerprint,
                    games_trained=games_trained,
                    critics=critics,
                    optimizers=optimizers,
                    critic_configs=critic_configs,
                    args=args,
                    report=report,
                )
            if args.max_hours is not None and (
                time.monotonic() - started >= args.max_hours * 3600
            ):
                print(f"Wall-clock limit reached after {games_trained:,} games")
                break
    except KeyboardInterrupt:
        print("\nInterrupted; saving critic probe.")
    finally:
        report = evaluate_critics(critics, validation, device)
        _save_probe(
            latest_path,
            source_path=loaded.path,
            source_fingerprint=source_fingerprint,
            games_trained=games_trained,
            critics=critics,
            optimizers=optimizers,
            critic_configs=critic_configs,
            args=args,
            report=report,
        )
        log_file.close()
        if pool is not None:
            pool.close()
            pool.join()
    print(f"Saved {latest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train raw-observation critics under a frozen static actor"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=100_000)
    parser.add_argument("--validation-games", type=int, default=2048)
    parser.add_argument("--games-per-update", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--td-lambda",
        type=float,
        help="Use fitted TD(lambda) targets instead of realized returns (0..1)",
    )
    parser.add_argument(
        "--train-cohort",
        choices=tuple(COHORTS),
        default="all",
        help="Which rollout states contribute critic gradients",
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--critics",
        nargs="+",
        choices=tuple(CRITIC_SPECS),
        default=["linear", "tanh", "silu", "silu_residual"],
        help="Critic variants to train (heteroscedastic is opt-in)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--resume")
    parser.add_argument(
        "--output-dir", default="checkpoints/critic_activation_probe"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
