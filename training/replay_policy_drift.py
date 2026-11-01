"""Measure how quickly stored PPO states become stale across checkpoints.

The behavior checkpoint generates fixed self-play states and sampled actions.
Every later checkpoint is then evaluated on exactly those states and legal
actions.  This separates policy drift from deal/state-distribution noise and
reports both sampled-action importance ratios and full categorical-distribution
movement.
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from tichu_env.cards import card_name
from training.arena import load_arena_model
from training.checkpoints import load_policy_checkpoint, resolve_checkpoint
from training.paired_checkpoint_eval import _atomic_write_json
from training.ppo_history import _legal_offsets, _rebuild_shared_history_numpy
from training.rollout_history import collect_rollouts_history


PRIMARY_METRICS = (
    "ratio_absolute_log",
    "outside_clip_range",
    "ppo_objective_saturated",
    "kl_behavior_to_candidate",
    "jensen_shannon",
    "total_variation",
    "chosen_probability_absolute_change",
    "max_action_probability_absolute_change",
    "argmax_changed",
)


def _checkpoint_entry(spec: str, device: torch.device) -> dict:
    label, sep, ref = spec.partition("=")
    if not sep:
        ref = spec
        label = Path(spec).stem
    loaded = load_policy_checkpoint(ref, device)
    games = loaded.checkpoint.get("games_played")
    if loaded.kind != "history":
        raise ValueError(f"{label} is {loaded.kind}, expected a history policy")
    if not isinstance(games, int):
        raise ValueError(f"{label} does not record integer games_played")
    return {
        "label": label,
        "ref": ref,
        "path": str(loaded.path),
        "games_played": games,
        "policy": loaded.policy,
    }


@torch.inference_mode()
def _score_policy(
    policy: torch.nn.Module,
    rollout: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Return every legal-action probability plus per-state summaries."""
    legal_lens = rollout["legal_lens"].astype(np.int64, copy=False)
    offsets = _legal_offsets(legal_lens)
    n_rows = len(legal_lens)
    flat_probs = np.empty(int(legal_lens.sum()), dtype=np.float32)
    selected_log_probs = np.empty(n_rows, dtype=np.float32)
    entropy = np.empty(n_rows, dtype=np.float32)
    argmax = np.empty(n_rows, dtype=np.int32)
    policy.eval()

    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        rows = np.arange(start, stop, dtype=np.int64)
        lens_np = legal_lens[rows]
        max_actions = int(lens_np.max())
        legal_pad_np = np.zeros(
            (len(rows), max_actions, rollout["legal_flat"].shape[1]),
            dtype=np.float32,
        )
        for j, (row, length) in enumerate(zip(rows, lens_np)):
            offset = int(offsets[row])
            legal_pad_np[j, :length] = rollout["legal_flat"][
                offset : offset + length
            ]

        history_lens_np = rollout["history_lens"][rows].astype(
            np.int32, copy=False
        )
        max_history = max(int(history_lens_np.max()), 1)
        history_np = _rebuild_shared_history_numpy(
            rollout,
            rows,
            max_history,
        )
        obs = torch.as_tensor(
            rollout["obs"][rows], dtype=torch.float32, device=device
        )
        legal_pad = torch.as_tensor(legal_pad_np, device=device)
        history = torch.as_tensor(history_np, device=device)
        history_lens = torch.as_tensor(
            history_lens_np, dtype=torch.int32, device=device
        )
        pass_source = torch.as_tensor(
            rollout["pass_source_ids"][rows],
            dtype=torch.int64,
            device=device,
        )
        pass_target = torch.as_tensor(
            rollout["pass_target_ids"][rows],
            dtype=torch.int64,
            device=device,
        )
        logits, _ = policy._forward_scores(
            obs,
            legal_pad,
            history,
            history_lens,
            pass_source,
            pass_target,
        )
        valid = torch.arange(max_actions, device=device).unsqueeze(0) < (
            torch.as_tensor(lens_np, device=device).unsqueeze(1)
        )
        logits = logits.masked_fill(~valid, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = torch.as_tensor(
            rollout["action_idxs"][rows], dtype=torch.int64, device=device
        )

        selected_log_probs[start:stop] = (
            log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1).cpu().numpy()
        )
        entropy[start:stop] = (
            -(probs * log_probs).nan_to_num().sum(dim=1).cpu().numpy()
        )
        argmax[start:stop] = probs.argmax(dim=1).cpu().numpy()
        probs_np = probs.cpu().numpy()
        for j, (row, length) in enumerate(zip(rows, lens_np)):
            offset = int(offsets[row])
            flat_probs[offset : offset + length] = probs_np[j, :length]

    return {
        "flat_probs": flat_probs,
        "selected_log_probs": selected_log_probs,
        "entropy": entropy,
        "argmax": argmax,
    }


def _distribution_metrics(
    behavior: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    rollout: dict[str, np.ndarray],
    clip_eps: float,
) -> dict[str, np.ndarray]:
    """Compute sampled-action and exact full-distribution drift per state."""
    legal_lens = rollout["legal_lens"].astype(np.int64, copy=False)
    offsets = _legal_offsets(legal_lens)
    selected_offsets = offsets + rollout["action_idxs"].astype(
        np.int64, copy=False
    )
    p = behavior["flat_probs"].astype(np.float64, copy=False)
    q = candidate["flat_probs"].astype(np.float64, copy=False)
    tiny = np.finfo(np.float64).tiny
    p_safe = np.maximum(p, tiny)
    q_safe = np.maximum(q, tiny)
    midpoint = 0.5 * (p + q)
    midpoint_safe = np.maximum(midpoint, tiny)

    kl_pq = np.add.reduceat(p * (np.log(p_safe) - np.log(q_safe)), offsets)
    kl_qp = np.add.reduceat(q * (np.log(q_safe) - np.log(p_safe)), offsets)
    js = 0.5 * (
        np.add.reduceat(
            p * (np.log(p_safe) - np.log(midpoint_safe)), offsets
        )
        + np.add.reduceat(
            q * (np.log(q_safe) - np.log(midpoint_safe)), offsets
        )
    )
    absolute_delta = np.abs(q - p)
    total_variation = 0.5 * np.add.reduceat(absolute_delta, offsets)
    max_absolute_delta = np.maximum.reduceat(absolute_delta, offsets)

    behavior_log_prob = rollout["log_probs"].astype(np.float64, copy=False)
    candidate_log_prob = candidate["selected_log_probs"].astype(
        np.float64, copy=False
    )
    log_ratio = candidate_log_prob - behavior_log_prob
    ratio = np.exp(np.clip(log_ratio, -80.0, 80.0))
    normalized_advantage = rollout["advantages"].astype(np.float64, copy=True)
    if len(normalized_advantage) > 1:
        normalized_advantage = (
            normalized_advantage - normalized_advantage.mean()
        ) / (normalized_advantage.std(ddof=1) + 1e-8)
    low = 1.0 - clip_eps
    high = 1.0 + clip_eps
    below = ratio < low
    above = ratio > high
    saturated = ((normalized_advantage >= 0.0) & above) | (
        (normalized_advantage < 0.0) & below
    )

    return {
        "ratio": ratio,
        "log_ratio": log_ratio,
        "ratio_absolute_log": np.abs(log_ratio),
        "below_clip_range": below.astype(np.float64),
        "above_clip_range": above.astype(np.float64),
        "outside_clip_range": (below | above).astype(np.float64),
        "ppo_objective_saturated": saturated.astype(np.float64),
        "kl_behavior_to_candidate": np.maximum(kl_pq, 0.0),
        "kl_candidate_to_behavior": np.maximum(kl_qp, 0.0),
        "jensen_shannon": np.maximum(js, 0.0),
        "total_variation": total_variation,
        "chosen_probability_absolute_change": np.abs(
            q[selected_offsets] - p[selected_offsets]
        ),
        "chosen_probability_signed_change": (
            q[selected_offsets] - p[selected_offsets]
        ),
        "max_action_probability_absolute_change": max_absolute_delta,
        "argmax_changed": (
            behavior["argmax"] != candidate["argmax"]
        ).astype(np.float64),
        "behavior_entropy": behavior["entropy"].astype(np.float64),
        "candidate_entropy": candidate["entropy"].astype(np.float64),
        "entropy_change": (
            candidate["entropy"] - behavior["entropy"]
        ).astype(np.float64),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probs = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    result = np.quantile(values, probs)
    return {
        name: float(value)
        for name, value in zip(
            ("p01", "p05", "p25", "p50", "p75", "p95", "p99"),
            result,
        )
    }


def _block_bootstrap_intervals(
    metrics_by_age: list[dict[str, np.ndarray]],
    game_ids: np.ndarray,
    *,
    games_per_inference_block: int,
    samples: int,
    alpha: float,
    seed: int,
) -> dict[tuple[int, str], dict[str, object]]:
    """Block-bootstrap simultaneous mean intervals for the primary metrics."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    unique_games = np.unique(game_ids)
    if len(unique_games) % games_per_inference_block:
        raise ValueError("game count must divide into complete inference blocks")
    n_blocks = len(unique_games) // games_per_inference_block
    block_ids = game_ids // games_per_inference_block
    if not np.array_equal(np.unique(block_ids), np.arange(n_blocks)):
        raise ValueError("game ids must be contiguous from zero")

    columns: list[tuple[int, str]] = []
    block_sums: list[np.ndarray] = []
    block_counts = np.bincount(block_ids, minlength=n_blocks).astype(np.float64)
    for age_index, metrics in enumerate(metrics_by_age):
        for metric in PRIMARY_METRICS:
            columns.append((age_index, metric))
            block_sums.append(
                np.bincount(
                    block_ids,
                    weights=metrics[metric],
                    minlength=n_blocks,
                ).astype(np.float64)
            )
    sums = np.stack(block_sums, axis=1)
    point = sums.sum(axis=0) / block_counts.sum()
    draws = np.empty((samples, len(columns)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    chunk = 512
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = rng.integers(0, n_blocks, size=(stop - start, n_blocks))
        draws[start:stop] = sums[indices].sum(axis=1) / (
            block_counts[indices].sum(axis=1)[:, None]
        )

    ordinary_lo = np.quantile(draws, alpha / 2.0, axis=0)
    ordinary_hi = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    family_size = len(columns)
    simultaneous_lo = np.quantile(
        draws, alpha / (2.0 * family_size), axis=0
    )
    simultaneous_hi = np.quantile(
        draws, 1.0 - alpha / (2.0 * family_size), axis=0
    )
    return {
        column: {
            "mean": float(point[index]),
            "ordinary_ci": [
                float(ordinary_lo[index]),
                float(ordinary_hi[index]),
            ],
            "simultaneous_ci": [
                float(simultaneous_lo[index]),
                float(simultaneous_hi[index]),
            ],
        }
        for index, column in enumerate(columns)
    }


def _action_payload(vector: np.ndarray) -> dict[str, object]:
    cards = np.flatnonzero(vector[:56] > 0.5)
    return {
        "cards": [card_name(int(card)) for card in cards],
        "call_tichu": bool(vector[56] > 0.5),
        "phoenix_high": bool(vector[57] > 0.5),
    }


def _top_state_examples(
    rollout: dict[str, np.ndarray],
    behavior: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    game_ids: np.ndarray,
    steps_in_game: np.ndarray,
    count: int,
) -> list[dict[str, object]]:
    legal_lens = rollout["legal_lens"].astype(np.int64, copy=False)
    offsets = _legal_offsets(legal_lens)
    top_rows = np.argsort(metrics["total_variation"])[-count:][::-1]
    examples = []
    for row in top_rows:
        offset = int(offsets[row])
        length = int(legal_lens[row])
        p = behavior["flat_probs"][offset : offset + length]
        q = candidate["flat_probs"][offset : offset + length]
        vectors = rollout["legal_flat"][offset : offset + length]
        change_order = np.argsort(np.abs(q - p))[-min(10, length) :][::-1]
        action_changes = []
        for action_index in change_order:
            action_changes.append(
                {
                    "action_index": int(action_index),
                    "action": _action_payload(vectors[action_index]),
                    "behavior_probability": float(p[action_index]),
                    "candidate_probability": float(q[action_index]),
                    "signed_change": float(q[action_index] - p[action_index]),
                }
            )
        chosen = int(rollout["action_idxs"][row])
        examples.append(
            {
                "row": int(row),
                "game_id": int(game_ids[row]),
                "step_in_game": int(steps_in_game[row]),
                "legal_action_count": length,
                "sampled_action_index": chosen,
                "sampled_action": _action_payload(vectors[chosen]),
                "behavior_argmax_index": int(behavior["argmax"][row]),
                "candidate_argmax_index": int(candidate["argmax"][row]),
                "kl_behavior_to_candidate": float(
                    metrics["kl_behavior_to_candidate"][row]
                ),
                "jensen_shannon": float(metrics["jensen_shannon"][row]),
                "total_variation": float(metrics["total_variation"][row]),
                "largest_probability_changes": action_changes,
            }
        )
    return examples


def _steps_within_games(game_ids: np.ndarray) -> np.ndarray:
    steps = np.empty(len(game_ids), dtype=np.int32)
    starts = np.flatnonzero(
        np.r_[True, game_ids[1:] != game_ids[:-1]]
    )
    stops = np.r_[starts[1:], len(game_ids)]
    for start, stop in zip(starts, stops):
        steps[start:stop] = np.arange(stop - start, dtype=np.int32)
    return steps


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_experiment(args: argparse.Namespace) -> dict:
    if args.rollout_blocks <= 0 or args.games_per_rollout <= 0:
        raise ValueError("rollout counts must be positive")
    if args.games_per_rollout % args.games_per_inference_block:
        raise ValueError(
            "games-per-rollout must divide into complete inference blocks"
        )
    if not 0.0 < args.clip_eps < 1.0:
        raise ValueError("clip-eps must be between zero and one")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    behavior_spec = args.behavior_checkpoint
    behavior_label, sep, behavior_ref = behavior_spec.partition("=")
    if not sep:
        behavior_ref = behavior_spec
        behavior_label = Path(behavior_ref).stem
    behavior_arena = load_arena_model(
        f"{behavior_label}={behavior_ref}", device
    )
    behavior_path = resolve_checkpoint(behavior_ref)
    raw_checkpoint = torch.load(
        behavior_path, map_location="cpu", weights_only=False
    )
    behavior_games_played = raw_checkpoint.get("games_played")
    behavior_config = raw_checkpoint.get("config")
    games_per_update = int(getattr(behavior_config, "games_per_update", 1024))
    if not isinstance(behavior_games_played, int):
        raise ValueError("behavior checkpoint lacks integer games_played")

    checkpoints = [
        {
            "label": behavior_label,
            "ref": behavior_ref,
            "path": str(behavior_path),
            "games_played": behavior_games_played,
            "policy": behavior_arena.policy,
        }
    ]
    checkpoints.extend(
        _checkpoint_entry(spec, device) for spec in args.candidate_checkpoints
    )
    games = [entry["games_played"] for entry in checkpoints]
    if any(later <= earlier for earlier, later in zip(games, games[1:])):
        raise ValueError("candidate checkpoints must be in increasing game order")
    for entry in checkpoints:
        delta = entry["games_played"] - behavior_games_played
        if delta % games_per_update:
            raise ValueError(
                f"{entry['label']} is not an integer update age from behavior"
            )
        entry["age_updates"] = delta // games_per_update
        entry["age_games"] = delta

    output = Path(args.output)
    raw_output = Path(args.raw_output) if args.raw_output else output.with_suffix(".npz")
    public_checkpoints = [
        {key: value for key, value in entry.items() if key != "policy"}
        for entry in checkpoints
    ]
    result: dict[str, object] = {
        "design": {
            "behavior_checkpoint": behavior_ref,
            "fixed_behavior_states": True,
            "rollout_blocks": args.rollout_blocks,
            "games_per_rollout": args.games_per_rollout,
            "total_games": args.rollout_blocks * args.games_per_rollout,
            "games_per_inference_block": args.games_per_inference_block,
            "bootstrap_samples": args.bootstrap_samples,
            "family_wise_alpha": args.alpha,
            "simultaneous_ci_family": (
                "separate all-state and discretionary-state families; each "
                "contains all checkpoint ages times all primary drift metrics"
            ),
            "clip_epsilon": args.clip_eps,
            "same_states_and_legal_actions_for_every_checkpoint": True,
            "all_deployment_specialist_heads_enabled_during_collection": True,
            "seed": args.seed,
            "device": str(device),
            "batch_size": args.batch,
            "threads": args.threads,
        },
        "checkpoints": public_checkpoints,
        "ages": [],
        "complete": False,
    }
    _atomic_write_json(output, result)

    score_chunks: list[list[dict[str, np.ndarray]]] = [
        [] for _ in checkpoints
    ]
    rollout_chunks: list[dict[str, np.ndarray]] = []
    global_game_id_chunks: list[np.ndarray] = []
    started = time.perf_counter()
    for block_index in range(args.rollout_blocks):
        block_seed = args.seed + 100_000 * block_index
        torch.manual_seed(block_seed)
        print(
            f"Rollout block {block_index + 1}/{args.rollout_blocks}: "
            f"{args.games_per_rollout} games (seed {block_seed})",
            flush=True,
        )
        rollout = collect_rollouts_history(
            behavior_arena.policy,
            args.games_per_rollout,
            device,
            use_rollout_bucketing=True,
            with_calling=True,
            gt_policy=behavior_arena.gt_policy,
            pass_policy=behavior_arena.pass_policy,
            disable_gt_explore=False,
            seed=block_seed,
        )
        local_games = rollout["history_game_ids"].astype(np.int64, copy=False)
        global_games = local_games + block_index * args.games_per_rollout
        global_game_id_chunks.append(global_games)
        rollout_chunks.append(rollout)
        for checkpoint_index, entry in enumerate(checkpoints):
            checkpoint_started = time.perf_counter()
            score_chunks[checkpoint_index].append(
                _score_policy(entry["policy"], rollout, device, args.batch)
            )
            print(
                f"  scored {entry['label']} in "
                f"{time.perf_counter() - checkpoint_started:.1f}s",
                flush=True,
            )

    def concatenate_scores(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            key: np.concatenate([chunk[key] for chunk in chunks])
            for key in chunks[0]
        }

    scores = [concatenate_scores(chunks) for chunks in score_chunks]
    global_game_ids = np.concatenate(global_game_id_chunks)
    steps_in_game = _steps_within_games(global_game_ids)

    # Flatten rollout fields whose row or legal-action alignment is needed.
    row_keys = (
        "action_idxs",
        "log_probs",
        "advantages",
        "legal_lens",
    )
    pooled_rollout = {
        key: np.concatenate([chunk[key] for chunk in rollout_chunks])
        for key in row_keys
    }
    pooled_rollout["legal_flat"] = np.concatenate(
        [chunk["legal_flat"] for chunk in rollout_chunks]
    )

    validation_error = np.abs(
        scores[0]["selected_log_probs"] - pooled_rollout["log_probs"]
    )
    # Re-batching changes floating-point reduction order. Rare, extremely
    # peaked distributions can therefore move more in log space than the
    # typical row even though their probabilities remain effectively equal.
    # Guard both the distribution bulk and the worst row; the 0.02 maximum is
    # still an order of magnitude inside PPO's log clipping boundaries
    # (log(1.2)=0.182 and log(0.8)=-0.223).
    validation_p99 = float(np.quantile(validation_error, 0.99))
    validation_max = float(validation_error.max())
    result["behavior_rescore_validation"] = {
        "maximum_absolute_log_probability_error": validation_max,
        "p99_absolute_log_probability_error": validation_p99,
        "fraction_above_1e_4": float((validation_error > 1e-4).mean()),
        "fraction_above_1e_3": float((validation_error > 1e-3).mean()),
        "p99_tolerance": 1e-4,
        "maximum_tolerance": 0.02,
        "passed": bool(validation_p99 < 1e-4 and validation_max < 0.02),
    }
    if not result["behavior_rescore_validation"]["passed"]:
        _atomic_write_json(output, result)
        raise RuntimeError(
            "behavior policy rescore does not reproduce rollout log-probs; "
            f"maximum error={validation_max:.9g}, p99={validation_p99:.9g}"
        )

    metrics_by_age = [
        _distribution_metrics(
            scores[0], candidate, pooled_rollout, args.clip_eps
        )
        for candidate in scores[1:]
    ]
    intervals = _block_bootstrap_intervals(
        metrics_by_age,
        global_game_ids,
        games_per_inference_block=args.games_per_inference_block,
        samples=args.bootstrap_samples,
        alpha=args.alpha,
        seed=args.seed + 77,
    )
    discretionary_mask = pooled_rollout["legal_lens"] > 1
    discretionary_metrics_by_age = [
        {
            metric: values[discretionary_mask]
            for metric, values in metrics.items()
        }
        for metrics in metrics_by_age
    ]
    discretionary_intervals = _block_bootstrap_intervals(
        discretionary_metrics_by_age,
        global_game_ids[discretionary_mask],
        games_per_inference_block=args.games_per_inference_block,
        samples=args.bootstrap_samples,
        alpha=args.alpha,
        seed=args.seed + 78,
    )
    stratum_masks = {
        "forced_1": pooled_rollout["legal_lens"] == 1,
        "legal_2_to_5": (
            (pooled_rollout["legal_lens"] >= 2)
            & (pooled_rollout["legal_lens"] <= 5)
        ),
        "legal_6_to_20": (
            (pooled_rollout["legal_lens"] >= 6)
            & (pooled_rollout["legal_lens"] <= 20)
        ),
        "legal_21_to_50": (
            (pooled_rollout["legal_lens"] >= 21)
            & (pooled_rollout["legal_lens"] <= 50)
        ),
        "legal_51_plus": pooled_rollout["legal_lens"] >= 51,
    }
    result["state_population"] = {
        "total_decisions": int(len(discretionary_mask)),
        "discretionary_decisions": int(discretionary_mask.sum()),
        "discretionary_fraction": float(discretionary_mask.mean()),
        "forced_decisions": int((~discretionary_mask).sum()),
        "forced_fraction": float((~discretionary_mask).mean()),
        "legal_action_count_quantiles": _quantiles(
            pooled_rollout["legal_lens"]
        ),
        "stratum_counts": {
            label: int(mask.sum()) for label, mask in stratum_masks.items()
        },
    }

    for age_index, (entry, candidate, metrics) in enumerate(
        zip(checkpoints[1:], scores[1:], metrics_by_age)
    ):
        ratio = metrics["ratio"]
        discretionary_metrics = discretionary_metrics_by_age[age_index]
        discretionary_ratio = discretionary_metrics["ratio"]
        checkpoint_summary = {
            "label": entry["label"],
            "games_played": entry["games_played"],
            "age_games": entry["age_games"],
            "age_updates": entry["age_updates"],
            "decision_count": int(len(ratio)),
            "importance_ratio": {
                "mean": float(ratio.mean()),
                "std": float(ratio.std(ddof=1)),
                "quantiles": _quantiles(ratio),
                "below_clip_fraction": float(
                    metrics["below_clip_range"].mean()
                ),
                "above_clip_fraction": float(
                    metrics["above_clip_range"].mean()
                ),
                "effective_sample_size": float(
                    ratio.sum() ** 2 / np.square(ratio).sum()
                ),
                "effective_sample_size_fraction": float(
                    ratio.sum() ** 2 / (np.square(ratio).sum() * len(ratio))
                ),
            },
            "metric_estimates": {
                metric: {
                    **intervals[(age_index, metric)],
                    "quantiles": _quantiles(metrics[metric]),
                }
                for metric in PRIMARY_METRICS
            },
            "discretionary_states": {
                "decision_count": int(discretionary_mask.sum()),
                "importance_ratio": {
                    "mean": float(discretionary_ratio.mean()),
                    "std": float(discretionary_ratio.std(ddof=1)),
                    "quantiles": _quantiles(discretionary_ratio),
                    "below_clip_fraction": float(
                        discretionary_metrics["below_clip_range"].mean()
                    ),
                    "above_clip_fraction": float(
                        discretionary_metrics["above_clip_range"].mean()
                    ),
                    "effective_sample_size_fraction": float(
                        discretionary_ratio.sum() ** 2
                        / (
                            np.square(discretionary_ratio).sum()
                            * len(discretionary_ratio)
                        )
                    ),
                    "maximum": float(discretionary_ratio.max()),
                    "p999": float(np.quantile(discretionary_ratio, 0.999)),
                },
                "metric_estimates": {
                    metric: {
                        **discretionary_intervals[(age_index, metric)],
                        "quantiles": _quantiles(
                            discretionary_metrics[metric]
                        ),
                    }
                    for metric in PRIMARY_METRICS
                },
            },
            "legal_action_count_strata": {
                label: {
                    "decision_count": int(mask.sum()),
                    "metric_means": {
                        metric: float(metrics[metric][mask].mean())
                        for metric in PRIMARY_METRICS
                    },
                }
                for label, mask in stratum_masks.items()
            },
            "additional_metrics": {
                metric: {
                    "mean": float(metrics[metric].mean()),
                    "quantiles": _quantiles(metrics[metric]),
                }
                for metric in (
                    "kl_candidate_to_behavior",
                    "chosen_probability_signed_change",
                    "behavior_entropy",
                    "candidate_entropy",
                    "entropy_change",
                )
            },
            "most_diverged_states": _top_state_examples(
                pooled_rollout,
                scores[0],
                candidate,
                metrics,
                global_game_ids,
                steps_in_game,
                args.top_states,
            ),
        }
        result["ages"].append(checkpoint_summary)

    legal_offsets = _legal_offsets(pooled_rollout["legal_lens"])
    _atomic_save_npz(
        raw_output,
        {
            "checkpoint_games": np.asarray(games, dtype=np.int64),
            "checkpoint_age_updates": np.asarray(
                [entry["age_updates"] for entry in checkpoints], dtype=np.int64
            ),
            "game_ids": global_game_ids,
            "steps_in_game": steps_in_game,
            "legal_lens": pooled_rollout["legal_lens"],
            "legal_offsets": legal_offsets,
            "legal_action_vectors": pooled_rollout["legal_flat"].astype(
                np.uint8, copy=False
            ),
            "sampled_action_indices": pooled_rollout["action_idxs"],
            "behavior_log_probs": pooled_rollout["log_probs"],
            "advantages": pooled_rollout["advantages"],
            "policy_action_probabilities": np.stack(
                [score["flat_probs"] for score in scores]
            ),
            "policy_selected_log_probs": np.stack(
                [score["selected_log_probs"] for score in scores]
            ),
            "policy_entropy": np.stack(
                [score["entropy"] for score in scores]
            ),
            "policy_argmax": np.stack([score["argmax"] for score in scores]),
            **{
                f"age_{age_index}_{metric}": values
                for age_index, metrics in enumerate(metrics_by_age)
                for metric, values in metrics.items()
            },
        },
    )
    result["raw_output"] = str(raw_output)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["complete"] = True
    _atomic_write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-state PPO replay and policy-drift diagnostic"
    )
    parser.add_argument("behavior_checkpoint", help="NAME=checkpoint")
    parser.add_argument(
        "candidate_checkpoints", nargs="+", help="Ordered NAME=checkpoint entries"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--rollout-blocks", type=int, default=8)
    parser.add_argument("--games-per-rollout", type=int, default=256)
    parser.add_argument("--games-per-inference-block", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--top-states", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    result = run_experiment(args)
    print("\nFixed-state replay-drift result:", flush=True)
    for age in result["ages"]:
        metrics = age["metric_estimates"]
        print(
            f"  age {age['age_updates']:>3} updates: "
            f"clip={metrics['outside_clip_range']['mean']:.3f}, "
            f"PPO-saturated={metrics['ppo_objective_saturated']['mean']:.3f}, "
            f"TV={metrics['total_variation']['mean']:.3f}, "
            f"argmax-flip={metrics['argmax_changed']['mean']:.3f}, "
            f"ESS={age['importance_ratio']['effective_sample_size_fraction']:.3f}",
            flush=True,
        )
    print(f"Wrote {args.output} and {result['raw_output']}", flush=True)


if __name__ == "__main__":
    main()
