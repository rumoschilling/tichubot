"""Diagnose decision-conditioned history pooling on on-policy states.

The control query is the empirical mean of the checkpoint's dynamic query,
not zero. This keeps the ablation on the scale and center seen during play.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math

import numpy as np
import torch

from training.checkpoints import load_policy_checkpoint
from training.ppo_history import _rebuild_shared_history_numpy
from training.rollout_history import collect_rollouts_history


DISTANCE_BINS = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4-5", 4, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-40", 21, 40),
    ("41+", 41, math.inf),
)


def _query_means(policy, rollout, device, batch_size):
    actor_sum = None
    critic_sum = None
    count = 0
    with torch.inference_mode():
        for start in range(0, len(rollout["obs"]), batch_size):
            stop = min(start + batch_size, len(rollout["obs"]))
            obs = torch.as_tensor(
                rollout["obs"][start:stop], dtype=torch.float32, device=device
            )
            pass_source = torch.as_tensor(
                rollout["pass_source_ids"][start:stop],
                dtype=torch.long,
                device=device,
            )
            pass_target = torch.as_tensor(
                rollout["pass_target_ids"][start:stop],
                dtype=torch.long,
                device=device,
            )
            static = policy._static_features(obs, pass_source, pass_target)
            actor = policy.history_static_query(static).sum(dim=0)
            critic = policy.critic_history_static_query(static).sum(dim=0)
            actor_sum = actor if actor_sum is None else actor_sum + actor
            critic_sum = critic if critic_sum is None else critic_sum + critic
            count += len(obs)
    return actor_sum / count, critic_sum / count


def _attention_stats(policy, rollout, indices, actor_mean, critic_mean, device, batch_size):
    collected = {"actor": [], "critic": []}
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            history_np = _rebuild_shared_history_numpy(rollout, batch_indices)
            history = torch.as_tensor(history_np, dtype=torch.float32, device=device)
            lengths = torch.as_tensor(
                rollout["history_lens"][batch_indices],
                dtype=torch.long,
                device=device,
            )
            obs = torch.as_tensor(
                rollout["obs"][batch_indices], dtype=torch.float32, device=device
            )
            pass_source = torch.as_tensor(
                rollout["pass_source_ids"][batch_indices],
                dtype=torch.long,
                device=device,
            )
            pass_target = torch.as_tensor(
                rollout["pass_target_ids"][batch_indices],
                dtype=torch.long,
                device=device,
            )
            static = policy._static_features(obs, pass_source, pass_target)
            encoded, pad_mask, empty = policy.history_encoder.encode_tokens(
                history, lengths
            )

            branches = (
                (
                    "actor",
                    policy.history_encoder.pool_query,
                    policy.history_static_query(static),
                    actor_mean,
                    policy.history_encoder.pool_attn,
                ),
                (
                    "critic",
                    policy.critic_pool_query,
                    policy.critic_history_static_query(static),
                    critic_mean,
                    policy.critic_pool_attn,
                ),
            )
            for name, base, dynamic, mean, attention in branches:
                full_query = base.expand(len(obs), -1, -1) + dynamic.unsqueeze(1)
                mean_query = base.expand(len(obs), -1, -1) + mean.view(1, 1, -1)
                _, full_weights = attention(
                    full_query,
                    encoded,
                    encoded,
                    key_padding_mask=pad_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                _, mean_weights = attention(
                    mean_query,
                    encoded,
                    encoded,
                    key_padding_mask=pad_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                full_weights = full_weights[:, :, 0, :]
                mean_weights = mean_weights[:, :, 0, :]
                eps = torch.finfo(full_weights.dtype).tiny
                entropy = -(
                    full_weights * full_weights.clamp_min(eps).log()
                ).sum(dim=-1)
                mean_entropy = -(
                    mean_weights * mean_weights.clamp_min(eps).log()
                ).sum(dim=-1)
                kl = (
                    full_weights
                    * (
                        full_weights.clamp_min(eps).log()
                        - mean_weights.clamp_min(eps).log()
                    )
                ).sum(dim=-1)
                valid_normalized = lengths > 1
                normalizer = lengths.clamp_min(2).float().log().unsqueeze(1)
                collected[name].append(
                    {
                        "entropy": entropy.cpu().numpy(),
                        "mean_entropy": mean_entropy.cpu().numpy(),
                        "normalized": (entropy / normalizer)[valid_normalized]
                        .cpu()
                        .numpy(),
                        "max": full_weights.max(dim=-1).values.cpu().numpy(),
                        "kl": kl.cpu().numpy(),
                    }
                )

    print("\nAttention over non-empty on-policy histories")
    print("branch   entropy  norm_entropy  max_weight  KL(dynamic||mean)")
    for name in ("actor", "critic"):
        rows = collected[name]
        merged = {
            key: np.concatenate([row[key].reshape(-1) for row in rows])
            for key in rows[0]
        }
        print(
            f"{name:<7} {merged['entropy'].mean():8.3f} "
            f"{merged['normalized'].mean():13.3f} "
            f"{merged['max'].mean():11.3f} {merged['kl'].mean():18.3f}"
        )


def _steps_to_end(game_ids: np.ndarray) -> np.ndarray:
    remaining = np.zeros(len(game_ids), dtype=np.int32)
    seen = np.zeros(int(game_ids.max()) + 1, dtype=np.int32)
    for index in range(len(game_ids) - 1, -1, -1):
        game_id = game_ids[index]
        seen[game_id] += 1
        remaining[index] = seen[game_id]
    return remaining


def _critic_with_mean_query(policy, rollout, critic_mean, device, batch_size):
    predictions = np.empty(len(rollout["obs"]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(predictions), batch_size):
            stop = min(start + batch_size, len(predictions))
            indices = np.arange(start, stop)
            history_np = _rebuild_shared_history_numpy(rollout, indices)
            history = torch.as_tensor(history_np, dtype=torch.float32, device=device)
            lengths = torch.as_tensor(
                rollout["history_lens"][indices], dtype=torch.long, device=device
            )
            obs = torch.as_tensor(
                rollout["obs"][indices], dtype=torch.float32, device=device
            )
            pass_source = torch.as_tensor(
                rollout["pass_source_ids"][indices], dtype=torch.long, device=device
            )
            pass_target = torch.as_tensor(
                rollout["pass_target_ids"][indices], dtype=torch.long, device=device
            )
            static = policy._static_features(obs, pass_source, pass_target)
            encoded, pad_mask, empty = policy.history_encoder.encode_tokens(
                history, lengths
            )
            query = (
                policy.critic_pool_query.expand(len(obs), -1, -1)
                + critic_mean.view(1, 1, -1)
            )
            pooled = policy.history_encoder.pool_encoded(
                encoded,
                pad_mask,
                empty,
                query=query,
                attention=policy.critic_pool_attn,
            )
            delta = policy.critic_history_fuse(torch.cat([static, pooled], dim=-1))
            predictions[start:stop] = (
                policy.value_head(static + delta).squeeze(-1).cpu().numpy()
            )
    return predictions


def _error_stats(predictions, targets, mask):
    errors = 400.0 * (predictions[mask] - targets[mask])
    return len(errors), errors.mean(), np.abs(errors).mean(), np.sqrt(np.mean(errors**2))


def _print_critic_report(rollout, mean_predictions):
    full_predictions = rollout["values"]
    targets = rollout["returns"]
    steps = _steps_to_end(rollout["history_game_ids"])
    print("\nCritic error on the same completed self-play rounds (points)")
    print("steps  query       n     bias     MAE    RMSE")
    for label, low, high in DISTANCE_BINS:
        mask = (steps >= low) & (steps <= high)
        for query, predictions in (
            ("dynamic", full_predictions),
            ("mean", mean_predictions),
        ):
            n, bias, mae, rmse = _error_stats(predictions, targets, mask)
            print(f"{label:>5}  {query:<7} {n:7d} {bias:+8.1f} {mae:7.1f} {rmse:7.1f}")
    mask = np.ones(len(targets), dtype=bool)
    for query, predictions in (("dynamic", full_predictions), ("mean", mean_predictions)):
        n, bias, mae, rmse = _error_stats(predictions, targets, mask)
        print(f"{'all':>5}  {query:<7} {n:7d} {bias:+8.1f} {mae:7.1f} {rmse:7.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--max-attention-states", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    loaded = load_policy_checkpoint(args.checkpoint, device=device, strict=True)
    policy = loaded.policy
    if not getattr(policy, "static_attention", False):
        raise ValueError("checkpoint does not enable static attention")
    if not getattr(policy, "split_actor_critic_history", False):
        raise ValueError("diagnostic requires split actor/critic history pooling")

    with contextlib.redirect_stdout(io.StringIO()):
        rollout = collect_rollouts_history(policy, args.games, device)
    actor_mean, critic_mean = _query_means(
        policy, rollout, device, args.batch_size
    )
    print(f"checkpoint: {loaded.path}")
    print(f"games: {args.games:,}; decisions: {len(rollout['obs']):,}")
    print(
        "dynamic-query empirical-mean norm: "
        f"actor={actor_mean.norm().item():.3f}, critic={critic_mean.norm().item():.3f}"
    )

    nonempty = np.flatnonzero(rollout["history_lens"] > 0)
    rng = np.random.default_rng(args.seed)
    if len(nonempty) > args.max_attention_states:
        nonempty = np.sort(
            rng.choice(nonempty, args.max_attention_states, replace=False)
        )
    _attention_stats(
        policy,
        rollout,
        nonempty,
        actor_mean,
        critic_mean,
        device,
        args.batch_size,
    )
    mean_predictions = _critic_with_mean_query(
        policy, rollout, critic_mean, device, args.batch_size
    )
    _print_critic_report(rollout, mean_predictions)


if __name__ == "__main__":
    main()
