"""PPO update step for the action-embedding policy.

Expects the buffer format produced by rollout_embed.py:
  obs          (N, 196)   float32
  legal_flat   (L, 58)    float32
  legal_lens   (N,)       int32
  action_idxs  (N,)       int64
  log_probs    (N,)       float32
  values       (N,)       float32
  returns      (N,)       float32
  advantages   (N,)       float32
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from training.config import TrainConfig


def ppo_update_embed(
    policy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """One PPO update from an embed-format rollout buffer."""

    obs         = torch.as_tensor(rollout["obs"],         device=device)        # (N, 196)
    action_idxs = torch.as_tensor(rollout["action_idxs"], device=device)        # (N,)
    pass_source_ids = torch.as_tensor(
        rollout["pass_source_ids"], dtype=torch.int64, device=device
    )
    pass_target_ids = torch.as_tensor(
        rollout["pass_target_ids"], dtype=torch.int64, device=device
    )
    old_lp      = torch.as_tensor(rollout["log_probs"],   device=device)        # (N,)
    returns     = torch.as_tensor(rollout["returns"],     device=device)        # (N,)
    advs        = torch.as_tensor(rollout["advantages"],  device=device)        # (N,)

    legal_lens_np = rollout["legal_lens"]                                        # (N,) int32
    legal_flat_np = rollout["legal_flat"]                                        # (L, 58) float32

    # Normalise advantages
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    N   = len(obs)
    idx = np.arange(N)

    # Precompute per-step start offsets into legal_flat_np — done once per rollout.
    offsets = np.concatenate([[0], np.cumsum(legal_lens_np[:-1])])  # (N,) int64

    stat_pl, stat_vl, stat_ent, stat_kl = [], [], [], []

    policy.train()
    for _ in range(cfg.n_epochs):
        np.random.shuffle(idx)
        for start in range(0, N, cfg.batch_size):
            b = idx[start : start + cfg.batch_size]   # minibatch indices

            b_lens    = legal_lens_np[b]              # (B,) int32
            b_offsets = offsets[b]                    # (B,) int64
            max_n     = int(b_lens.max())

            # Vectorised gather: build (B, max_n) flat indices into legal_flat_np,
            # then index in one shot — no Python loops.
            col_range   = np.arange(max_n, dtype=np.int64)                 # (max_n,)
            flat_idx    = b_offsets[:, None] + col_range[None, :]          # (B, max_n)
            valid       = col_range[None, :] < b_lens[:, None]             # (B, max_n) bool
            flat_idx    = np.where(valid, flat_idx, 0)                     # clamp OOB to 0
            legal_pad   = legal_flat_np[flat_idx]                          # (B, max_n, 58)
            legal_pad  *= valid[:, :, None]                                # zero padding slots

            legal_pad_t = torch.as_tensor(legal_pad,  device=device)            # (B, max_n, 58)
            n_legal_t   = torch.as_tensor(b_lens,     dtype=torch.int32,
                                          device=device)                          # (B,)

            log_prob, value, entropy = policy.evaluate(
                obs[b], legal_pad_t, n_legal_t, action_idxs[b], pass_source_ids[b], pass_target_ids[b]
            )

            ratio   = torch.exp(log_prob - old_lp[b])
            b_advs  = advs[b]
            pg_loss = torch.max(
                -b_advs * ratio,
                -b_advs * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps),
            ).mean()

            v_loss   = ((value - returns[b]) ** 2).mean()
            ent_loss = entropy.mean()

            loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl = ((ratio - 1) - torch.log(ratio)).mean()

            stat_pl.append(pg_loss.item())
            stat_vl.append(v_loss.item())
            stat_ent.append(ent_loss.item())
            stat_kl.append(kl.item())

    return {
        "policy_loss": float(np.mean(stat_pl)),
        "value_loss":  float(np.mean(stat_vl)),
        "entropy":     float(np.mean(stat_ent)),
        "approx_kl":   float(np.mean(stat_kl)),
    }
