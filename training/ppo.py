"""PPO update step."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from training.config import TrainConfig


def ppo_update(
    policy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Perform standard PPO updates from a rollout buffer.
    
    This function implements the Proximal Policy Optimization (PPO) logic:
      1. Advantage Normalization: Reduces variance of gradient estimates.
      2. Clipped PG Loss: Ensures policy updates don't move too far from 
         the current behavior.
      3. Value Loss: Minimizes error between Critic's V and actual returns.
      4. Entropy Bonus: Encourages exploration by penalizing over-certainty.
    """
    from tichu_env.constants import MAX_ACTIONS

    obs      = torch.as_tensor(rollout["obs"],        device=device)
    actions  = torch.as_tensor(rollout["actions"],    device=device)
    old_lp   = torch.as_tensor(rollout["log_probs"],  device=device)
    returns  = torch.as_tensor(rollout["returns"],    device=device)
    advs     = torch.as_tensor(rollout["advantages"], device=device)

    # Reconstruct boolean mask from n_legal (avoids sending 2000-wide arrays over IPC)
    n_legal  = torch.as_tensor(rollout["n_legal"], dtype=torch.int32, device=device)
    idx      = torch.arange(MAX_ACTIONS, device=device)
    mask     = idx.unsqueeze(0) < n_legal.unsqueeze(1)  # (N, 2000)

    # Normalise advantages over the whole batch
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    N   = len(obs)
    idx = np.arange(N)

    stat_pl, stat_vl, stat_ent, stat_kl = [], [], [], []

    policy.train()
    for _ in range(cfg.n_epochs):
        np.random.shuffle(idx)
        for start in range(0, N, cfg.batch_size):
            b = idx[start : start + cfg.batch_size]

            log_prob, value, entropy = policy.evaluate(
                obs[b], mask[b], actions[b]
            )

            ratio    = torch.exp(log_prob - old_lp[b])
            b_advs   = advs[b]
            pg_loss  = torch.max(
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
