"""PPO update for the passing network — realized-return signal.

The pass net trains on what its directed passes actually earned in
self-play, not on a value-net proxy. The original own-view
V(post-pass) − V(pre-pass) proxy was exactly invariant to the
left/partner/right assignment — direction trained on pure noise, and the
harm of donating strength to opponents was invisible. Realized team
returns price both. (The bootstrap pipeline, rollout_passing.py, now uses
a team-signed all-hands V-delta instead — direction-aware, but main-loop
retired all the same: still a value-net proxy, the GT lesson.)

Buffer format (harvested from the main self-play stream, rollout_history.py):

  pass_obs     (N, 60)    float32  — hand binary + GT declaration bits
  hand_cards   (N, 14)    int64
  choices      (N, 3)     int64    — slot order: left, partner, right
  log_probs    (N, 3)     float32

plus exactly one of:

  returns      (N,)       float32  — realized normalized team round return of
                                     the passer; advantage = return − b(obs),
                                     and the baseline trains on these rows
  advantages   (N,)       float32  — pre-baselined causal signal (the gt-probe
                                     practice channel's control-differenced
                                     returns); used as-is, baseline untouched

Each sample has 3 sequential pointer steps. We sum log-probs across
steps to get the joint log-probability, then apply standard PPO clipping.
The return credits the directed triple as a block; per-slot credit
resolves statistically over many samples.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from training.config import TrainConfig


def ppo_update_passing(
    policy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """One PPO update on the passing network."""

    hand_binary = torch.as_tensor(rollout["pass_obs"], device=device)
    hand_cards = torch.as_tensor(rollout["hand_cards"], device=device)
    choices = torch.as_tensor(rollout["choices"], device=device)
    old_lp = torch.as_tensor(rollout["log_probs"], device=device)  # (N, 3)

    # Joint log-prob: sum over 3 steps
    old_lp_joint = old_lp.sum(dim=-1)  # (N,)

    train_baseline = "returns" in rollout
    if train_baseline:
        rets = torch.as_tensor(rollout["returns"], device=device)  # (N,)
        # Advantages frozen from the pre-update baseline (standard PPO practice)
        policy.eval()
        with torch.no_grad():
            advs = rets - policy.baseline(hand_binary)
    else:
        advs = torch.as_tensor(rollout["advantages"], device=device)
    if len(advs) > 1:
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    N = len(hand_binary)
    idx = np.arange(N)

    stat_pl, stat_bl, stat_ent, stat_kl = [], [], [], []
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp)

    policy.train()
    for _ in range(cfg.n_epochs):
        np.random.shuffle(idx)
        optimizer.zero_grad()

        for start in range(0, N, cfg.batch_size):
            b = idx[start : start + cfg.batch_size]

            with torch.amp.autocast('cuda', enabled=cfg.use_amp):
                new_lp, new_ent = policy.evaluate(
                    hand_binary[b], hand_cards[b], choices[b],
                )  # (B, 3), (B, 3)

                new_lp_joint = new_lp.sum(dim=-1)   # (B,)
                new_ent_mean = new_ent.mean(dim=-1)  # (B,)

                ratio = torch.exp(new_lp_joint - old_lp_joint[b])
                b_advs = advs[b]

                pg_loss = torch.max(
                    -b_advs * ratio,
                    -b_advs * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps),
                ).mean()

                ent_loss = new_ent_mean.mean()

                if train_baseline:
                    base_loss = ((policy.baseline(hand_binary[b]) - rets[b]) ** 2).mean()
                else:
                    base_loss = torch.zeros((), device=device)

                # Scale loss so the sum of gradients equals the average gradient
                scale = len(b) / N
                loss = (
                    pg_loss
                    + cfg.value_coef * base_loss
                    - cfg.entropy_coef * ent_loss
                ) * scale

            scaler.scale(loss).backward()

            with torch.no_grad():
                kl = ((ratio - 1) - torch.log(ratio)).mean()

            stat_pl.append(pg_loss.item())
            stat_bl.append(base_loss.item())
            stat_ent.append(ent_loss.item())
            stat_kl.append(kl.item())

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

    return {
        "pass_policy_loss": float(np.mean(stat_pl)),
        "pass_baseline_loss": float(np.mean(stat_bl)),
        "pass_entropy": float(np.mean(stat_ent)),
        "pass_approx_kl": float(np.mean(stat_kl)),
    }
