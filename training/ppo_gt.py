"""PPO update for the Grand Tichu network — realized-return signal.

The head trains on what its own decisions actually earned in self-play,
not on a value net's counterfactual (which measured +59 pts/round optimistic,
see training.gt_calibration). Buffer format (harvested during normal
self-play rollouts, rollout_history.py):

  gt_h8        (N, 56)  float32 — first-8-cards binary
  gt_declare   (N,)     int64   — the decision actually taken
  gt_log_probs (N,)     float32 — behavior log-prob (head + explore-eps mix)
  gt_returns   (N,)     float32 — realized normalized round return of the
                                  decision-maker's team, GT bonus included

advantage = return − b(h8), with the baseline b deliberately blind to the
decision: it absorbs hand quality, leaving the decision's causal
contribution. Declined hands are the built-in control group — they carry
the information of what strong hands earn when played quietly, which is
what keeps the baseline honest. Both branches stay sampled via
GT_EXPLORE_EPS in the rollouts, so a wrong "never call" (or "always call")
keeps being measured and self-corrects.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from training.config import TrainConfig


def ppo_update_gt(
    policy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """One PPO update on the Grand Tichu network."""

    h8 = torch.as_tensor(rollout["gt_h8"], device=device)
    declare = torch.as_tensor(rollout["gt_declare"], device=device)
    old_lp = torch.as_tensor(rollout["gt_log_probs"], device=device)
    rets = torch.as_tensor(rollout["gt_returns"], device=device)

    # Advantages frozen from the pre-update baseline (standard PPO practice)
    policy.eval()
    with torch.no_grad():
        advs = rets - policy.baseline(h8)
    if len(advs) > 1:
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    N = len(h8)
    idx = np.arange(N)

    stat_pl, stat_bl, stat_ent, stat_kl = [], [], [], []

    policy.train()
    for _ in range(cfg.n_epochs):
        np.random.shuffle(idx)
        for start in range(0, N, cfg.batch_size):
            b = idx[start : start + cfg.batch_size]

            new_lp, new_ent = policy.evaluate(h8[b], declare[b])

            ratio = torch.exp(new_lp - old_lp[b])
            b_advs = advs[b]

            pg_loss = torch.max(
                -b_advs * ratio,
                -b_advs * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps),
            ).mean()

            base_loss = ((policy.baseline(h8[b]) - rets[b]) ** 2).mean()
            ent_loss = new_ent.mean()

            loss = (
                pg_loss
                + cfg.value_coef * base_loss
                - cfg.entropy_coef * ent_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl = ((ratio - 1) - torch.log(ratio)).mean()

            stat_pl.append(pg_loss.item())
            stat_bl.append(base_loss.item())
            stat_ent.append(ent_loss.item())
            stat_kl.append(kl.item())

    return {
        "gt_policy_loss": float(np.mean(stat_pl)),
        "gt_baseline_loss": float(np.mean(stat_bl)),
        "gt_entropy": float(np.mean(stat_ent)),
        "gt_approx_kl": float(np.mean(stat_kl)),
    }


def gt_probe_update(
    policy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Regress the declare advantage on paired-probe data; policy is derived.

    Both branches of every probed deal were actually played, so the causal
    declare effect δ = return(declared) − return(quiet) is observed directly
    — no importance ratio, no clipping, and no baseline needed (the paired
    quiet branch IS the baseline, and it cancels the whole deal's luck, not
    just the b(h8)-explainable part). adv_head fits d(h8) ≈ E[δ|h8] by MSE
    and the calling policy is p = σ(d / GT_DERIVE_TEMP) (model_gt) — the
    same fixed point p = σ(E[δ]/entropy_coef) the previous policy-gradient
    objective (maximize δ·p + entropy bonus) converged to, minus its fatal
    flaw: the PG gradient carried a factor p(1−p), so once the head honestly
    saturated at "never call" it could barely learn which hands are the
    exceptions. The regression gradient is independent of where p sits, and
    the forced probe branches keep supplying data no matter how rarely the
    derived policy declares on-policy.

    b(h8) keeps training on the quiet-branch return so calibration probes
    stay meaningful.
    """
    h8 = torch.as_tensor(rollout["gt_probe_h8"], device=device)
    diff = torch.as_tensor(rollout["gt_probe_diff"], device=device)
    quiet = torch.as_tensor(rollout["gt_probe_quiet"], device=device)

    N = len(h8)
    idx = np.arange(N)
    stat_dl, stat_bl, stat_ent, stat_p = [], [], [], []

    policy.train()
    for _ in range(cfg.n_epochs):
        np.random.shuffle(idx)
        for start in range(0, N, cfg.batch_size):
            b = idx[start : start + cfg.batch_size]

            feat = policy.trunk(h8[b])
            d = policy.adv_head(feat).squeeze(-1)

            adv_loss = ((d - diff[b]) ** 2).mean()
            base_loss = ((policy.baseline_head(feat).squeeze(-1)
                          - quiet[b]) ** 2).mean()

            loss = adv_loss + cfg.value_coef * base_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                dist = torch.distributions.Bernoulli(
                    logits=policy.call_logits(h8[b]))
                stat_ent.append(dist.entropy().mean().item())
                stat_p.append(dist.probs.mean().item())

            stat_dl.append(adv_loss.item())
            stat_bl.append(base_loss.item())

    return {
        "gt_adv_loss": float(np.mean(stat_dl)),
        "gt_baseline_loss": float(np.mean(stat_bl)),
        "gt_entropy": float(np.mean(stat_ent)),
        "gt_mean_p": float(np.mean(stat_p)),
        "gt_diff_mean": float(diff.mean().item()),
        "gt_diff_std": float(diff.std().item()),
        "gt_diff_pos_frac": float((diff > 0).float().mean().item()),
    }
