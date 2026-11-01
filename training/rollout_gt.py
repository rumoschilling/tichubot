"""RETIRED from training (2026-07-14): the GT head now trains on realized
round returns harvested in rollout_history.py (see ppo_gt.py). The
counterfactual signal below measured +59 pts/round optimistic on the
declare decision (training.gt_calibration). This module is kept because
the calibration probe reuses `_get_batch_values`, and as a record of the
retired design.

For each deal, we compute V(hand, GT=1) and V(hand, GT=0) using the play
policy's value head.  The advantage of calling GT is:

    if declared:  adv = V(hand, GT=1) - V(hand, GT=0)
    if declined:  adv = V(hand, GT=0) - V(hand, GT=1)

This isolates the marginal value of the GT decision from hand quality noise.

The counterfactual values are evaluated on real post-passing round-start
states: full 14-card deal, GT phase decided by the head, a random passing
phase, real pass-source/target ids, empty history — all through the policy's
real feature path.  Earlier versions queried the value net on fabricated
observations (own hand bits only, every other feature zeroed), which is far
off-distribution; the resulting GT advantage was negative for ~99.8% of all
hands and collapsed the head to "never call" (v28/v29).
"""

from __future__ import annotations

import random

import numpy as np
import torch

from tichu_env.cards import NUM_CARDS
from tichu_env.encoding import (
    PLAY_TOKEN_SIZE,
    STATIC_OBS_SIZE,
    encode_pass_source_ids,
    encode_pass_target_ids,
    encode_static,
)
from tichu_env.game_state import GameState


@torch.no_grad()
def collect_gt_rollouts(
    gt_policy,
    value_policy,
    n_deals: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Collect Grand Tichu training data using counterfactual value baselines.

    For each player-hand, computes V(hand, GT=1) and V(hand, GT=0) to get
    a proper advantage signal that isolates the GT decision from hand quality.
    """
    gt_policy.eval()
    value_policy.eval()
    rng = random.Random()

    games = [GameState() for _ in range(n_deals)]
    for gs in games:
        gs.reset(with_calling=True, with_passing=True)

    # GT decisions on the first 8 cards (batched; the head only sees own hand)
    N = n_deals * 4
    hand_8_binaries = np.zeros((N, NUM_CARDS), dtype=np.float32)
    for i, gs in enumerate(games):
        for p in range(4):
            hand_8_binaries[i * 4 + p, list(gs.hands[p])] = 1.0
    declare_t, lp_t = gt_policy.act(
        torch.as_tensor(hand_8_binaries, device=device))
    declare_np = declare_t.cpu().numpy().astype(np.int64)

    # Apply declarations and run a random passing phase: a cheap stand-in for
    # the pass policy that still yields real, reachable round-start states.
    for i, gs in enumerate(games):
        for p in range(4):
            gs.declare_grand_tichu(p, bool(declare_np[i * 4 + p]))
        for p in range(4):
            left, partner, right = rng.sample(gs.get_passing_hand(p), 3)
            gs.pass_cards(p, left, partner, right)

    # Counterfactual observations: toggle only the observer's own GT bit,
    # keeping everyone else's actual declarations in place.
    obs_gt1 = np.zeros((N, STATIC_OBS_SIZE), dtype=np.float32)
    obs_gt0 = np.zeros((N, STATIC_OBS_SIZE), dtype=np.float32)
    ps = np.zeros((N, 3), dtype=np.int64)
    pt = np.zeros((N, 3), dtype=np.int64)
    for i, gs in enumerate(games):
        for p in range(4):
            j = i * 4 + p
            ps[j] = encode_pass_source_ids(gs, p)
            pt[j] = encode_pass_target_ids(gs, p)
            original = gs.tichu_declarations[p]
            gs.tichu_declarations[p] = 2
            encode_static(gs, p, out=obs_gt1[j])
            gs.tichu_declarations[p] = 0
            encode_static(gs, p, out=obs_gt0[j])
            gs.tichu_declarations[p] = original

    v_gt1 = _get_batch_values(value_policy, obs_gt1, ps, pt, device)
    v_gt0 = _get_batch_values(value_policy, obs_gt0, ps, pt, device)

    # advantage = V(chosen) - V(not chosen)
    advantages = np.where(declare_np == 1, v_gt1 - v_gt0, v_gt0 - v_gt1)

    return {
        "hand_8_binary": hand_8_binaries,
        "declare":       declare_np,
        "log_probs":     lp_t.cpu().numpy(),
        "advantages":    advantages.astype(np.float32),
    }


@torch.no_grad()
def _get_batch_values(
    policy,
    obs_np: np.ndarray,
    ps_np: np.ndarray,
    pt_np: np.ndarray,
    device: torch.device,
    chunk: int = 16384,
) -> np.ndarray:
    """Value estimates through the policy's real feature path (empty history)."""
    out = np.empty(len(obs_np), dtype=np.float32)
    is_history = hasattr(policy, "history_encoder")
    for s in range(0, len(obs_np), chunk):
        e = s + chunk
        obs = torch.as_tensor(obs_np[s:e], device=device)
        if is_history:
            B = obs.shape[0]
            fused = policy.value_features(
                obs,
                torch.zeros((B, 1, PLAY_TOKEN_SIZE), device=device),
                torch.zeros(B, dtype=torch.int32, device=device),
                torch.as_tensor(ps_np[s:e], device=device),
                torch.as_tensor(pt_np[s:e], device=device),
            )
        else:
            fused = policy.trunk(obs)
        out[s:e] = policy.value_head(fused).squeeze(-1).cpu().numpy()
    return out
