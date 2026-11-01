"""Rollout collection for the passing network — team-signed V-delta proxy.

RETIRED from the main training loop (2026-07-25): the pass net now trains
on realized team round returns harvested from the self-play rollouts
(rollout_history.py pass_train_* keys, ppo_passing.py). Kept for the
standalone bootstrap trainer (train_passing.py).

Upgraded the same day: the original reward was the passer's OWN
V(post)−V(pre), which is exactly invariant to the left/partner/right
assignment (direction trained on pure noise) and blind to strength donated
to opponents. Centralized training may peek at all four hands, so the
reward is now the team-signed sum of all four players' hand deltas:

    R = Σ_{q in team A} [V_q(post) − V_q(pre)] − Σ_{q in team B} [...]

credited +R to both team-A seats and −R to both team-B seats of the deal.
A card donated cross-team counts twice (the giver's loss and the
receiver's gain); under a purely additive V an intra-team pass cancels
exactly (team-neutral), so partner-routing signal comes from V's
nonlinearity — combination and support value.

For each deal we:
  1. Create a GameState with passing enabled, record pre-pass hands.
     When a gt_policy is given, the grand tichu phase runs first (with the
     same explore eps as self-play) so passing states carry real
     declarations — otherwise the declaration bits in the pass obs are
     always zero and the net can never learn to support a declarer.
  2. Run the passing network for all 4 players.
  3. Execute the pass via GameState.pass_cards().
  4. Use the main play policy's value head to estimate V(pre) and V(post)
     for every seat.
  5. Reward = team-signed sum of the four deltas, per seat as above.

Buffer format:
  pass_obs     (N, 60)    float32  — hand binary + GT declaration bits
  hand_cards   (N, 14)    int64    — sorted card indices
  choices      (N, 3)     int64    — indices into the 14-card hand
  log_probs    (N, 3)     float32
  rewards      (N,)       float32  — team-signed exchange score (±R per deal)
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import multiprocessing as mp

from tichu_env.cards import MAHJONG_INDEX, NUM_CARDS
from tichu_env.encoding import (
    PASS_OBS_SIZE,
    STATIC_OBS_SIZE,
    encode_pass_obs,
)
from tichu_env.game_state import GameState

from training.config import TrainConfig
from training.observations import encode_policy_static


class _StaticValuePolicy(torch.nn.Module):
    """Frozen worker copy of the exact static value function used for rewards.

    Passing rewards use only trunk and value_head, so workers need neither
    actor/history modules nor the unrelated passing trainer architecture.
    Copy the modules to preserve activations and normalization as well as weights.
    """

    def __init__(self, policy):
        super().__init__()
        self.trunk = copy.deepcopy(policy.trunk).cpu()
        self.value_head = copy.deepcopy(policy.value_head).cpu()
        self.obs_size = getattr(policy, "obs_size", STATIC_OBS_SIZE)
        self.requires_grad_(False)
        self.eval()


@torch.no_grad()
def collect_passing_rollouts(
    pass_policy,
    value_policy,
    n_deals: int,
    device: torch.device,
    n_workers: int = 1,
    pool: mp.Pool | None = None,
    cfg: TrainConfig | None = None,
    gt_policy=None,
) -> dict[str, np.ndarray]:
    """Collect training data for the passing network in parallel.

    The reward for each seat is the team-signed sum of all four players'
    V(post-pass) − V(pre-pass) hand deltas (see module docstring): +R for
    the two team-A seats of a deal, −R for team B.
    """
    if n_workers <= 1:
        return _collect_passing_batched(
            pass_policy, value_policy, n_deals, device, gt_policy=gt_policy)

    # 1. Prepare weights to send to workers
    pass_weights = {k: v.cpu().numpy() for k, v in pass_policy.state_dict().items()}
    worker_value_policy = _StaticValuePolicy(value_policy)
    gt_weights = None
    if gt_policy is not None:
        gt_weights = {k: v.cpu().numpy() for k, v in gt_policy.state_dict().items()}

    # 2. Split deals across workers
    base, rem = divmod(n_deals, n_workers)
    sizes = [base + (1 if i < rem else 0) for i in range(n_workers)]
    sizes = [size for size in sizes if size > 0]

    device_str = str(device)
    # Keep cfg in the public API for existing callers; the loaded policies
    # determine worker architecture, independently of passing trainer defaults.
    args = [(pass_weights, worker_value_policy, gt_weights, s, device_str)
            for s in sizes]
    
    # 3. Collect in parallel
    if pool is None:
        with mp.Pool(n_workers) as p:
            results = p.map(_worker_collect_passing, args)
    else:
        results = pool.map(_worker_collect_passing, args)
        
    # 4. Merge results
    return {k: np.concatenate([r[k] for r in results], axis=0) for k in results[0]}


def _worker_collect_passing(args):
    pass_weights, value_policy, gt_weights, n_deals, device_str = args
    device = torch.device(device_str)

    from training.model_passing import TichuPassingPolicy

    pass_policy = TichuPassingPolicy(
        hidden_dim=pass_weights["pointer_head.weight"].shape[1],
        n_layers=sum(
            key.startswith("trunk.") and key.endswith(".weight")
            for key in pass_weights
        ),
        card_embed_dim=pass_weights["card_embed.weight"].shape[1],
    ).to(device)
    pass_policy.load_state_dict({k: torch.as_tensor(v) for k, v in pass_weights.items()})

    gt_policy = None
    if gt_weights is not None:
        from training.model_gt import TichuGrandTichuPolicy
        gt_policy = TichuGrandTichuPolicy(
            hidden_dim=gt_weights["adv_head.weight"].shape[1],
            n_layers=sum(
                key.startswith("trunk.") and key.endswith(".weight")
                for key in gt_weights
            ),
        ).to(device)
        gt_policy.load_state_dict({k: torch.as_tensor(v) for k, v in gt_weights.items()})
        gt_policy.eval()

    value_policy = value_policy.to(device).eval()
    return _collect_passing_batched(pass_policy, value_policy, n_deals, device,
                                    gt_policy=gt_policy)


@torch.no_grad()
def _collect_passing_batched(
    pass_policy,
    value_policy,
    n_deals: int,
    device: torch.device,
    gt_policy=None,
) -> dict[str, np.ndarray]:
    """Internal batched helper."""
    pass_policy.eval()
    value_policy.eval()

    # 1. Initialize all games (grand tichu phase first when a head is given)
    with_calling = gt_policy is not None
    games = [GameState() for _ in range(n_deals)]
    for gs in games:
        gs.reset(with_calling=with_calling, with_passing=True)

    if with_calling:
        # Resolve the GT phase like self-play does (incl. explore eps), so
        # passing states carry the same declaration distribution the pass
        # net will face in rollouts. Decisions are independent per hand, so
        # one batched forward covers all of them.
        from training.rollout_history import GT_EXPLORE_EPS
        rng = random.Random()
        h8 = np.zeros((n_deals * 4, NUM_CARDS), dtype=np.float32)
        for i, gs in enumerate(games):
            for p in range(4):
                h8[i * 4 + p, list(gs.hands[p])] = 1.0
        declare_t, _ = gt_policy.act(torch.as_tensor(h8, device=device))
        declare_np = declare_t.cpu().numpy()
        for i, gs in enumerate(games):
            for p in range(4):
                declare = bool(declare_np[i * 4 + p])
                if rng.random() < GT_EXPLORE_EPS:
                    declare = rng.random() < 0.5
                gs.declare_grand_tichu(p, declare)

    # 2. Extract pre-pass hands and pass observations for all players
    N = n_deals * 4
    all_pre_hands = []
    for gs in games:
        for p in range(4):
            all_pre_hands.append(gs.get_passing_hand(p))

    pass_obs = np.zeros((N, PASS_OBS_SIZE), dtype=np.float32)  # (N, 60)
    for i, gs in enumerate(games):
        for p in range(4):
            encode_pass_obs(gs, p, out=pass_obs[i * 4 + p])
    hand_cards = np.array(all_pre_hands, dtype=np.int64)  # (N, 14)

    pass_obs_t = torch.as_tensor(pass_obs, device=device)
    hand_cards_t = torch.as_tensor(hand_cards, device=device)

    # 3. Batched passing decisions
    choices_t, lp_t = pass_policy.act(pass_obs_t, hand_cards_t)  # (N, 3), (N, 3)
    choices_np = choices_t.cpu().numpy()

    # 4. Evaluate V(pre)
    obs_size = getattr(value_policy, "obs_size", STATIC_OBS_SIZE)
    pre_obs = np.zeros((N, obs_size), dtype=np.float32)
    for i, gs in enumerate(games):
        gs_pre = GameState()
        gs_pre.reset()
        gs_pre.tichu_declarations = list(gs.tichu_declarations)
        for p in range(4):
            gs_pre.hands[p] = set(all_pre_hands[i*4 + p])
            if MAHJONG_INDEX in gs_pre.hands[p]:
                gs_pre.current_player = p

        for p in range(4):
            pre_obs[i*4 + p] = encode_policy_static(value_policy, gs_pre, p)

    pre_obs_t = torch.as_tensor(pre_obs, device=device)
    
    # ── Minibatched V(pre) evaluation (Optimized: Skip History) ──────
    v_pre = torch.zeros(N, device=device)
    eval_batch = 4096
    for start in range(0, N, eval_batch):
        end = min(start + eval_batch, N)
        v_pre[start:end] = _get_values(value_policy, pre_obs_t[start:end], device)

    # 5. Execute passes and evaluate V(post)
    post_obs = np.zeros((N, obs_size), dtype=np.float32)
    for i, gs in enumerate(games):
        for p in range(4):
            idx = i * 4 + p
            hand = all_pre_hands[idx]
            c = choices_np[idx]
            gs.pass_cards(p, hand[c[0]], hand[c[1]], hand[c[2]])
        
        for p in range(4):
            post_obs[i*4 + p] = encode_policy_static(value_policy, gs, p)

    post_obs_t = torch.as_tensor(post_obs, device=device)
    
    # ── Minibatched V(post) evaluation ───────────────────────────────
    v_post = torch.zeros(N, device=device)
    for start in range(0, N, eval_batch):
        end = min(start + eval_batch, N)
        v_post[start:end] = _get_values(value_policy, post_obs_t[start:end], device)

    # Team-signed exchange score: one number per deal, +R for the two
    # team-A seats, -R for team B. Peeking at all four deltas is legal
    # here (centralized training); it is what makes the reward sensitive
    # to WHERE the cards went, not just what the passer kept.
    deltas = (v_post - v_pre).cpu().numpy().reshape(n_deals, 4)
    team_r = deltas[:, 0] + deltas[:, 2] - deltas[:, 1] - deltas[:, 3]
    signs = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    rewards = (team_r[:, None] * signs[None, :]).reshape(-1).astype(np.float32)

    return {
        "pass_obs": pass_obs,
        "hand_cards": hand_cards,
        "choices": choices_np,
        "log_probs": lp_t.cpu().numpy(),
        "rewards": rewards,
    }


@torch.no_grad()
def _get_values(
    policy,
    obs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Extract value estimates using only the static trunk (ignores history for T=0 rewards)."""
    # Since history is empty at T=0, we only need the static trunk + value head.
    # The history bias would be constant and cancel out in V(post)-V(pre) anyway.
    feat = policy.trunk(obs)
    value = policy.value_head(feat).squeeze(-1)
    return value
