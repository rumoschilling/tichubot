"""Rollout collection for the action-embedding policy.

Buffer format (differs from rollout.py):
  obs          (N, obs_size) float32 (196 base, 200 with counted trick points)
  pass_source_ids (N, 3)  int64
  pass_target_ids (N, 3)  int64
  legal_flat   (L, 58)    float32  — all legal action vectors, concatenated
  legal_lens   (N,)       int32    — n_legal per step (L = sum of legal_lens)
  action_idxs  (N,)       int64    — chosen action index within the legal list
  log_probs    (N,)       float32
  values       (N,)       float32
  returns      (N,)       float32
  advantages   (N,)       float32
  players      (N,)       int8     — absolute acting seat, used for perspective-correct bootstrapping
  steps_from_start (N,)   int16    — 1-based decision index within the round

With a pass_policy, each game also emits one passing row per player:
  pass_train_obs        (P, 60)  float32
  pass_train_hand_cards (P, 14)  int64
  pass_train_choices    (P, 3)   int64
  pass_train_log_probs  (P, 3)   float32
  pass_train_returns    (P,)     float32

The key difference from rollout.py: instead of storing n_legal (a single int) and
reconstructing a positional range-mask, we store the actual 58-dim action vectors
(56 card bits + 1 tichu-call bit + 1 Phoenix-choice bit) of every legal action.
This lets ppo_embed.py
recompute logits without knowing the game state.

Public API
----------
collect_rollouts_embed(policy, n_games, device, pool=None, n_workers=1)
"""

from __future__ import annotations

from functools import lru_cache
import os
import random

import numpy as np
import torch

from tichu_env.cards import MAHJONG_INDEX
from tichu_env.encoding import (
    PASS_OBS_SIZE,
    encode_pass_obs,
    encode_pass_source_ids,
    encode_pass_target_ids,
    encode_static,
    encode_static_counted_trick_points,
)
from tichu_env.game_state import GameState, TEAM_A

from training.model_embed import ACTION_DIM


_USE_ACTION_VEC_CACHE = True


@lru_cache(maxsize=16384)
def _action_vec_cached(
    cards: tuple[int, ...], call_tichu: bool, phoenix_high: bool
) -> np.ndarray:
    vec = np.zeros(ACTION_DIM, dtype=np.uint8)
    for c in cards:
        vec[c] = 1
    if call_tichu:
        vec[56] = 1
    if phoenix_high:
        vec[57] = 1
    vec.flags.writeable = False
    return vec


def _action_vec(
    action: tuple[list[int], bool] | tuple[list[int], bool, bool]
) -> np.ndarray:
    """Convert a play action to a 58-dimensional uint8 vector.

    Bits 0-55 are card indicators, bit 56 is the Tichu-call flag, and
    bit 57 selects the higher of two Phoenix straight/full-house meanings.
    """
    cards, call_tichu = action[:2]
    phoenix_high = bool(action[2]) if len(action) == 3 else False
    if not _USE_ACTION_VEC_CACHE:
        vec = np.zeros(ACTION_DIM, dtype=np.uint8)
        for c in cards:
            vec[c] = 1
        if call_tichu:
            vec[56] = 1
        if phoenix_high:
            vec[57] = 1
        return vec
    key = (tuple(sorted(cards)), call_tichu, phoenix_high)
    # Cached arrays are immutable by convention. Rollout callers either copy
    # them into a batch or use them read-only.
    return _action_vec_cached(*key)


def reset_action_vec_cache_stats():
    _action_vec_cached.cache_clear()


def set_action_vec_cache_enabled(enabled: bool):
    global _USE_ACTION_VEC_CACHE
    _USE_ACTION_VEC_CACHE = enabled
    _action_vec_cached.cache_clear()


def get_action_vec_cache_stats() -> dict[str, int | float]:
    info = _action_vec_cached.cache_info()
    calls = info.hits + info.misses
    return {
        "calls": calls,
        "hits": info.hits,
        "misses": info.misses,
        "unique_actions": info.currsize,
        "hit_rate": (info.hits / calls) if calls else 0.0,
    }


def _hand_to_binary(hand: list[int]) -> np.ndarray:
    """Convert card index list to 56-dim binary vector."""
    from tichu_env.cards import NUM_CARDS
    vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for card in hand:
        vec[card] = 1.0
    return vec


# ── Level-1: batched single-process collection ────────────────────────────────

def _collect_batched(
    policy,
    n_games: int,
    device: torch.device,
    with_calling: bool = False,
    gt_policy=None,
    pass_policy=None,
    counted_trick_points: bool = False,
) -> dict[str, np.ndarray]:
    """Run n_games in lock-step, one batched forward pass per tick."""

    rng = random.Random()

    games: list[GameState] = [GameState() for _ in range(n_games)]
    for gs in games:
        gs.reset(
            with_calling=with_calling,
            with_passing=pass_policy is not None,
        )

    done = [False] * n_games

    ep_steps: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float, int]]] = [
        [] for _ in range(n_games)
    ]
    # (pass_obs, hand_cards, choices, log_probs, player)
    pass_steps: list[list[tuple]] = [[] for _ in range(n_games)]

    policy.eval()
    if pass_policy is not None:
        pass_policy.eval()

    while not all(done):
        # Phase 0: resolve grand-tichu (pre-play)
        for i, gs in enumerate(games):
            if done[i]:
                continue
            
            # Resolve Grand Tichu
            while gs.phase.name == "GRAND_TICHU":
                p = gs.current_player
                if gt_policy is not None:
                    # Provide first 8 cards
                    h8 = sorted(list(gs.hands[p]))[:8]
                    h8_bin = torch.as_tensor(_hand_to_binary(h8)[None, :], device=device)
                    decl, _ = gt_policy.act(h8_bin)
                    gs.declare_grand_tichu(p, bool(decl.item()))
                else:
                    gs.declare_grand_tichu(p, False) # Fallback to no GT

        # Phase 1: resolve passing. Each game advances through four passing
        # seats, so batch the currently active seats until every game reaches
        # normal play.
        while True:
            pass_active = [
                (i, gs, gs.current_player)
                for i, gs in enumerate(games)
                if not done[i] and gs.phase.name == "PASSING"
            ]
            if not pass_active:
                break

            hand_cards_np = np.array(
                [
                    gs.get_passing_hand(player)
                    for _, gs, player in pass_active
                ],
                dtype=np.int64,
            )
            pass_obs_np = np.zeros(
                (len(pass_active), PASS_OBS_SIZE), dtype=np.float32
            )
            for j, (_, gs, player) in enumerate(pass_active):
                encode_pass_obs(gs, player, out=pass_obs_np[j])

            pass_obs_t = torch.as_tensor(
                pass_obs_np, dtype=torch.float32, device=device
            )
            hand_cards_t = torch.as_tensor(
                hand_cards_np, dtype=torch.int64, device=device
            )
            choices_t, pass_lp_t = pass_policy.act(
                pass_obs_t, hand_cards_t
            )
            choices_np = choices_t.cpu().numpy()
            pass_lp_np = pass_lp_t.cpu().numpy()

            for j, (game_idx, gs, player) in enumerate(pass_active):
                choices = choices_np[j]
                cards = hand_cards_np[j, choices]
                pass_steps[game_idx].append(
                    (
                        pass_obs_np[j],
                        hand_cards_np[j],
                        choices,
                        pass_lp_np[j],
                        player,
                    )
                )
                if not gs.pass_cards(
                    player,
                    int(cards[0]),
                    int(cards[1]),
                    int(cards[2]),
                ):
                    raise RuntimeError(
                        f"Passing phase rejected sampled choices for P{player}"
                    )

        # Phase 2: resolve dragon-give (now with policy)
        for i, gs in enumerate(games):
            if done[i]:
                continue
            while gs.dragon_give_pending is not None and not gs.round_over:
                # Use policy to decide who gets the dragon
                with torch.no_grad():
                    player = gs.dragon_give_pending
                    encoder = (
                        encode_static_counted_trick_points
                        if counted_trick_points
                        else encode_static
                    )
                    obs = torch.as_tensor(
                        encoder(gs, player)[None, :], device=device
                    )
                    ps = torch.as_tensor(encode_pass_source_ids(gs, player)[None, :], device=device)
                    pt = torch.as_tensor(encode_pass_target_ids(gs, player)[None, :], device=device)
                    feat = policy.trunk(obs)
                    # Add pass source info
                    pf = policy.pass_source_embed(ps).reshape(1, -1)
                    feat = feat + policy.pass_source_proj(pf)
                    pf = policy.pass_target_embed(pt).reshape(1, -1)
                    feat = feat + policy.pass_target_proj(pf)
                    
                    logits = policy.get_dragon_gift_logits(feat) # (1, 2)
                    choice = torch.argmax(logits, dim=-1).item() # 0 or 1
                    
                    # Logits are for relative players 1 (next) and 3 (prev)
                    # Mapping: choice 0 -> player 1, choice 1 -> player 3
                    options = gs.get_dragon_give_options(player)
                    # Find which option corresponds to the relative choice
                    # Relative player 1 is (player+1)%4, 3 is (player+3)%4
                    target_abs = (player + (1 if choice == 0 else 3)) % 4
                    if target_abs in options:
                        gs.give_dragon_trick(player, target_abs)
                    else:
                        # Fallback if policy is weird
                        gs.give_dragon_trick(player, rng.choice(options))

            if gs.round_over:
                done[i] = True


        if all(done):
            break

        # Phase 3: collect observations
        # Each entry: (game_idx, player, obs_arr, legal_vecs_arr, legal_list)
        active: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]] = []
        for i, gs in enumerate(games):
            if done[i]:
                continue
            player = gs.current_player
            legal  = gs.get_policy_legal_actions(player)
            encoder = (
                encode_static_counted_trick_points
                if counted_trick_points
                else encode_static
            )
            obs = encoder(gs, player)
            pass_src = encode_pass_source_ids(gs, player)
            pass_tgt = encode_pass_target_ids(gs, player)
            vecs = np.array([_action_vec(a) for a in legal], dtype=np.float32)
            active.append((i, player, obs, pass_src, pass_tgt, vecs, legal))

        if not active:
            break

        # Phase 4: batched forward pass
        # Pad action vecs to (B, max_n, 58)
        B     = len(active)
        n_arr = np.array([x[5].shape[0] for x in active], dtype=np.int32)
        max_n = int(n_arr.max())

        legal_pad = np.zeros((B, max_n, ACTION_DIM), dtype=np.float32)
        for j, (_, _, _, _, _, vecs, _) in enumerate(active):
            legal_pad[j, :len(vecs)] = vecs

        obs_t       = torch.as_tensor(np.stack([x[2] for x in active]),
                                      dtype=torch.float32, device=device)
        pass_src_t  = torch.as_tensor(np.stack([x[3] for x in active]),
                                      dtype=torch.int64, device=device)
        pass_tgt_t  = torch.as_tensor(np.stack([x[4] for x in active]),
                                      dtype=torch.int64, device=device)
        legal_pad_t = torch.as_tensor(legal_pad, device=device)
        n_legal_t   = torch.as_tensor(n_arr, device=device)

        actions_t, logprobs_t, values_t = policy.act(
            obs_t, legal_pad_t, n_legal_t, pass_src_t, pass_tgt_t
        )

        # Phase 5: execute and record
        for j, (game_idx, player, obs, pass_src, pass_tgt, vecs, legal) in enumerate(active):
            gs         = games[game_idx]
            action_idx = int(actions_t[j].item())
            action     = legal[action_idx] if action_idx < len(legal) else legal[0]

            ep_steps[game_idx].append(
                (
                    obs,
                    pass_src,
                    pass_tgt,
                    vecs,
                    action_idx,
                    float(logprobs_t[j].item()),
                    float(values_t[j].item()),
                    player,
                )
            )

            gs.play(player, action)

            # ── NEW: Handle Mahjong Wish ──
            cards = action[0]
            if MAHJONG_INDEX in cards and not gs.round_over and gs.mahjong_wish is None:
                # If we led Mahjong, we might need to wish
                # (Note: the environment might already be asking for a wish)
                # To be sure, we check if the environment is in a state that allows wishing.
                # In tichu_env, gs.play() handles the wish if it's part of the action,
                # but here we might need to call it explicitly if the env expects it.
                # Actually, gs.play() in tichu_env doesn't take the wish as an argument.
                # It sets gs.mahjong_wish = None and expects a set_mahjong_wish() call.
                
                # Use policy to wish
                with torch.no_grad():
                    # Get fresh fused features for the current state (post-play)
                    encoder = (
                        encode_static_counted_trick_points
                        if counted_trick_points
                        else encode_static
                    )
                    obs_post = torch.as_tensor(
                        encoder(gs, player)[None, :], device=device
                    )
                    ps_post = torch.as_tensor(encode_pass_source_ids(gs, player)[None, :], device=device)
                    pt_post = torch.as_tensor(encode_pass_target_ids(gs, player)[None, :], device=device)
                    feat = policy.trunk(obs_post)
                    # Add pass source info
                    pf = policy.pass_source_embed(ps_post).reshape(1, -1)
                    feat = feat + policy.pass_source_proj(pf)
                    pf = policy.pass_target_embed(pt_post).reshape(1, -1)
                    feat = feat + policy.pass_target_proj(pf)
                    
                    wish_logits = policy.get_mahjong_wish_logits(feat)
                    wish_idx = torch.argmax(wish_logits, dim=-1).item()
                    # Mapping: index 0 -> no wish (None/0), index 1-13 -> ranks 2-14
                    if wish_idx == 0:
                        gs.set_mahjong_wish(None)
                    else:
                        gs.set_mahjong_wish(int(wish_idx + 1))

            if gs.round_over:
                done[game_idx] = True

    # Compile flat buffer
    all_obs, all_pass_src, all_pass_tgt, all_legal_vecs, all_action_idxs = [], [], [], [], []
    all_logprobs, all_values, all_returns = [], [], []
    all_steps_to_end, all_steps_from_start = [], []
    all_double_wins, all_players = [], []
    game_rewards: list[list[float]] = []

    for i, gs in enumerate(games):
        score_a, score_b = gs.round_scores
        diff         = (score_a - score_b) / 400.0
        player_reward = [diff if p in TEAM_A else -diff for p in range(4)]
        game_rewards.append(player_reward)

        first, second = gs.finish_order[:2]
        double_win = (first - second) % 2 == 0
        n_steps = len(ep_steps[i])
        for step_idx, (
            obs,
            pass_src,
            pass_tgt,
            legal_vecs,
            action_idx,
            logprob,
            value,
            player,
        ) in enumerate(ep_steps[i]):
            all_obs.append(obs)
            all_pass_src.append(pass_src)
            all_pass_tgt.append(pass_tgt)
            all_legal_vecs.append(legal_vecs)
            all_action_idxs.append(action_idx)
            all_logprobs.append(logprob)
            all_values.append(value)
            all_returns.append(player_reward[player])
            all_steps_to_end.append(n_steps - step_idx)
            all_steps_from_start.append(step_idx + 1)
            all_double_wins.append(double_win)
            all_players.append(player)

    obs_arr    = np.array(all_obs,         dtype=np.float32)       # (N, 196)
    pass_src_arr = np.array(all_pass_src,  dtype=np.int64)         # (N, 3)
    pass_tgt_arr = np.array(all_pass_tgt,  dtype=np.int64)         # (N, 3)
    legal_flat = np.concatenate(all_legal_vecs, axis=0)            # (L, 58)
    legal_lens = np.array([v.shape[0] for v in all_legal_vecs],
                          dtype=np.int32)                           # (N,)
    act_arr    = np.array(all_action_idxs, dtype=np.int64)         # (N,)
    lp_arr     = np.array(all_logprobs,    dtype=np.float32)       # (N,)
    val_arr    = np.array(all_values,      dtype=np.float32)       # (N,)
    ret_arr    = np.array(all_returns,     dtype=np.float32)       # (N,)
    steps_arr  = np.array(all_steps_to_end, dtype=np.int16)        # (N,)
    start_steps_arr = np.array(all_steps_from_start, dtype=np.int16)  # (N,)
    double_arr = np.array(all_double_wins, dtype=np.bool_)         # (N,)
    players_arr = np.array(all_players, dtype=np.int8)              # (N,)

    buffer = {
        "obs":         obs_arr,
        "pass_source_ids": pass_src_arr,
        "pass_target_ids": pass_tgt_arr,
        "legal_flat":  legal_flat,
        "legal_lens":  legal_lens,
        "action_idxs": act_arr,
        "log_probs":   lp_arr,
        "values":      val_arr,
        "returns":     ret_arr,
        "advantages":  ret_arr - val_arr,
        "steps_to_end": steps_arr,
        "steps_from_start": start_steps_arr,
        "double_win": double_arr,
        "players": players_arr,
    }
    if pass_policy is not None:
        pass_rows = [
            (obs, cards, choices, log_probs, game_rewards[game_idx][player])
            for game_idx in range(len(games))
            for obs, cards, choices, log_probs, player
            in pass_steps[game_idx]
        ]
        buffer["pass_train_obs"] = np.stack(
            [row[0] for row in pass_rows]
        ).astype(np.float32)
        buffer["pass_train_hand_cards"] = np.stack(
            [row[1] for row in pass_rows]
        ).astype(np.int64)
        buffer["pass_train_choices"] = np.stack(
            [row[2] for row in pass_rows]
        ).astype(np.int64)
        buffer["pass_train_log_probs"] = np.stack(
            [row[3] for row in pass_rows]
        ).astype(np.float32)
        buffer["pass_train_returns"] = np.array(
            [row[4] for row in pass_rows], dtype=np.float32
        )
    return buffer


# ── Level-2: worker entry point ───────────────────────────────────────────────

def _worker_collect(args: tuple) -> dict[str, np.ndarray]:
    (
        weights_np,
        gt_weights_np,
        pass_weights_np,
        n_games,
        device_str,
        n_torch_threads,
        with_calling,
        model_cfg,
        pass_model_cfg,
        gt_model_cfg,
        counted_trick_points,
    ) = args
    torch.set_num_threads(n_torch_threads)
    torch.manual_seed(os.getpid())
    device = torch.device(device_str)

    from training.model_embed import TichuPolicyEmbed
    policy = TichuPolicyEmbed(**model_cfg).to(device)
    policy.load_state_dict({k: torch.from_numpy(v) for k, v in weights_np.items()})
    policy.eval()

    gt_policy = None
    if gt_weights_np is not None:
        from training.model_gt import TichuGrandTichuPolicy
        gt_policy = TichuGrandTichuPolicy(**gt_model_cfg).to(device)
        gt_policy.load_state_dict({k: torch.from_numpy(v) for k, v in gt_weights_np.items()})
        gt_policy.eval()

    pass_policy = None
    if pass_weights_np is not None:
        from training.model_passing import TichuPassingPolicy
        pass_policy = TichuPassingPolicy(**pass_model_cfg).to(device)
        pass_policy.load_state_dict({k: torch.from_numpy(v) for k, v in pass_weights_np.items()})
        pass_policy.eval()

    return _collect_batched(
        policy,
        n_games,
        device,
        with_calling=with_calling,
        gt_policy=gt_policy,
        pass_policy=pass_policy,
        counted_trick_points=counted_trick_points,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def collect_rollouts_embed(
    policy,
    n_games: int,
    device: torch.device,
    pool=None,
    n_workers: int = 1,
    with_calling: bool = False,
    gt_policy=None,
    pass_policy=None,
    model_cfg: dict | None = None,
    pass_model_cfg: dict | None = None,
    gt_model_cfg: dict | None = None,
    counted_trick_points: bool = False,
) -> dict[str, np.ndarray]:
    """Collect n_games of self-play rollouts using action-embedding buffer format."""
    if pool is None or n_workers <= 1:
        return _collect_batched(
            policy, n_games, device,
            with_calling=with_calling,
            gt_policy=gt_policy,
            pass_policy=pass_policy,
            counted_trick_points=counted_trick_points,
        )

    weights_np = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}
    gt_weights_np = None
    if gt_policy is not None:
        gt_weights_np = {k: v.detach().cpu().numpy() for k, v in gt_policy.state_dict().items()}
    pass_weights_np = None
    if pass_policy is not None:
        pass_weights_np = {k: v.detach().cpu().numpy() for k, v in pass_policy.state_dict().items()}
    
    device_str = str(device)
    n_torch_threads = max(1, os.cpu_count() // n_workers)
    mdl_cfg = model_cfg or {}
    pass_mdl_cfg = pass_model_cfg or {}
    gt_mdl_cfg = gt_model_cfg or {}

    base, rem = divmod(n_games, n_workers)
    sizes = [base + (1 if i < rem else 0) for i in range(n_workers)]
    sizes = [s for s in sizes if s > 0]
    args = [
        (
            weights_np,
            gt_weights_np,
            pass_weights_np,
            s,
            device_str,
            n_torch_threads,
            with_calling,
            mdl_cfg,
            pass_mdl_cfg,
            gt_mdl_cfg,
            counted_trick_points,
        )
        for s in sizes
    ]

    buffers = pool.map(_worker_collect, args)

    # Concatenate — legal_flat is variable-length per buffer but concat is fine
    return {k: np.concatenate([b[k] for b in buffers]) for k in buffers[0]}
