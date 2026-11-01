"""Collect self-play rollouts — batched across N parallel games.

Two levels of parallelism:

  1. Within-process batching: N games advance in lock-step; observations
     from all of them are stacked into one (N, 196) matrix for a single
     forward pass.  Turns ~N×T individual (1,196) matmuls into ~T batched
     (N,196) matmuls.

  2. Multi-process: the caller can pass a multiprocessing.Pool; the N games
     are split across W worker processes, each doing level-1 batching on
     their share.  Workers receive model weights once per update (pickled,
     ~2.5 MB), run their games, and return the buffer.

Public API
----------
collect_rollouts(policy, n_games, device, pool=None, n_workers=1)
"""

from __future__ import annotations

import os
import random
import time
import numpy as np
import torch
from training.stats import get_system_stats

from tichu_env.cards import MAHJONG_INDEX
from tichu_env.encoding import encode_static
from tichu_env.constants import MAX_ACTIONS
from tichu_env.game_state import GameState, TEAM_A


# ── Level-1: batched single-process collection ────────────────────────────────

def _collect_batched(
    policy,
    n_games: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run n_games in lock-step, one batched forward pass per tick."""

    rng = random.Random()

    games: list[GameState] = []
    for _ in range(n_games):
        gs = GameState()
        gs.reset()
        games.append(gs)

    done = [False] * n_games

    ep_obs      : list[list[np.ndarray]] = [[] for _ in range(n_games)]
    ep_n_legal  : list[list[int]]        = [[] for _ in range(n_games)]
    ep_actions  : list[list[int]]        = [[] for _ in range(n_games)]
    ep_logprobs : list[list[float]]      = [[] for _ in range(n_games)]
    ep_values   : list[list[float]]      = [[] for _ in range(n_games)]
    ep_players  : list[list[int]]        = [[] for _ in range(n_games)]

    policy.eval()

    while not all(done):
        # Phase 1: advance dragon-give decisions (no policy needed)
        for i, gs in enumerate(games):
            if done[i]:
                continue
            while gs.dragon_give_pending is not None and not gs.round_over:
                opts = gs.get_dragon_give_options(gs.dragon_give_pending)
                gs.give_dragon_trick(gs.dragon_give_pending, rng.choice(opts))
            if gs.round_over:
                done[i] = True

        if all(done):
            break

        # Phase 2: collect observations from all active games
        active: list[tuple[int, int, np.ndarray, int, list]] = []
        for i, gs in enumerate(games):
            if done[i]:
                continue
            player  = gs.current_player
            legal   = gs.get_legal_actions(player)
            n_legal = min(len(legal), MAX_ACTIONS)

            obs = encode_static(gs, player)
            active.append((i, player, obs, n_legal, legal))

        if not active:
            break

        # Phase 3: one batched forward pass over all active games
        # Build mask from n_legal: mask[i, :n_legal[i]] = True
        n_legal_arr = np.array([x[3] for x in active], dtype=np.int16)
        obs_t  = torch.as_tensor(
            np.stack([x[2] for x in active]), dtype=torch.float32, device=device
        )
        idx    = torch.arange(MAX_ACTIONS, device=device)
        mask_t = idx.unsqueeze(0) < torch.as_tensor(n_legal_arr, device=device).unsqueeze(1)

        actions_t, logprobs_t, values_t = policy.act(obs_t, mask_t)

        # Phase 4: execute and record
        for j, (game_idx, player, obs, n_legal, legal) in enumerate(active):
            gs         = games[game_idx]
            action_idx = int(actions_t[j].item())
            card_play  = legal[action_idx] if action_idx < len(legal) else legal[0]

            ep_obs[game_idx].append(obs)
            ep_n_legal[game_idx].append(n_legal)
            ep_actions[game_idx].append(action_idx)
            ep_logprobs[game_idx].append(float(logprobs_t[j].item()))
            ep_values[game_idx].append(float(values_t[j].item()))
            ep_players[game_idx].append(player)

            gs.play(player, card_play)

            cards = card_play[0]
            if MAHJONG_INDEX in cards and not gs.round_over and rng.random() < 0.5:
                gs.set_mahjong_wish(rng.randint(2, 14))

            if gs.round_over:
                done[game_idx] = True

    # Compile flat buffer
    all_obs, all_n_legal, all_actions = [], [], []
    all_logprobs, all_values, all_returns = [], [], []

    for i, gs in enumerate(games):
        score_a, score_b = gs.round_scores
        diff          = (score_a - score_b) / 400.0
        player_reward = [diff if p in TEAM_A else -diff for p in range(4)]

        for step, player in enumerate(ep_players[i]):
            all_obs.append(ep_obs[i][step])
            all_n_legal.append(ep_n_legal[i][step])
            all_actions.append(ep_actions[i][step])
            all_logprobs.append(ep_logprobs[i][step])
            all_values.append(ep_values[i][step])
            all_returns.append(player_reward[player])

    obs_arr     = np.array(all_obs,      dtype=np.float32)
    n_legal_arr = np.array(all_n_legal,  dtype=np.int16)
    act_arr     = np.array(all_actions,  dtype=np.int64)
    lp_arr      = np.array(all_logprobs, dtype=np.float32)
    val_arr     = np.array(all_values,   dtype=np.float32)
    ret_arr     = np.array(all_returns,  dtype=np.float32)

    return {
        "obs":        obs_arr,
        "n_legal":    n_legal_arr,   # int16 per step — mask is reconstructed in ppo_update
        "actions":    act_arr,
        "log_probs":  lp_arr,
        "values":     val_arr,
        "returns":    ret_arr,
        "advantages": ret_arr - val_arr,
    }


# ── Level-2: worker entry point (must be top-level to be pickleable) ──────────

def _worker_collect(args: tuple) -> dict[str, np.ndarray]:
    """Run in a worker process: rebuild policy from weights, collect games."""
    weights_np, n_games, device_str, n_torch_threads = args

    # Without this each worker spawns a full torch/OpenMP thread pool (all CPU
    # cores), so N workers × all-core pools thrash on a fixed number of cores.
    torch.set_num_threads(n_torch_threads)

    # Re-seed so workers produce different action sequences
    torch.manual_seed(os.getpid())

    from training.model import TichuPolicy  # noqa: PLC0415
    device = torch.device(device_str)
    policy = TichuPolicy(
        obs_size=weights_np["trunk.0.weight"].shape[1],
        n_actions=weights_np["policy_head.weight"].shape[0],
        hidden_dim=weights_np["policy_head.weight"].shape[1],
        n_layers=sum(
            key.startswith("trunk.") and key.endswith(".weight")
            for key in weights_np
        ),
    ).to(device)
    policy.load_state_dict({k: torch.from_numpy(v) for k, v in weights_np.items()})
    policy.eval()
    return _collect_batched(policy, n_games, device)


# ── Public API ─────────────────────────────────────────────────────────────────

def collect_rollouts(
    policy,
    n_games: int,
    device: torch.device,
    pool=None,
    n_workers: int = 1,
) -> dict[str, np.ndarray]:
    """Collect n_games of self-play rollouts.

    If pool is None (or n_workers == 1), runs in the calling process.
    Otherwise splits games across the pool's workers and concatenates results.
    """
    if pool is None or n_workers <= 1:
        return _collect_batched(policy, n_games, device)

    # Serialize weights as numpy — fast pickle, ~2.5 MB for our network
    weights_np = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}
    device_str = str(device)

    # Give each worker an equal share of CPU cores for torch's internal threads
    n_torch_threads = max(1, os.cpu_count() // n_workers)

    print(f"  [collecting] Starting collection of {n_games} games across {n_workers} workers...", flush=True)

    base, rem = divmod(n_games, n_workers)
    sizes = [base + (1 if i < rem else 0) for i in range(n_workers)]
    sizes = [size for size in sizes if size > 0]
    args  = [(weights_np, s, device_str, n_torch_threads) for s in sizes]

    result = pool.map_async(_worker_collect, args)
    
    # Heartbeat while waiting for workers
    start_t = time.perf_counter()
    last_heartbeat = start_t
    while not result.ready():
        now = time.perf_counter()
        if now - last_heartbeat > 10.0:
            stats = get_system_stats(device)
            elapsed = now - start_t
            print(f"  [collecting] {elapsed:.0f}s elapsed | cpu={stats['cpu_load_1m']} | ram={stats['ram_avail_gb']}GB", flush=True)
            last_heartbeat = now
        time.sleep(1.0)

    buffers = result.get()

    return {k: np.concatenate([b[k] for b in buffers]) for k in buffers[0]}
