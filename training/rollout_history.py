"""Rollout collection for the history-aware policy.

Buffer format:
  obs           (N, 196)               float32
  legal_flat    (L, 58)                float32   — all legal action vectors, concatenated
  legal_lens    (N,)                   int32     — n_legal per step
  action_idxs   (N,)                   int64
  log_probs     (N,)                   float32
  values        (N,)                   float32
  returns       (N,)                   float32
  advantages    (N,)                   float32
  history_tokens       (H, 59)         float16   — objective rows, stored once per game
  history_players      (H,)            int8      — absolute actor for each token
  history_game_offsets (G+1,)          int32     — token range for each game
  history_game_ids     (N,)            int32     — source game per transition
  history_observers    (N,)            int8      — observing seat per transition
  history_lens         (N,)            int32     — prefix length per transition
  pass_source_ids (N, 3)               int64
  pass_target_ids (N, 3)               int64

With a pass_policy, every pass decision is harvested for the pass net's
realized-return training (ppo_passing.py):
  pass_train_obs        (P, 60)  float32
  pass_train_hand_cards (P, 14)  int64
  pass_train_choices    (P, 3)   int64   — slot order: left, partner, right
  pass_train_log_probs  (P, 3)   float32
  pass_train_returns    (P,)     float32 — passer's realized team return
(in gt_probe mode the last key is instead pass_train_rewards: the declared
branches' control-differenced returns, already a causal advantage.)

Public API
----------
collect_rollouts_history(policy, n_games, device, pool=None, n_workers=1,
                         encoder_cfg=None)
"""

from __future__ import annotations

import copy
import os
import random
import time
import multiprocessing as mp
from typing import NamedTuple
import numpy as np

import torch
from torch.distributions import Categorical

from training.stats import get_system_stats

from tichu_env.cards import MAHJONG_INDEX, NUM_CARDS
from tichu_env.encoding import (
    encode_static,
    encode_history,
    encode_pass_source_ids,
    encode_pass_target_ids,
    MAX_HISTORY_LEN,
    PASS_OBS_SIZE,
    PLAY_TOKEN_SIZE,
    STATIC_OBS_SIZE,
)
from tichu_env.game_state import GameState, TEAM_A

from training.model_embed import ACTION_DIM
from training.rollout_embed import _action_vec

ROLLOUT_FORWARD_BUCKET = 512

# The main PPO process has peaked a little above 3 GiB on the production T4.
# Rollout workers keep their models alive between jobs, which is useful, but
# their independent CUDA caching allocators can otherwise retain almost the
# entire device.  Keep enough global headroom for PPO and release a worker's
# *unused* cache only when the device is under pressure.  The ordinary fast
# path performs memory counter reads but no allocator cleanup.
WORKER_CUDA_MIN_FREE_BYTES = 4 * 1024**3
WORKER_CUDA_MIN_RELEASABLE_BYTES = 64 * 1024**2
WORKER_CUDA_LIVE_GROWTH_TOLERANCE_BYTES = 64 * 1024**2
# Release every worker's CUDA cache after each collection, regardless of
# headroom. Needed when the trainer shares a small GPU with its workers (the
# 10 GB RTX 3080 runs out during PPO otherwise); costs re-allocation time.
WORKER_CUDA_ALWAYS_RELEASE = os.environ.get("TICHU_WORKER_CUDA_ALWAYS_RELEASE", "") not in ("", "0")

# With this probability a player's Grand Tichu decision is replaced by a coin
# flip. Keeps GT=1 states in the self-play distribution so the value net stays
# calibrated on them — without it, a "never call" GT head starves the value
# net of GT=1 data and its (negative) belief about declaring can never be
# corrected (the v28/v29 collapse).
# Raised 0.02 -> 0.10 (2026-07-21, user-directed): with the head honestly at
# p~0 the main stream had zero declared rounds; eps supplies them at no rollout
# cost (~18.5% of rounds get a declarer at 0.10). The head never trains on
# these (probe-only since v32) — this feeds the play/value/pass nets.
GT_EXPLORE_EPS = 0.10

# Global stats for profiling
ROLLOUT_STATS = {
    "initialization_s": 0.0,
    "immediate_decisions_s": 0.0,
    "legal_actions_s": 0.0,
    "obs_encode_s": 0.0,
    "action_vec_s": 0.0,
    "history_encode_s": 0.0,
    "forward_s": 0.0,
    "env_step_s": 0.0,
    "specialist_heads_s": 0.0,
    "specialist_preplay_s": 0.0,
    "specialist_postplay_s": 0.0,
    "buffer_compile_s": 0.0,
    "buffer_pass1_s": 0.0,
    "buffer_legal_concat_s": 0.0,
    "buffer_history_rebuild_s": 0.0,
    "worker_setup_s": 0.0,
    "worker_cuda_check_s": 0.0,
    "worker_cuda_check_count": 0,
    "worker_cuda_allocated_mb": 0.0,
    "worker_cuda_reserved_mb": 0.0,
    "worker_cuda_cached_mb": 0.0,
    "worker_cuda_free_before_min_mb": 0.0,
    "worker_cuda_free_after_min_mb": 0.0,
    "worker_cuda_cache_released_mb": 0.0,
    "worker_cuda_cleanup_count": 0,
    "worker_cuda_leak_suspected_count": 0,
    "worker_wall_s": 0.0,
    "worker_wall_max_s": 0.0,
    "worker_count": 0,
    "main_dispatch_s": 0.0,
    "main_wait_s": 0.0,
    "main_ipc_s": 0.0,
    "main_merge_s": 0.0,
    "total_steps": 0,
}

def reset_rollout_stats():
    for k in ROLLOUT_STATS:
        ROLLOUT_STATS[k] = 0.0


def _hand_to_binary(hand: list[int]) -> np.ndarray:
    out = np.zeros(NUM_CARDS, dtype=np.float32)
    out[np.asarray(hand, dtype=np.int64)] = 1.0
    return out


def _rebuild_history_buffer(
    games: list[GameState],
    ep_steps: list[list[tuple]],
    hist_lens_arr: np.ndarray,
) -> np.ndarray:
    """Rebuild per-step relative history snapshots from final game logs.

    Rollout collection stores only each step's observer and history length. The
    objective token data is immutable after a play, so the final per-game log can
    be replayed into every step and only the relative-player one-hot bits need
    to be recomputed per observer.
    """
    n_steps = int(hist_lens_arr.shape[0])
    max_h = int(hist_lens_arr.max()) if n_steps and hist_lens_arr.max() > 0 else 1
    history_arr = np.zeros((n_steps, max_h, PLAY_TOKEN_SIZE), dtype=np.float16)

    curr = 0
    for gs, steps in zip(games, ep_steps):
        if not steps:
            continue

        steps_meta = np.array([(s[8], s[7]) for s in steps], dtype=np.int32)
        h_lens = steps_meta[:, 0]
        p_idxs = steps_meta[:, 1]
        num_steps = len(steps)
        max_copy = int(min(max(h_lens), max_h, MAX_HISTORY_LEN))

        if max_copy > 0 and gs.history_tokens is not None and gs.player_indices is not None:
            max_copy = min(max_copy, len(gs.history_tokens))
            col_ids = np.arange(max_copy, dtype=np.int32)[None, :]
            valid = col_ids < h_lens[:, None]
            step_ids, token_ids = np.nonzero(valid)

            if len(step_ids) > 0:
                history_arr[curr + step_ids, token_ids, 4:63] = gs.history_tokens[
                    token_ids
                ]
                abs_players = gs.player_indices[token_ids]
                observers = p_idxs[step_ids]
                rel_players = (abs_players - observers) % 4
                history_arr[curr + step_ids, token_ids, rel_players] = 1.0

        curr += num_steps

    return history_arr


def _compile_shared_history(
    games: list[GameState],
    ep_steps: list[list[tuple]],
) -> dict[str, np.ndarray]:
    """Pack each immutable game log once plus transition prefix references."""
    n_steps = sum(len(steps) for steps in ep_steps)
    game_ids = np.empty(n_steps, dtype=np.int32)
    observers = np.empty(n_steps, dtype=np.int8)
    game_offsets = np.zeros(len(games) + 1, dtype=np.int32)
    token_chunks: list[np.ndarray] = []
    player_chunks: list[np.ndarray] = []

    step_offset = 0
    token_offset = 0
    for game_id, (gs, steps) in enumerate(zip(games, ep_steps)):
        count = len(steps)
        if count:
            game_ids[step_offset : step_offset + count] = game_id
            observers[step_offset : step_offset + count] = np.fromiter(
                (step[7] for step in steps),
                dtype=np.int8,
                count=count,
            )
            step_offset += count

        max_prefix = min(
            max((int(step[8]) for step in steps), default=0),
            MAX_HISTORY_LEN,
        )
        if max_prefix:
            assert gs.history_tokens is not None
            assert gs.player_indices is not None
            token_chunks.append(
                gs.history_tokens[:max_prefix].astype(np.float16, copy=True)
            )
            player_chunks.append(
                gs.player_indices[:max_prefix].astype(np.int8, copy=True)
            )
            token_offset += max_prefix
        game_offsets[game_id + 1] = token_offset

    objective_dim = PLAY_TOKEN_SIZE - 4
    return {
        "history_tokens": (
            np.concatenate(token_chunks, axis=0)
            if token_chunks
            else np.zeros((0, objective_dim), dtype=np.float16)
        ),
        "history_players": (
            np.concatenate(player_chunks)
            if player_chunks
            else np.zeros(0, dtype=np.int8)
        ),
        "history_game_offsets": game_offsets,
        "history_game_ids": game_ids,
        "history_observers": observers,
    }


def _compile_spec_buffer(kind: str, rows: list[tuple]) -> dict[str, np.ndarray]:
    """Stack specialist-head decisions (dragon/wish) into flat arrays.

    History stays padded to MAX_HISTORY_LEN so multi-worker merging can use a
    plain concatenate; the PPO update trims it before the forward pass.
    """
    D = len(rows)
    out = {
        f"{kind}_obs": np.zeros((D, STATIC_OBS_SIZE), dtype=np.float32),
        f"{kind}_pass_source_ids": np.zeros((D, 3), dtype=np.int64),
        f"{kind}_pass_target_ids": np.zeros((D, 3), dtype=np.int64),
        f"{kind}_history": np.zeros((D, MAX_HISTORY_LEN, PLAY_TOKEN_SIZE), dtype=np.float16),
        f"{kind}_history_lens": np.zeros(D, dtype=np.int32),
        f"{kind}_actions": np.zeros(D, dtype=np.int64),
        f"{kind}_log_probs": np.zeros(D, dtype=np.float32),
        f"{kind}_values": np.zeros(D, dtype=np.float32),
        f"{kind}_returns": np.zeros(D, dtype=np.float32),
    }
    for j, (obs, ps, pt, hist, h_len, a_idx, lp, val, ret) in enumerate(rows):
        out[f"{kind}_obs"][j] = obs
        out[f"{kind}_pass_source_ids"][j] = ps
        out[f"{kind}_pass_target_ids"][j] = pt
        out[f"{kind}_history"][j, : hist.shape[0]] = hist
        out[f"{kind}_history_lens"][j] = h_len
        out[f"{kind}_actions"][j] = a_idx
        out[f"{kind}_log_probs"][j] = lp
        out[f"{kind}_values"][j] = val
        out[f"{kind}_returns"][j] = ret
    return out


def _actor_critic_features(
    policy,
    obs: torch.Tensor,
    history: torch.Tensor,
    history_len: torch.Tensor,
    pass_source_ids: torch.Tensor,
    pass_target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use split readouts when supported, with legacy-policy compatibility."""
    split_features = getattr(policy, "actor_critic_features", None)
    if split_features is not None:
        return split_features(
            obs, history, history_len, pass_source_ids, pass_target_ids
        )
    fused = policy.fused_features(
        obs, history, history_len, pass_source_ids, pass_target_ids
    )
    return fused, fused


# ── Single-process batched collection ────────────────────────────────────────


class RolloutAgent(NamedTuple):
    """One seat controller: play policy plus its optional GT and pass heads."""

    policy: object
    gt_policy: object = None
    pass_policy: object = None


def _collect_batched(
    policy,
    n_games: int,
    device: torch.device,
    use_rollout_bucketing: bool = True,
    with_calling: bool = False,
    gt_policy=None,
    pass_policy=None,
    gt_probe: bool = False,
    disable_gt_explore: bool = False,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Single-policy self-play collection (all four seats share ``policy``)."""
    buffers, stats = _collect_batched_multi(
        [RolloutAgent(policy, gt_policy, pass_policy)],
        n_games,
        device,
        use_rollout_bucketing=use_rollout_bucketing,
        with_calling=with_calling,
        gt_probe=gt_probe,
        disable_gt_explore=disable_gt_explore,
        seed=seed,
    )
    return buffers[0], stats


def _collect_batched_multi(
    agents: list[RolloutAgent],
    n_games: int,
    device: torch.device,
    seat_owners: list[tuple[int, int, int, int]] | None = None,
    learners: tuple[int, ...] = (0,),
    use_rollout_bucketing: bool = True,
    with_calling: bool = False,
    gt_probe: bool = False,
    disable_gt_explore: bool = False,
    seed: int | None = None,
) -> tuple[dict[int, dict[str, np.ndarray]], dict]:
    """Run n_games in lock-step with batched forward passes.

    ``seat_owners[i][p]`` is the index into ``agents`` controlling seat ``p``
    of game ``i`` (default: agent 0 everywhere, i.e. pure self-play). Each
    agent acts with its own heads; forward passes are batched per agent.
    Only decisions of agents listed in ``learners`` are recorded, and one
    training buffer is returned per learner. ``GameState.player_model_ids``
    is set from the owners so the partner same-weights bit is truthful.

    gt_probe mode plays shared-control probe groups for the Grand Tichu head:
    each group of 5 games shares the exact same deal (deepcopy after reset,
    before any decision) — game 5j is the all-quiet control (no seat declares)
    and game 5j+1+k forces seat k as the sole declarer. Each declared branch's
    outcome minus the shared control, from that seat's perspective, is the
    causal effect of a lone declare on that hand with the whole-deal luck
    cancelled: 4 marginal measurements (and 4 declared-round practice games)
    per 5 rollouts instead of 1 per 2 under the old one-seat-toggle pairing.
    The four diffs of a group share the control's noise (correlated errors)
    but each is unbiased. Declarations are fully forced — the head is never
    sampled — so every branch stays in the deployment regime of exactly one
    declarer. Probe decisions are never harvested into the PPO gt_* buffers;
    results land in gt_probe_* keys instead.
    """

    initialization_started = time.perf_counter()
    rng = random.Random(seed)

    if seat_owners is None:
        seat_owners = [(0, 0, 0, 0)] * n_games
    if len(seat_owners) != n_games or any(
        len(owners) != 4 or not all(0 <= o < len(agents) for o in owners)
        for owners in seat_owners
    ):
        raise ValueError("seat_owners needs one 4-tuple of agent indices per game")
    learner_set = frozenset(learners)
    if not learner_set or not learner_set <= set(range(len(agents))):
        raise ValueError("learners must be a non-empty subset of agent indices")
    if gt_probe and (len(agents) != 1 or learner_set != {0}):
        raise ValueError("gt_probe supports single-agent self-play only")
    pass_policy = agents[0].pass_policy

    if gt_probe:
        if not with_calling or n_games % 5 != 0:
            raise ValueError(
                "gt_probe needs with_calling=True and n_games divisible by 5 "
                "(one shared quiet control + four sole-declarer branches per "
                "deal)")
        n_groups = n_games // 5
        originals = [GameState(track_history=True) for _ in range(n_groups)]
        for gs in originals:
            gs.reset(
                seed=rng.randrange(2**63),
                with_calling=True,
                with_passing=pass_policy is not None,
            )
        games: list[GameState] = []
        for gs in originals:
            games.append(gs)
            for _ in range(4):
                games.append(copy.deepcopy(gs))

        # All four 8-card hands are known at reset, so every declaration in
        # every branch is decided up front; the play loop only looks them up.
        # Row 4*j + k of probe_h8 is group j, seat k — matching the harvest
        # order of the diffs below.
        probe_h8 = np.stack([
            _hand_to_binary(sorted(gs.hands[p]))
            for gs in originals for p in range(4)
        ]).astype(np.float32)
        probe_forced: list[dict[int, bool]] = []
        for j in range(n_groups):
            probe_forced.append({q: False for q in range(4)})
            for k in range(4):
                probe_forced.append({q: q == k for q in range(4)})
    else:
        games = [GameState(track_history=True) for _ in range(n_games)]
        for gs in games:
            gs.reset(
                seed=rng.randrange(2**63),
                with_calling=with_calling,
                with_passing=pass_policy is not None,
            )

    for gs, owners in zip(games, seat_owners):
        gs.player_model_ids = list(owners)

    done = [False] * n_games

    ep_steps: list[
        list[
            tuple[
                np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                int, float, float, int, int
            ]
        ]
    ] = [[] for _ in range(n_games)]

    # Specialist-head decisions (dragon give / mahjong wish), trained via PPO
    # alongside the main policy. Entries:
    # (kind, obs, pass_src, pass_tgt, hist_f16, hist_len, action, log_prob, value, player)
    spec_steps: list[list[tuple]] = [[] for _ in range(n_games)]

    # Grand tichu decisions, trained on realized round returns (ppo_gt.py).
    # Entries: (h8_binary, declare, behavior_log_prob, player)
    gt_steps: list[list[tuple]] = [[] for _ in range(n_games)]

    # Passing decisions, trained on realized round returns (ppo_passing.py).
    # Entries: (pass_obs, hand_cards, choices, log_probs, player)
    pass_steps: list[list[tuple]] = [[] for _ in range(n_games)]

    for agent in agents:
        agent.policy.eval()
    ROLLOUT_STATS["initialization_s"] += (
        time.perf_counter() - initialization_started
    )

    total_steps_in_collection = 0
    while not all(done):
        total_steps_in_collection += 1
        
        # ── Diagnostic Failsafe ──────────────────────────────────────
        if total_steps_in_collection > 1900 or (total_steps_in_collection % 100 == 0 and total_steps_in_collection > 500):
            stuck_indices = [i for i, d in enumerate(done) if not d]
            print(f"\n[DEBUG] Step {total_steps_in_collection}. Stuck: {len(stuck_indices)}/{n_games}", flush=True)
            for idx in stuck_indices[:2]:
                gs = games[idx]
                print(f"  G{idx}: ph={gs.phase.name} cp={gs.current_player} "
                      f"dg={gs.dragon_give_pending} round_over={gs.round_over} "
                      f"hands={[len(h) for h in gs.hands]} "
                      f"finished={gs.finish_order}", flush=True)

        if total_steps_in_collection > 2000:
            raise RuntimeError(
                f"Too many steps in collection loop ({total_steps_in_collection})"
            )
        
        # ── IMMEDIATE DECISIONS LOOP (GT & Dragon Give) ───────────────
        # Some decisions don't advance the "main" trick flow but must be resolved.
        # We loop here because resolving a Dragon Give might trigger another
        # Dragon Give (rare) or finish a round.
        immediate_started = time.perf_counter()
        inner_safety = 0
        while True:
            inner_safety += 1
            if inner_safety > 100:
                raise RuntimeError("Stuck in immediate decision loop")
            
            any_changed = False
            
            # 1. Resolve Grand Tichu. Head probabilities come from one batched
            # forward per owner; sampling then walks the games in order so the
            # rng stream matches per-game evaluation.
            gt_h8: dict[int, np.ndarray] = {}
            gt_prob: dict[int, float] = {}
            if not gt_probe:
                gt_rows: dict[int, list[int]] = {}
                for i, gs in enumerate(games):
                    if not done[i] and gs.phase.name == "GRAND_TICHU":
                        owner = seat_owners[i][gs.current_player]
                        if agents[owner].gt_policy is not None:
                            gt_h8[i] = _hand_to_binary(sorted(list(gs.hands[gs.current_player]))[:8])
                            gt_rows.setdefault(owner, []).append(i)
                for owner, rows in sorted(gt_rows.items()):
                    h8_bin = torch.as_tensor(np.stack([gt_h8[i] for i in rows]), device=device)
                    with torch.no_grad():
                        logits = agents[owner].gt_policy.call_logits(h8_bin)
                    gt_prob.update(zip(rows, torch.sigmoid(logits).reshape(-1).cpu().tolist()))
            for i, gs in enumerate(games):
                if done[i]: continue
                if gs.phase.name == "GRAND_TICHU":
                    p = gs.current_player
                    owner = seat_owners[i][p]
                    gt_policy = agents[owner].gt_policy
                    if gt_probe:
                        gs.declare_grand_tichu(p, probe_forced[i][p])
                    elif gt_policy is not None:
                        h8_vec = gt_h8[i]
                        prob = gt_prob[i]
                        eps = 0.0 if disable_gt_explore else GT_EXPLORE_EPS
                        if rng.random() < eps:
                            declare = rng.random() < 0.5
                        else:
                            declare = rng.random() < prob
                        # Behavior prob includes the explore mixture — the
                        # PPO importance ratio needs the true sampling prob.
                        p_act = prob if declare else 1.0 - prob
                        behave = (1.0 - eps) * p_act + eps * 0.5
                        if owner in learner_set:
                            gt_steps[i].append(
                                (h8_vec, int(declare),
                                 float(np.log(max(behave, 1e-8))), p, owner))
                        gs.declare_grand_tichu(p, declare)
                    else:
                        gs.declare_grand_tichu(p, False)
                    any_changed = True

            # 2. Resolve Dragon Give (BATCHED)
            pass_active = []
            for i, gs in enumerate(games):
                if done[i]:
                    continue
                if gs.phase.name == "PASSING":
                    pass_active.append((i, gs, gs.current_player))

            pass_groups: dict[int, list] = {}
            for entry in pass_active:
                pass_groups.setdefault(seat_owners[entry[0]][entry[2]], []).append(entry)
            for owner, pass_active in sorted(pass_groups.items()):
                any_changed = True
                pass_policy = agents[owner].pass_policy
                if pass_policy is None:
                    for _, gs_pass, player_pass in pass_active:
                        hand = gs_pass.get_passing_hand(player_pass)
                        left, partner, right = rng.sample(hand, 3)
                        ok = gs_pass.pass_cards(player_pass, left, partner, right)
                        if not ok:
                            raise RuntimeError(
                                f"Passing phase rejected random choices for P{player_pass}"
                            )
                else:
                    hand_cards_np = np.array(
                        [gs_pass.get_passing_hand(player_pass) for _, gs_pass, player_pass in pass_active],
                        dtype=np.int64,
                    )
                    hand_binary = np.zeros(
                        (len(pass_active), PASS_OBS_SIZE),
                        dtype=np.float32,
                    )
                    rows = np.repeat(np.arange(len(pass_active)), hand_cards_np.shape[1])
                    cols = hand_cards_np.reshape(-1)
                    hand_binary[rows, cols] = 1.0
                    # Grand tichu declaration bits (relative, [56] = self)
                    for j, (_, gs_pass, player_pass) in enumerate(pass_active):
                        for k in range(4):
                            if gs_pass.tichu_declarations[(player_pass + k) % 4] == 2:
                                hand_binary[j, NUM_CARDS + k] = 1.0

                    with torch.no_grad():
                        hand_binary_t = torch.as_tensor(
                            hand_binary, dtype=torch.float32, device=device
                        )
                        hand_cards_t = torch.as_tensor(
                            hand_cards_np, dtype=torch.int64, device=device
                        )
                        choices_t, pass_lp_t = pass_policy.act(hand_binary_t, hand_cards_t)
                    choices_np = choices_t.cpu().numpy()

                    pass_lp_np = pass_lp_t.cpu().numpy()
                    if owner in learner_set:
                        for j, (gi, _gs, player_pass) in enumerate(pass_active):
                            pass_steps[gi].append(
                                (hand_binary[j], hand_cards_np[j],
                                 choices_np[j], pass_lp_np[j], player_pass, owner))

                    for j, (_, gs_pass, player_pass) in enumerate(pass_active):
                        idxs = choices_np[j]
                        left = hand_cards_np[j, int(idxs[0])]
                        partner = hand_cards_np[j, int(idxs[1])]
                        right = hand_cards_np[j, int(idxs[2])]
                        ok = gs_pass.pass_cards(
                            player_pass, int(left), int(partner), int(right)
                        )
                        if not ok:
                            raise RuntimeError(
                                f"Passing phase rejected sampled choices for P{player_pass}"
                            )

            dg_active = []
            for i, gs in enumerate(games):
                if not done[i] and gs.dragon_give_pending is not None and not gs.round_over:
                    dg_active.append((i, gs, gs.dragon_give_pending))

            dg_groups: dict[int, list] = {}
            for entry in dg_active:
                dg_groups.setdefault(seat_owners[entry[0]][entry[2]], []).append(entry)
            for owner, dg_active in sorted(dg_groups.items()):
                policy = agents[owner].policy
                any_changed = True
                t_spec_start = time.perf_counter()
                obs_np = np.stack([encode_static(x[1], x[2]) for x in dg_active])
                ps_np = np.stack([encode_pass_source_ids(x[1], x[2]) for x in dg_active])
                pt_np = np.stack([encode_pass_target_ids(x[1], x[2]) for x in dg_active])

                h_list = []
                hl_list = []
                for _, gs_dg, p_dg in dg_active:
                    h, hl = encode_history(gs_dg, p_dg)
                    h_list.append(h)
                    hl_list.append(hl)
                max_h = max(max(hl_list), 1)

                with torch.no_grad():
                    obs_batch = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                    ps_batch = torch.as_tensor(ps_np, dtype=torch.int64, device=device)
                    pt_batch = torch.as_tensor(pt_np, dtype=torch.int64, device=device)
                    h_batch = torch.as_tensor(np.stack([h[:max_h] for h in h_list]), dtype=torch.float32, device=device)
                    hl_batch = torch.as_tensor(np.array(hl_list, dtype=np.int32), device=device)

                    actor_fused, critic_fused = _actor_critic_features(
                        policy,
                        obs_batch, h_batch, hl_batch, ps_batch, pt_batch
                    )
                    dg_values = policy.value_head(critic_fused).squeeze(-1).cpu().numpy()
                    logits = policy.get_dragon_gift_logits(actor_fused)
                    dist = Categorical(logits=logits)
                    choices_t = dist.sample()
                    dg_logprobs = dist.log_prob(choices_t).cpu().numpy()
                    choices = choices_t.cpu().numpy()

                # Execute
                for j, (idx, gs_dg, p_dg) in enumerate(dg_active):
                    choice = int(choices[j])
                    target_abs = (p_dg + (1 if choice == 0 else 3)) % 4
                    options = gs_dg.get_dragon_give_options(p_dg)
                    if target_abs in options:
                        if owner in learner_set:
                            spec_steps[idx].append((
                                "dragon", obs_np[j], ps_np[j], pt_np[j],
                                h_list[j].astype(np.float16), hl_list[j],
                                choice, float(dg_logprobs[j]), float(dg_values[j]), p_dg,
                                owner,
                            ))
                        gs_dg.give_dragon_trick(p_dg, target_abs)
                    else:
                        gs_dg.give_dragon_trick(p_dg, rng.choice(options))
                    if gs_dg.round_over:
                        done[idx] = True
                specialist_seconds = time.perf_counter() - t_spec_start
                ROLLOUT_STATS["specialist_heads_s"] += specialist_seconds
                ROLLOUT_STATS["specialist_preplay_s"] += specialist_seconds
            
            if not any_changed:
                break
        ROLLOUT_STATS["immediate_decisions_s"] += (
            time.perf_counter() - immediate_started
        )

        if all(done):
            break

        n_done = sum(done)
        t_start = time.perf_counter()

        # Phase 2: collect observations + history
        active: list[
            tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, np.ndarray, int]
        ] = []
        # Pre-allocate obs buffer for this step to reuse
        obs_buffer = np.zeros(STATIC_OBS_SIZE, dtype=np.float32)
        
        for i, gs in enumerate(games):
            if done[i]:
                continue
            player = gs.current_player
            
            t0 = time.perf_counter()
            legal = gs.get_policy_legal_actions(player)
            ROLLOUT_STATS["legal_actions_s"] += time.perf_counter() - t0
            
            if not legal:
                if gs.round_over:
                    done[i] = True
                else:
                    # Failsafe: This should never happen with a perfect state machine.
                    # If it does, we log it and force-end the game to avoid hanging the entire batch.
                    print(f"      [critical] G{i} stalled (no legal actions, round not over). Force-ending.", flush=True)
                    done[i] = True
                continue

            t0 = time.perf_counter()
            obs = encode_static(gs, player, out=obs_buffer).copy()
            pass_src = encode_pass_source_ids(gs, player)
            pass_tgt = encode_pass_target_ids(gs, player)
            ROLLOUT_STATS["obs_encode_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            vecs = np.array(
                [_action_vec(a) for a in legal], dtype=np.float32
            )  # (n_legal, 58)
            ROLLOUT_STATS["action_vec_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            hist, hist_len = encode_history(gs, player)  # (200, 63), int
            ROLLOUT_STATS["history_encode_s"] += time.perf_counter() - t0

            active.append((i, player, obs, pass_src, pass_tgt, vecs, legal, hist, hist_len))

        if not active:
            break

        # Phase 3: smart batched forward pass, one batch family per agent
        t_fwd_start = time.perf_counter()
        active_owners = [seat_owners[entry[0]][entry[1]] for entry in active]

        action_idxs = np.zeros(len(active), dtype=np.int64)
        logprobs = np.zeros(len(active), dtype=np.float32)
        values = np.zeros(len(active), dtype=np.float32)

        buckets: list[tuple[object, list[int]]] = []
        for owner in sorted(set(active_owners)):
            owned = [idx for idx in range(len(active)) if active_owners[idx] == owner]
            # Sort by history length and number of legal actions for better bucketing
            if use_rollout_bucketing:
                order = sorted(
                    owned,
                    key=lambda idx: (active[idx][8], active[idx][5].shape[0]),
                )
                bucket_size = ROLLOUT_FORWARD_BUCKET
            else:
                order = owned
                bucket_size = len(owned)
            for start in range(0, len(order), bucket_size):
                buckets.append((agents[owner].policy, order[start : start + bucket_size]))

        for policy, bucket_idx in buckets:
            B = len(bucket_idx)
            
            # Extract bucket data
            b_obs = [active[idx][2] for idx in bucket_idx]
            b_pass_src = [active[idx][3] for idx in bucket_idx]
            b_pass_tgt = [active[idx][4] for idx in bucket_idx]
            b_vecs = [active[idx][5] for idx in bucket_idx]
            b_hist = [active[idx][7] for idx in bucket_idx]
            b_hl = [active[idx][8] for idx in bucket_idx]

            n_arr = np.array([v.shape[0] for v in b_vecs], dtype=np.int32)
            max_n = int(n_arr.max())
            max_h = max(max(b_hl), 1)

            # Pre-allocate padded arrays
            legal_pad = np.zeros((B, max_n, ACTION_DIM), dtype=np.uint8)
            hist_batch = np.zeros((B, max_h, PLAY_TOKEN_SIZE), dtype=np.float32)
            
            for j in range(B):
                if n_arr[j] > 0:
                    legal_pad[j, :n_arr[j]] = b_vecs[j]
                hist_batch[j] = b_hist[j][:max_h]

            # Single transfer to device
            obs_t = torch.as_tensor(np.stack(b_obs), dtype=torch.float32, device=device)
            pass_src_t = torch.as_tensor(np.stack(b_pass_src), dtype=torch.int64, device=device)
            pass_tgt_t = torch.as_tensor(np.stack(b_pass_tgt), dtype=torch.int64, device=device)
            # Explicitly cast uint8 to float32 before sending to GPU
            legal_pad_t = torch.as_tensor(legal_pad, dtype=torch.float32, device=device)
            n_legal_t = torch.as_tensor(n_arr, device=device)
            hist_t = torch.as_tensor(hist_batch, device=device)
            hist_len_t = torch.as_tensor(np.array(b_hl, dtype=np.int32), device=device)

            actions_t, logprobs_t, values_t = policy.act(
                obs_t, legal_pad_t, n_legal_t, hist_t, hist_len_t, pass_src_t, pass_tgt_t
            )

            # One device-to-host copy per output, not one sync per element.
            action_idxs[bucket_idx] = actions_t.cpu().numpy()
            logprobs[bucket_idx] = logprobs_t.cpu().numpy()
            values[bucket_idx] = values_t.cpu().numpy()
        ROLLOUT_STATS["forward_s"] += time.perf_counter() - t_fwd_start

        # Phase 4: execute and record
        t_env_start = time.perf_counter()
        for j, (game_idx, player, obs, pass_src, pass_tgt, vecs, legal, hist, hist_len) in enumerate(
            active
        ):
            gs = games[game_idx]
            action_idx = int(action_idxs[j])
            action = legal[action_idx] if action_idx < len(legal) else legal[0]

            # Track if the game actually progresses
            old_cp = gs.current_player
            old_hand_sum = sum(len(h) for h in gs.hands)
            old_trick_num = gs.trick_num
            old_passes = gs.trick.passes
            old_phase = gs.phase
            
            success = gs.play(player, action)
            if not success:
                from tichu_env.combinations import classify_cards
                test_combo = classify_cards(action[0])
                raise RuntimeError(
                    f"G{game_idx} rejected {action} for P{player}; "
                    f"classified={test_combo} wish={gs.mahjong_wish} "
                    f"leading={gs.trick.leading_combo}"
                )

            # Check for non-advancing state (real deadlock)
            if (gs.current_player == old_cp and 
                sum(len(h) for h in gs.hands) == old_hand_sum and 
                gs.trick_num == old_trick_num and
                gs.trick.passes == old_passes and
                gs.phase == old_phase and
                not gs.round_over and 
                gs.dragon_give_pending is None):
                raise RuntimeError(f"G{game_idx} deadlocked on {action}")

            if active_owners[j] in learner_set:
                ep_steps[game_idx].append(
                    (
                        obs,
                        pass_src,
                        pass_tgt,
                        vecs,
                        action_idx,
                        float(logprobs[j]),
                        float(values[j]),
                        player,
                        hist_len,
                        active_owners[j],
                    )
                )
            ROLLOUT_STATS["total_steps"] += 1
            # Specialist: Handle Mahjong Wish (BATCHED)

        mw_active = []
        for j, (game_idx, player, _, _, _, _, legal, _, _) in enumerate(active):
            gs = games[game_idx]
            action = legal[int(action_idxs[j])] if int(action_idxs[j]) < len(legal) else legal[0]
            if MAHJONG_INDEX in action[0] and not gs.round_over and gs.mahjong_wish is None:
                mw_active.append((game_idx, gs, player))

        mw_groups: dict[int, list] = {}
        for entry in mw_active:
            mw_groups.setdefault(seat_owners[entry[0]][entry[2]], []).append(entry)
        for owner, mw_active in sorted(mw_groups.items()):
            policy = agents[owner].policy
            t_spec_start = time.perf_counter()
            obs_np = np.stack([encode_static(x[1], x[2]) for x in mw_active])
            ps_np = np.stack([encode_pass_source_ids(x[1], x[2]) for x in mw_active])
            pt_np = np.stack([encode_pass_target_ids(x[1], x[2]) for x in mw_active])

            h_list = []
            hl_list = []
            for _, gs_mw, p_mw in mw_active:
                h, hl = encode_history(gs_mw, p_mw)
                h_list.append(h)
                hl_list.append(hl)
            max_h = max(max(hl_list), 1)

            with torch.no_grad():
                obs_mw = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                ps_mw = torch.as_tensor(ps_np, dtype=torch.int64, device=device)
                pt_mw = torch.as_tensor(pt_np, dtype=torch.int64, device=device)
                h_mw = torch.as_tensor(np.stack([h[:max_h] for h in h_list]), dtype=torch.float32, device=device)
                hl_mw = torch.as_tensor(np.array(hl_list, dtype=np.int32), device=device)

                actor_fused, critic_fused = _actor_critic_features(
                    policy,
                    obs_mw, h_mw, hl_mw, ps_mw, pt_mw
                )
                mw_values = policy.value_head(critic_fused).squeeze(-1).cpu().numpy()
                wish_logits = policy.get_mahjong_wish_logits(actor_fused)
                dist = Categorical(logits=wish_logits)
                wish_t = dist.sample()
                mw_logprobs = dist.log_prob(wish_t).cpu().numpy()
                wish_idxs = wish_t.cpu().numpy()

            for j, (idx, gs_mw, p_mw) in enumerate(mw_active):
                w_idx = int(wish_idxs[j])
                if owner in learner_set:
                    spec_steps[idx].append((
                        "wish", obs_np[j], ps_np[j], pt_np[j],
                        h_list[j].astype(np.float16), hl_list[j],
                        w_idx, float(mw_logprobs[j]), float(mw_values[j]), p_mw,
                        owner,
                    ))
                if w_idx == 0:
                    gs_mw.set_mahjong_wish(None)
                else:
                    gs_mw.set_mahjong_wish(int(w_idx + 1))
            specialist_seconds = time.perf_counter() - t_spec_start
            ROLLOUT_STATS["specialist_heads_s"] += specialist_seconds
            ROLLOUT_STATS["specialist_postplay_s"] += specialist_seconds

        for i, gs in enumerate(games):
            if not done[i] and gs.round_over:
                done[i] = True
        ROLLOUT_STATS["env_step_s"] += time.perf_counter() - t_env_start

    buffers = {
        learner: _compile_rollout_buffer(
            games,
            [[s[:-1] for s in steps if s[-1] == learner] for steps in ep_steps],
            [[s[:-1] for s in steps if s[-1] == learner] for steps in spec_steps],
            [[s[:-1] for s in steps if s[-1] == learner] for steps in gt_steps],
            [[s[:-1] for s in steps if s[-1] == learner] for steps in pass_steps],
            gt_probe=gt_probe,
            probe_h8=probe_h8 if gt_probe else None,
        )
        for learner in sorted(learner_set)
    }
    return buffers, ROLLOUT_STATS


def _compile_rollout_buffer(
    games: list[GameState],
    ep_steps: list[list[tuple]],
    spec_steps: list[list[tuple]],
    gt_steps: list[list[tuple]],
    pass_steps: list[list[tuple]],
    *,
    gt_probe: bool,
    probe_h8: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Compile one learner's recorded decisions into a flat PPO buffer."""
    # ── Compile flat buffer ──────────────────────────────────────────────
    t_buf_start = time.perf_counter()
    N = sum(len(steps) for steps in ep_steps)
    if N == 0:
        return {}

    obs_arr = np.zeros((N, STATIC_OBS_SIZE), dtype=np.float32)
    pass_src_arr = np.zeros((N, 3), dtype=np.int64)
    pass_tgt_arr = np.zeros((N, 3), dtype=np.int64)
    act_arr = np.zeros(N, dtype=np.int64)
    lp_arr = np.zeros(N, dtype=np.float32)
    val_arr = np.zeros(N, dtype=np.float32)
    ret_arr = np.zeros(N, dtype=np.float32)
    hist_lens_arr = np.zeros(N, dtype=np.int32)

    all_legal_vecs = [None] * N

    t_pass1 = time.perf_counter()
    spec_rows: dict[str, list[tuple]] = {"dragon": [], "wish": []}
    gt_rows: list[tuple] = []
    game_rewards: list[list[float]] = [None] * len(games)
    curr = 0
    for i, gs in enumerate(games):
        score_a, score_b = gs.round_scores
        diff = (score_a - score_b) / 400.0
        player_rewards = [diff if p in TEAM_A else -diff for p in range(4)]
        game_rewards[i] = player_rewards

        for obs, ps, pt, vecs, a_idx, lp, val, p_idx, h_len in ep_steps[i]:
            obs_arr[curr] = obs
            pass_src_arr[curr] = ps
            pass_tgt_arr[curr] = pt
            act_arr[curr] = a_idx
            lp_arr[curr] = lp
            val_arr[curr] = val
            ret_arr[curr] = player_rewards[p_idx]
            hist_lens_arr[curr] = h_len
            all_legal_vecs[curr] = vecs
            curr += 1

        for kind, obs, ps, pt, hist, h_len, a_idx, lp, val, p_idx in spec_steps[i]:
            spec_rows[kind].append(
                (obs, ps, pt, hist, h_len, a_idx, lp, val, player_rewards[p_idx])
            )

        for h8_vec, declare, blp, p_idx in gt_steps[i]:
            gt_rows.append((h8_vec, declare, blp, player_rewards[p_idx]))
    ROLLOUT_STATS["buffer_pass1_s"] += time.perf_counter() - t_pass1

    t_legal_concat = time.perf_counter()
    legal_flat = np.concatenate(all_legal_vecs, axis=0)
    legal_lens = np.array([v.shape[0] for v in all_legal_vecs], dtype=np.int32)
    ROLLOUT_STATS["buffer_legal_concat_s"] += time.perf_counter() - t_legal_concat

    t_hist_rebuild = time.perf_counter()
    shared_history = _compile_shared_history(games, ep_steps)
    ROLLOUT_STATS["buffer_history_rebuild_s"] += time.perf_counter() - t_hist_rebuild

    ROLLOUT_STATS["buffer_compile_s"] += time.perf_counter() - t_buf_start
    buffer = {
        "obs": obs_arr,
        "pass_source_ids": pass_src_arr,
        "pass_target_ids": pass_tgt_arr,
        "legal_flat": legal_flat,
        "legal_lens": legal_lens,
        "action_idxs": act_arr,
        "log_probs": lp_arr,
        "values": val_arr,
        "returns": ret_arr,
        "advantages": ret_arr - val_arr,
        "history_lens": hist_lens_arr,
    }
    buffer.update(shared_history)
    for kind, rows in spec_rows.items():
        buffer.update(_compile_spec_buffer(kind, rows))
    if gt_rows:
        buffer["gt_h8"] = np.stack([r[0] for r in gt_rows]).astype(np.float32)
        buffer["gt_declare"] = np.array([r[1] for r in gt_rows], dtype=np.int64)
        buffer["gt_log_probs"] = np.array([r[2] for r in gt_rows], dtype=np.float32)
        buffer["gt_returns"] = np.array([r[3] for r in gt_rows], dtype=np.float32)
    if not gt_probe and any(pass_steps):
        # Pass-net harvest: every pass decision, credited with the passer's
        # realized team round return. Unlike the retired V(post)-V(pre)
        # proxy this is sensitive to the left/partner/right ASSIGNMENT and
        # to the strength donated to opponents; b(pass_obs) absorbs hand
        # quality in ppo_update_passing.
        pass_rows = [
            (hb, hc, ch, lp, game_rewards[i][p])
            for i in range(len(games))
            for hb, hc, ch, lp, p in pass_steps[i]
        ]
        buffer["pass_train_obs"] = np.stack(
            [r[0] for r in pass_rows]).astype(np.float32)
        buffer["pass_train_hand_cards"] = np.stack(
            [r[1] for r in pass_rows]).astype(np.int64)
        buffer["pass_train_choices"] = np.stack(
            [r[2] for r in pass_rows]).astype(np.int64)
        buffer["pass_train_log_probs"] = np.stack(
            [r[3] for r in pass_rows]).astype(np.float32)
        buffer["pass_train_returns"] = np.array(
            [r[4] for r in pass_rows], dtype=np.float32)
    if gt_probe:
        n_groups = len(games) // 5
        declared_ret = np.array(
            [game_rewards[5 * j + 1 + k][k]
             for j in range(n_groups) for k in range(4)],
            dtype=np.float32)
        quiet_ret = np.array(
            [game_rewards[5 * j][k]
             for j in range(n_groups) for k in range(4)],
            dtype=np.float32)
        buffer["gt_probe_h8"] = probe_h8
        buffer["gt_probe_diff"] = declared_ret - quiet_ret
        buffer["gt_probe_quiet"] = quiet_ret

        # Pass-net practice channel: every pass decision in a declared branch,
        # rewarded with that seat's realized team return minus the same deal's
        # quiet-control return. The control's outcome does not depend on this
        # branch's passes, so it is a valid baseline that cancels whole-deal
        # luck — a strictly sharper baseline than the adapt stream's
        # b(pass_obs), and the pass net's densest source of declared-round
        # practice. Quiet branches are not trained on; quiet passing is
        # covered by the adapt stream's realized-return harvest above.
        pass_rows = [
            (hb, hc, ch, lp,
             game_rewards[5 * j + 1 + k][p] - game_rewards[5 * j][p])
            for j in range(n_groups) for k in range(4)
            for hb, hc, ch, lp, p in pass_steps[5 * j + 1 + k]
        ]
        if pass_rows:
            buffer["pass_train_obs"] = np.stack(
                [r[0] for r in pass_rows]).astype(np.float32)
            buffer["pass_train_hand_cards"] = np.stack(
                [r[1] for r in pass_rows]).astype(np.int64)
            buffer["pass_train_choices"] = np.stack(
                [r[2] for r in pass_rows]).astype(np.int64)
            buffer["pass_train_log_probs"] = np.stack(
                [r[3] for r in pass_rows]).astype(np.float32)
            buffer["pass_train_rewards"] = np.array(
                [r[4] for r in pass_rows], dtype=np.float32)
    return buffer


# ── Worker entry point ───────────────────────────────────────────────────────

_WORKER_MODELS = {}
_WORKER_CUDA_ALLOCATED_FLOORS: dict[str, int] = {}


def _check_worker_cuda_memory(device: torch.device) -> dict[str, float | int]:
    """Inspect one worker's CUDA allocator and conditionally return its cache.

    ``memory_allocated`` tracks live tensors; sustained growth there can be a
    real reference leak. ``memory_reserved - memory_allocated`` is reusable
    cache owned by this process.  Releasing that cache is useful only when
    global device headroom is low, and avoids slowing the healthy fast path.
    """
    started = time.perf_counter()
    stats: dict[str, float | int] = {
        "worker_cuda_check_s": 0.0,
        "worker_cuda_check_count": 0,
        "worker_cuda_allocated_mb": 0.0,
        "worker_cuda_reserved_mb": 0.0,
        "worker_cuda_cached_mb": 0.0,
        "worker_cuda_free_before_min_mb": 0.0,
        "worker_cuda_free_after_min_mb": 0.0,
        "worker_cuda_cache_released_mb": 0.0,
        "worker_cuda_cleanup_count": 0,
        "worker_cuda_leak_suspected_count": 0,
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        stats["worker_cuda_check_s"] = time.perf_counter() - started
        return stats

    allocated = int(torch.cuda.memory_allocated(device))
    reserved_before = int(torch.cuda.memory_reserved(device))
    free_before, total = (
        int(value) for value in torch.cuda.mem_get_info(device)
    )
    cached_before = max(0, reserved_before - allocated)

    device_key = str(device)
    allocated_floor = _WORKER_CUDA_ALLOCATED_FLOORS.get(device_key)
    if allocated_floor is None or allocated < allocated_floor:
        allocated_floor = allocated
        _WORKER_CUDA_ALLOCATED_FLOORS[device_key] = allocated
    leak_tolerance = max(
        WORKER_CUDA_LIVE_GROWTH_TOLERANCE_BYTES,
        allocated_floor // 4,
    )
    leak_suspected = allocated > allocated_floor + leak_tolerance

    # Four GiB is the measured T4 target.  On smaller devices, use 30% of the
    # device rather than making an impossible fixed-headroom demand.
    target_free = min(WORKER_CUDA_MIN_FREE_BYTES, int(total * 0.30))
    cleanup = (
        (WORKER_CUDA_ALWAYS_RELEASE or free_before < target_free)
        and cached_before >= WORKER_CUDA_MIN_RELEASABLE_BYTES
    )
    if cleanup:
        torch.cuda.empty_cache()

    reserved_after = int(torch.cuda.memory_reserved(device))
    free_after, _ = (int(value) for value in torch.cuda.mem_get_info(device))
    cached_after = max(0, reserved_after - allocated)
    released = max(0, reserved_before - reserved_after)

    mib = float(2**20)
    stats.update(
        {
            "worker_cuda_check_count": 1,
            "worker_cuda_allocated_mb": allocated / mib,
            "worker_cuda_reserved_mb": reserved_after / mib,
            "worker_cuda_cached_mb": cached_after / mib,
            "worker_cuda_free_before_min_mb": free_before / mib,
            "worker_cuda_free_after_min_mb": free_after / mib,
            "worker_cuda_cache_released_mb": released / mib,
            "worker_cuda_cleanup_count": int(cleanup),
            "worker_cuda_leak_suspected_count": int(leak_suspected),
            "worker_cuda_check_s": time.perf_counter() - started,
        }
    )
    if cleanup or leak_suspected:
        print(
            f"  [worker cuda] pid={os.getpid()} "
            f"allocated={allocated / mib:.0f}MiB "
            f"reserved={reserved_before / mib:.0f}->{reserved_after / mib:.0f}MiB "
            f"device_free={free_before / mib:.0f}->{free_after / mib:.0f}MiB "
            f"released={released / mib:.0f}MiB "
            f"live_growth_alert={leak_suspected}",
            flush=True,
        )
    return stats

def _worker_collect(
    args: tuple,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    global _WORKER_MODELS
    reset_rollout_stats()
    worker_started = time.perf_counter()
    (
        weights_np,
        gt_weights_np,
        pass_weights_np,
        n_games,
        device_str,
        n_torch_threads,
        encoder_cfg,
        model_cfg,
        use_rollout_bucketing,
        with_calling,
        gt_probe,
        disable_gt_explore,
        seed,
    ) = args
    torch.set_num_threads(n_torch_threads)
    torch.manual_seed(seed if seed is not None else os.getpid())
    device = torch.device(device_str)
    setup_started = time.perf_counter()

    policy, gt_policy, pass_policy = _build_worker_agent(
        weights_np, gt_weights_np, pass_weights_np, encoder_cfg, model_cfg, device
    )

    worker_setup_seconds = time.perf_counter() - setup_started
    buffer, stats = _collect_batched(
        policy, n_games, device,
        use_rollout_bucketing=use_rollout_bucketing,
        with_calling=with_calling,
        gt_policy=gt_policy,
        pass_policy=pass_policy,
        gt_probe=gt_probe,
        disable_gt_explore=disable_gt_explore,
        seed=seed,
    )
    stats.update(_check_worker_cuda_memory(device))
    stats["worker_setup_s"] = worker_setup_seconds
    stats["worker_wall_s"] = time.perf_counter() - worker_started
    stats["worker_count"] = 1
    return buffer, stats


def _build_worker_agent(
    weights_np: dict,
    gt_weights_np: dict | None,
    pass_weights_np: dict | None,
    encoder_cfg: dict,
    model_cfg: dict,
    device: torch.device,
    slot: object = None,
) -> tuple:
    """Rebuild (policy, gt_policy, pass_policy) from dispatched weights.

    Modules are cached per process. ``slot`` separates agents that share an
    architecture, so loading one agent's weights never overwrites another's.
    """
    # The history norms are whatever the dispatched weights contain, so a
    # caller's partial config can never build the wrong layout for them.
    encoder_cfg = {**encoder_cfg, "final_norm": "history_encoder.encoder.norm.weight" in weights_np}
    model_cfg = {**model_cfg, "history_pool_norm": "history_pool_norm.weight" in weights_np}
    # Single-agent keys keep their historical shape; agents get a slot tag.
    tag = () if slot is None else (slot,)
    policy_key = (
        "policy",
        *tag,
        tuple(sorted(encoder_cfg.items())),
        tuple(sorted(model_cfg.items())),
    )
    if policy_key not in _WORKER_MODELS:
        from training.model_history import TichuPolicyHistory, TransformerHistoryEncoder
        encoder = TransformerHistoryEncoder(**encoder_cfg)
        _WORKER_MODELS[policy_key] = TichuPolicyHistory(
            encoder, **model_cfg
        ).to(device)
    
    policy = _WORKER_MODELS[policy_key].to(device)
    model_state = {k: torch.from_numpy(v) for k, v in weights_np.items()}
    from training.checkpoints import _resize_wish_head_for_state_dict

    _resize_wish_head_for_state_dict(policy, model_state, device, "rollout worker")
    policy.load_state_dict(model_state)
    policy.eval()

    if gt_weights_np is not None:
        # Reconstruct the actual dispatched architecture, not CLI defaults.
        gt_cfg = {
            "hidden_dim": gt_weights_np["adv_head.weight"].shape[1],
            "n_layers": sum(
                key.startswith("trunk.") and key.endswith(".weight")
                for key in gt_weights_np
            ),
        }
        gt_key = ("gt_policy", *tag, tuple(sorted(gt_cfg.items())))
        if gt_key not in _WORKER_MODELS:
            from training.model_gt import TichuGrandTichuPolicy
            _WORKER_MODELS[gt_key] = TichuGrandTichuPolicy(**gt_cfg).to(device)
        gt_policy = _WORKER_MODELS[gt_key].to(device)
        gt_policy.load_state_dict({k: torch.from_numpy(v) for k, v in gt_weights_np.items()})
        gt_policy.eval()
    else:
        gt_policy = None

    if pass_weights_np is not None:
        pass_cfg = {
            "hidden_dim": pass_weights_np["pointer_head.weight"].shape[1],
            "n_layers": sum(
                key.startswith("trunk.") and key.endswith(".weight")
                for key in pass_weights_np
            ),
            "card_embed_dim": pass_weights_np["card_embed.weight"].shape[1],
        }
        pass_key = ("pass_policy", *tag, tuple(sorted(pass_cfg.items())))
        if pass_key not in _WORKER_MODELS:
            from training.model_passing import TichuPassingPolicy
            _WORKER_MODELS[pass_key] = TichuPassingPolicy(**pass_cfg).to(device)
        pass_policy = _WORKER_MODELS[pass_key].to(device)
        pass_policy.load_state_dict({k: torch.from_numpy(v) for k, v in pass_weights_np.items()})
        pass_policy.eval()
    else:
        pass_policy = None
    return policy, gt_policy, pass_policy


def _agent_payload(agent: RolloutAgent, encoder_cfg: dict, model_cfg: dict) -> tuple:
    """CPU copies of one agent's weights, safe to queue while training mutates it."""

    def copy_state(module):
        if module is None:
            return None
        return {
            k: v.detach().to("cpu", copy=True).numpy()
            for k, v in module.state_dict().items()
        }

    return (
        copy_state(agent.policy),
        copy_state(agent.gt_policy),
        copy_state(agent.pass_policy),
        encoder_cfg,
        model_cfg,
    )


def _worker_collect_multi(args: tuple) -> tuple[dict[int, dict[str, np.ndarray]], dict]:
    reset_rollout_stats()
    worker_started = time.perf_counter()
    (
        payloads,
        seat_owners,
        learners,
        device_str,
        n_torch_threads,
        use_rollout_bucketing,
        with_calling,
        disable_gt_explore,
        seed,
    ) = args
    torch.set_num_threads(n_torch_threads)
    torch.manual_seed(seed if seed is not None else os.getpid())
    device = torch.device(device_str)
    setup_started = time.perf_counter()
    agents = [
        RolloutAgent(*_build_worker_agent(
            weights, gt_weights, pass_weights, enc_cfg, mdl_cfg, device, slot=("agent", slot)
        ))
        for slot, (weights, gt_weights, pass_weights, enc_cfg, mdl_cfg) in enumerate(payloads)
    ]
    worker_setup_seconds = time.perf_counter() - setup_started
    buffers, stats = _collect_batched_multi(
        agents, len(seat_owners), device,
        seat_owners=seat_owners,
        learners=learners,
        use_rollout_bucketing=use_rollout_bucketing,
        with_calling=with_calling,
        disable_gt_explore=disable_gt_explore,
        seed=seed,
    )
    stats.update(_check_worker_cuda_memory(device))
    stats["worker_setup_s"] = worker_setup_seconds
    stats["worker_wall_s"] = time.perf_counter() - worker_started
    stats["worker_count"] = 1
    return buffers, dict(stats)


def collect_rollouts_multi(
    agents: list[RolloutAgent],
    agent_cfgs: list[tuple[dict, dict]],
    seat_owners: list[tuple[int, int, int, int]],
    learners: tuple[int, ...],
    device: torch.device,
    pool=None,
    n_workers: int = 1,
    use_rollout_bucketing: bool = True,
    with_calling: bool = True,
    disable_gt_explore: bool = False,
    seed: int | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Collect one batch of mixed-agent games; returns a buffer per learner.

    ``agent_cfgs[k]`` is the (encoder_cfg, model_cfg) needed to rebuild agent
    ``k`` inside a worker process. Games are split across workers in
    contiguous slices of ``seat_owners``.
    """
    reset_rollout_stats()
    learners = tuple(sorted(set(learners)))
    if pool is None or n_workers <= 1:
        worker_started = time.perf_counter()
        buffers, stats = _collect_batched_multi(
            agents, len(seat_owners), device,
            seat_owners=seat_owners,
            learners=learners,
            use_rollout_bucketing=use_rollout_bucketing,
            with_calling=with_calling,
            disable_gt_explore=disable_gt_explore,
            seed=seed,
        )
        stats["worker_wall_s"] = time.perf_counter() - worker_started
        stats["worker_count"] = 1
        stats["worker_wall_max_s"] = stats["worker_wall_s"]
        for k, v in stats.items():
            ROLLOUT_STATS[k] = v
        _print_rollout_stats()
        return buffers

    payloads = [
        _agent_payload(agent, *cfgs) for agent, cfgs in zip(agents, agent_cfgs)
    ]
    n_torch_threads = max(1, os.cpu_count() // n_workers)
    base, rem = divmod(len(seat_owners), n_workers)
    sizes = [base + (1 if i < rem else 0) for i in range(n_workers)]
    args = []
    start = 0
    for i, size in enumerate(sizes):
        if size == 0:
            continue
        args.append((
            payloads,
            list(seat_owners[start : start + size]),
            learners,
            str(device),
            n_torch_threads,
            use_rollout_bucketing,
            with_calling,
            disable_gt_explore,
            None if seed is None else seed + i,
        ))
        start += size
    results = pool.map(_worker_collect_multi, args)
    # Device-free memory is global, not additive: keep the minimum seen.
    cuda_min_keys = {"worker_cuda_free_before_min_mb", "worker_cuda_free_after_min_mb"}
    for _, stats in results:
        for k, v in stats.items():
            if isinstance(v, (int, float)) and k not in cuda_min_keys:
                ROLLOUT_STATS[k] = ROLLOUT_STATS.get(k, 0) + v
    for key in cuda_min_keys:
        values = [float(st[key]) for _, st in results if st.get(key, 0)]
        if values:
            ROLLOUT_STATS[key] = min(values)
    starts = np.cumsum([0] + [len(a[1]) for a in args[:-1]]).tolist()
    merged = {}
    for learner in learners:
        kept = [(r[0][learner], start) for r, start in zip(results, starts) if r[0].get(learner)]
        merged[learner] = _merge_buffers(
            [buffer for buffer, _ in kept], [start for _, start in kept]
        )
    _print_rollout_stats()
    return merged


# ── Public API ───────────────────────────────────────────────────────────────


def collect_rollouts_history(
    policy,
    n_games: int,
    device: torch.device,
    pool=None,
    n_workers: int = 1,
    encoder_cfg: dict | None = None,
    model_cfg: dict | None = None,
    use_rollout_bucketing: bool = True,
    with_calling: bool = False,
    gt_policy=None,
    pass_policy=None,
    block: bool = True,
    gt_probe: bool = False,
    disable_gt_explore: bool = False,
    seed: int | None = None,
) -> dict[str, np.ndarray] | mp.pool.AsyncResult:
    """Collect self-play rollouts. Supports both single-process and multi-process modes.
    
    In multi-process mode (pool is not None), this function returns an AsyncResult
    immediately (if block=False), allowing for APPO pipelining where the main 
    process can run PPO updates while workers collect the next batch of games.
    """
    reset_rollout_stats()
    if pool is None or n_workers <= 1:
        worker_started = time.perf_counter()
        rollout, stats = _collect_batched(
            policy, n_games, device,
            use_rollout_bucketing=use_rollout_bucketing,
            with_calling=with_calling,
            gt_policy=gt_policy,
            pass_policy=pass_policy,
            gt_probe=gt_probe,
            disable_gt_explore=disable_gt_explore,
            seed=seed,
        )
        stats["worker_wall_s"] = time.perf_counter() - worker_started
        stats["worker_count"] = 1
        stats["worker_wall_max_s"] = stats["worker_wall_s"]
        # Update global stats for printing
        for k, v in stats.items():
            ROLLOUT_STATS[k] = v
        _print_rollout_stats()
        return rollout

    dispatch_started = time.perf_counter()
    # Pool task serialization happens asynchronously. CPU tensors must be
    # copied too, so subsequent PPO updates cannot mutate queued weights.
    weights_np = {
        k: v.detach().to("cpu", copy=True).numpy()
        for k, v in policy.state_dict().items()
    }
    gt_weights_np = None
    if gt_policy is not None:
        gt_weights_np = {
            k: v.detach().to("cpu", copy=True).numpy()
            for k, v in gt_policy.state_dict().items()
        }
    pass_weights_np = None
    if pass_policy is not None:
        pass_weights_np = {
            k: v.detach().to("cpu", copy=True).numpy()
            for k, v in pass_policy.state_dict().items()
        }

    device_str = str(device)
    n_torch_threads = max(1, os.cpu_count() // n_workers)
    enc_cfg = encoder_cfg or {}
    mdl_cfg = model_cfg or {}

    print(f"  [collecting] Starting collection of {n_games} games across {n_workers} workers...", flush=True)

    if gt_probe:
        # Split whole groups so each worker keeps all 5 branches of its deals.
        base, rem = divmod(n_games // 5, n_workers)
        sizes = [5 * (base + (1 if i < rem else 0)) for i in range(n_workers)]
    else:
        base, rem = divmod(n_games, n_workers)
        sizes = [base + (1 if i < rem else 0) for i in range(n_workers)]
    sizes = [s for s in sizes if s > 0]

    # Check for multiple GPUs
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0

    args = []
    for i, s in enumerate(sizes):
        # Assign worker i to GPU (i % n_gpus) if multiple GPUs exist
        worker_device = device_str
        if n_gpus > 1:
            worker_device = f"cuda:{i % n_gpus}"

        args.append((
            weights_np,
            gt_weights_np,
            pass_weights_np,
            s,
            worker_device,
            n_torch_threads,
            enc_cfg,
            mdl_cfg,
            use_rollout_bucketing,
            with_calling,
            gt_probe,
            disable_gt_explore,
            None if seed is None else seed + i,
        ))

    result = pool.map_async(_worker_collect, args)
    ROLLOUT_STATS["main_dispatch_s"] = (
        time.perf_counter() - dispatch_started
    )

    if not block:
        return result

    return finish_rollout(result, device)


def finish_rollout(result: mp.pool.AsyncResult | dict[str, np.ndarray], device: torch.device) -> dict[str, np.ndarray]:
    """Wait for, merge, and profile a rollout result."""
    if isinstance(result, dict):
        return result

    # Heartbeat while waiting for workers
    start_wait = time.perf_counter()
    last_heartbeat = start_wait
    while not result.ready():
        now = time.perf_counter()
        if now - last_heartbeat > 10.0:
            stats = get_system_stats(device)
            elapsed = now - start_wait
            print(f"  [collecting] {elapsed:.0f}s elapsed | cpu={stats['cpu_load_1m']} | ram={stats['ram_avail_gb']}GB", flush=True)
            last_heartbeat = now
        time.sleep(0.5)
    ROLLOUT_STATS["main_wait_s"] = time.perf_counter() - start_wait

    # 1. Measure IPC overhead (Unpickling/Transfer)
    t_ipc_start = time.perf_counter()
    results = result.get()
    ROLLOUT_STATS["main_ipc_s"] = time.perf_counter() - t_ipc_start

    buffers = [r[0] for r in results if r[0]]
    worker_stats = [r[1] for r in results]
    if worker_stats:
        ROLLOUT_STATS["worker_wall_max_s"] = max(
            float(stats.get("worker_wall_s", 0.0)) for stats in worker_stats
        )

    # Aggregate worker stats. Device-free memory is global rather than
    # additive, so retain the minimum observed by any worker.
    cuda_min_keys = {
        "worker_cuda_free_before_min_mb",
        "worker_cuda_free_after_min_mb",
    }
    for ws in worker_stats:
        for k, v in ws.items():
            if k in cuda_min_keys:
                continue
            ROLLOUT_STATS[k] += v
    for key in cuda_min_keys:
        values = [float(ws[key]) for ws in worker_stats if ws.get(key, 0)]
        if values:
            ROLLOUT_STATS[key] = min(values)

    # 2. Measure Merge/Concatenation in main process
    t_merge_start = time.perf_counter()
    final_rollout = _merge_buffers(buffers)
    ROLLOUT_STATS["main_merge_s"] = time.perf_counter() - t_merge_start
    _print_rollout_stats()
    return final_rollout


def _merge_buffers(
    buffers: list[dict[str, np.ndarray]],
    game_starts: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Concatenate worker buffers, re-basing shared-history references.

    ``game_starts[i]`` is buffer ``i``'s first global game index; without it,
    games are numbered consecutively across the given buffers.
    """
    final_rollout: dict[str, np.ndarray] = {}
    if not buffers:
        return final_rollout

    history_keys = {
        "history_tokens",
        "history_players",
        "history_game_offsets",
        "history_game_ids",
        "history_observers",
    }
    if "history_tokens" in buffers[0]:
        token_base = 0
        game_base = 0
        merged_offsets = [0]
        merged_game_ids = []
        for index, buffer in enumerate(buffers):
            offsets = buffer["history_game_offsets"]
            if game_starts is not None:
                # Pad offsets for games owned by skipped (empty) buffers.
                gap = game_starts[index] - game_base
                merged_offsets.extend([token_base] * gap)
                game_base = game_starts[index]
            merged_offsets.extend((offsets[1:] + token_base).tolist())
            merged_game_ids.append(buffer["history_game_ids"] + game_base)
            token_base += len(buffer["history_tokens"])
            game_base += len(offsets) - 1
        final_rollout.update(
            {
                "history_tokens": np.concatenate(
                    [buffer["history_tokens"] for buffer in buffers],
                    axis=0,
                ),
                "history_players": np.concatenate(
                    [buffer["history_players"] for buffer in buffers],
                ),
                "history_game_offsets": np.asarray(
                    merged_offsets,
                    dtype=np.int32,
                ),
                "history_game_ids": np.concatenate(merged_game_ids),
                "history_observers": np.concatenate(
                    [buffer["history_observers"] for buffer in buffers],
                ),
            }
        )
    else:
        # Compatibility for tests and external callers that still provide the
        # old dense transition history tensor.
        history_keys = {"history"}
        max_hist = max(buffer["history"].shape[1] for buffer in buffers)
        total_steps = sum(buffer["history"].shape[0] for buffer in buffers)
        merged_history = np.zeros(
            (total_steps, max_hist, PLAY_TOKEN_SIZE),
            dtype=np.float16,
        )
        curr = 0
        for buffer in buffers:
            history = buffer["history"]
            n_steps = history.shape[0]
            merged_history[curr : curr + n_steps, : history.shape[1]] = history
            curr += n_steps
        final_rollout["history"] = merged_history

    # 2. Concatenate all other arrays directly
    for k in buffers[0].keys():
        if k in history_keys:
            continue
        final_rollout[k] = np.concatenate([b[k] for b in buffers], axis=0)
    return final_rollout

def _print_rollout_stats():
    worker_keys = (
        "worker_setup_s",
        "worker_cuda_check_s",
        "initialization_s",
        "immediate_decisions_s",
        "legal_actions_s",
        "obs_encode_s",
        "action_vec_s",
        "history_encode_s",
        "forward_s",
        "env_step_s",
        "buffer_compile_s",
    )
    nested_keys = (
        "specialist_preplay_s",
        "specialist_postplay_s",
        "buffer_pass1_s",
        "buffer_legal_concat_s",
        "buffer_history_rebuild_s",
    )
    main_keys = (
        "main_dispatch_s",
        "main_wait_s",
        "main_ipc_s",
        "main_merge_s",
    )
    worker_wall = float(ROLLOUT_STATS["worker_wall_s"])
    worker_count = max(int(ROLLOUT_STATS["worker_count"]), 1)
    profiled_worker = sum(float(ROLLOUT_STATS[key]) for key in worker_keys)
    if worker_wall <= 0.0:
        worker_wall = profiled_worker
    main_wall = sum(float(ROLLOUT_STATS[key]) for key in main_keys)
    if worker_wall + main_wall == 0.0:
        return

    steps = ROLLOUT_STATS["total_steps"]
    worker_mean = worker_wall / worker_count
    worker_max = float(ROLLOUT_STATS["worker_wall_max_s"])
    print(
        f"\n  [rollout profile] {steps} steps | "
        f"workers: sum={worker_wall:.2f}s mean={worker_mean:.2f}s "
        f"max={worker_max:.2f}s | main={main_wall:.2f}s",
        flush=True,
    )

    print("    Worker timeline (non-overlapping):", flush=True)
    for key in worker_keys:
        value = float(ROLLOUT_STATS[key])
        percent = value / worker_wall * 100.0 if worker_wall else 0.0
        print(f"      {key:<28}: {value:>7.2f}s ({percent:>5.1f}%)", flush=True)
    unprofiled = max(0.0, worker_wall - profiled_worker)
    percent = unprofiled / worker_wall * 100.0 if worker_wall else 0.0
    print(f"      {'worker_unprofiled_s':<28}: {unprofiled:>7.2f}s ({percent:>5.1f}%)", flush=True)

    print("    Nested diagnostics (already included above):", flush=True)
    for key in nested_keys:
        print(f"      {key:<28}: {ROLLOUT_STATS[key]:>7.2f}s", flush=True)

    print("    Main-process timeline:", flush=True)
    for key in main_keys:
        value = float(ROLLOUT_STATS[key])
        percent = value / main_wall * 100.0 if main_wall else 0.0
        print(f"      {key:<28}: {value:>7.2f}s ({percent:>5.1f}%)", flush=True)
    cuda_checks = int(ROLLOUT_STATS["worker_cuda_check_count"])
    if cuda_checks:
        print(
            "    Worker CUDA after rollout: "
            f"allocated={ROLLOUT_STATS['worker_cuda_allocated_mb']:.0f}MiB "
            f"reserved={ROLLOUT_STATS['worker_cuda_reserved_mb']:.0f}MiB "
            f"cached={ROLLOUT_STATS['worker_cuda_cached_mb']:.0f}MiB | "
            "device free(min)="
            f"{ROLLOUT_STATS['worker_cuda_free_before_min_mb']:.0f}->"
            f"{ROLLOUT_STATS['worker_cuda_free_after_min_mb']:.0f}MiB | "
            f"released={ROLLOUT_STATS['worker_cuda_cache_released_mb']:.0f}MiB "
            f"cleanups={int(ROLLOUT_STATS['worker_cuda_cleanup_count'])}/"
            f"{cuda_checks} "
            "live-growth-alerts="
            f"{int(ROLLOUT_STATS['worker_cuda_leak_suspected_count'])} | "
            f"check={ROLLOUT_STATS['worker_cuda_check_s']:.4f}s",
            flush=True,
        )
    print(flush=True)
