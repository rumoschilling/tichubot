"""PPO update step for the history-aware policy.

Expects the buffer format produced by rollout_history.py:
  obs           (N, 196)               float32
  legal_flat    (L, 58)                float32
  legal_lens    (N,)                   int32
  action_idxs   (N,)                   int64
  log_probs     (N,)                   float32
  values        (N,)                   float32
  returns       (N,)                   float32
  advantages    (N,)                   float32
  history_tokens       (H, 59)         float16   — objective rows, once per game
  history_players      (H,)            int8      — absolute actor for each row
  history_game_offsets (G+1,)          int32
  history_game_ids     (N,)            int32
  history_observers    (N,)            int8
  history_lens         (N,)            int32

Legacy callers may still provide ``history`` as a dense
``(N, max_hist, 63)`` tensor.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from training.mmd import kl_to_magnet, magnet_log_probs
from torch.distributions import Categorical

from training.config import TrainConfig
from tichu_env.encoding import PLAY_TOKEN_SIZE


class _PpoStageTimer:
    """Low-overhead paired stage timing for controlled runtime experiments.

    CUDA events are recorded without synchronizing between stages, preserving
    the normal queued execution. ``ppo_update_history`` already synchronizes
    once at the end of an update, after which event durations are safe to
    read. CPU runs fall back to ``perf_counter``.
    """

    def __init__(self, device: torch.device, enabled: bool):
        self.device = device
        self.enabled = enabled
        self._cuda_events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = defaultdict(list)
        self._cpu_seconds: dict[str, float] = defaultdict(float)

    def start(self):
        if not self.enabled:
            return None
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, name: str, started) -> None:
        if started is None:
            return
        if self.device.type == "cuda":
            finished = torch.cuda.Event(enable_timing=True)
            finished.record()
            self._cuda_events[name].append((started, finished))
            return
        self._cpu_seconds[name] += time.perf_counter() - started

    def seconds(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        if self.device.type == "cuda":
            return {
                name: sum(
                    started.elapsed_time(finished)
                    for started, finished in pairs
                )
                / 1000.0
                for name, pairs in self._cuda_events.items()
            }
        return dict(self._cpu_seconds)


def _legal_offsets(legal_lens: np.ndarray) -> np.ndarray:
    """Start offset of each step's legal actions in the flat legal buffer."""
    return np.concatenate([[0], np.cumsum(legal_lens[:-1])]).astype(np.int64)


def _ppo_order(
    indices: np.ndarray,
    rollout: dict[str, np.ndarray],
    legal_lens: np.ndarray,
    microbatch_size: int,
    use_bucketing: bool,
    shuffle_bucket_microbatches: bool,
) -> np.ndarray:
    """Build an epoch order while retaining padding-efficient microbatches.

    Bucketing sorts similarly shaped rows together. If parameters step before
    the full rollout is consumed, keeping the buckets monotonic would make the
    optimizer sequence systematically homogeneous. Shuffling the completed
    microbatches preserves their padding efficiency while mixing their order.
    """
    np.random.shuffle(indices)
    order = indices.copy()
    if not use_bucketing:
        return order

    order = order[np.argsort(rollout["history_lens"][order], kind="stable")]
    order = order[np.argsort(legal_lens[order], kind="stable")]
    if not shuffle_bucket_microbatches:
        return order

    chunks = [
        order[start : start + microbatch_size]
        for start in range(0, len(order), microbatch_size)
    ]
    np.random.shuffle(chunks)
    return np.concatenate(chunks) if chunks else order


def _pad_legal_eager(
    legal_storage: torch.Tensor,
    offsets_storage: torch.Tensor,
    b_idx_t: torch.Tensor,
    b_lens: torch.Tensor,
) -> torch.Tensor:
    """Vectorized gather from a flat legal-action tensor into a padded batch."""
    max_n = max(int(b_lens.max().item()), 1)
    col_range = torch.arange(max_n, dtype=torch.int64, device=b_lens.device)
    b_offsets = offsets_storage[b_idx_t]
    valid = col_range.unsqueeze(0) < b_lens.to(torch.int64).unsqueeze(1)
    flat_idx = b_offsets.unsqueeze(1) + col_range.unsqueeze(0)
    flat_idx = torch.where(valid, flat_idx, torch.zeros_like(flat_idx))
    legal_pad = legal_storage[flat_idx].to(torch.float32)
    return legal_pad * valid.unsqueeze(-1)


def _pad_legal_lazy(
    legal_storage: np.ndarray,
    offsets_storage: np.ndarray,
    b_idx: np.ndarray,
    legal_lens: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """CPU reference-style gather used when rollout buffers stay off-device."""
    b_lens_np = legal_lens[b_idx]
    max_n = max(int(b_lens_np.max()), 1)
    legal_batch_cpu = np.zeros(
        (len(b_idx), max_n, legal_storage.shape[1]),
        dtype=legal_storage.dtype,
    )
    b_offsets_np = offsets_storage[b_idx]
    for i, (n, off) in enumerate(zip(b_lens_np, b_offsets_np)):
        if n > 0:
            legal_batch_cpu[i, :n] = legal_storage[off : off + n]
    return torch.as_tensor(legal_batch_cpu, dtype=torch.float32, device=device)


def _rebuild_shared_history_numpy(
    rollout: dict[str, np.ndarray],
    b_idx: np.ndarray,
    max_h: int | None = None,
) -> np.ndarray:
    """Materialize shared game histories for one CPU minibatch."""
    b_lens = rollout["history_lens"][b_idx].astype(np.int64, copy=False)
    if max_h is None:
        max_h = max(int(b_lens.max()), 1) if len(b_lens) else 1

    history = np.zeros(
        (len(b_idx), max_h, PLAY_TOKEN_SIZE),
        dtype=np.float32,
    )
    if not len(b_idx) or not np.any(b_lens):
        return history

    game_ids = rollout["history_game_ids"][b_idx]
    starts = rollout["history_game_offsets"][game_ids].astype(
        np.int64, copy=False
    )
    observers = rollout["history_observers"][b_idx].astype(
        np.int64, copy=False
    )
    columns = np.arange(max_h, dtype=np.int64)[None, :]
    valid = columns < b_lens[:, None]
    rows, token_columns = np.nonzero(valid)
    token_ids = starts[rows] + token_columns

    history[rows, token_columns, 4:] = rollout["history_tokens"][token_ids]
    relative_players = (
        rollout["history_players"][token_ids].astype(np.int64) - observers[rows]
    ) % 4
    history[rows, token_columns, relative_players] = 1.0
    return history


def _rebuild_shared_history_eager(
    history_storage: dict[str, torch.Tensor],
    b_idx_t: torch.Tensor,
    b_lens: torch.Tensor,
    max_h: int,
) -> torch.Tensor:
    """Materialize shared game histories directly on the training device."""
    batch_size = len(b_idx_t)
    device = b_idx_t.device
    history = torch.zeros(
        (batch_size, max_h, PLAY_TOKEN_SIZE),
        dtype=torch.float32,
        device=device,
    )
    if batch_size == 0 or int(b_lens.max().item()) == 0:
        return history

    game_ids = history_storage["game_ids"][b_idx_t]
    starts = history_storage["game_offsets"][game_ids]
    observers = history_storage["observers"][b_idx_t]
    columns = torch.arange(max_h, dtype=torch.int64, device=device).unsqueeze(0)
    valid = columns < b_lens.to(torch.int64).unsqueeze(1)
    token_ids = starts.unsqueeze(1) + columns
    safe_token_ids = torch.where(
        valid,
        token_ids,
        torch.zeros_like(token_ids),
    )

    history[:, :, 4:] = (
        history_storage["tokens"][safe_token_ids].to(torch.float32)
        * valid.unsqueeze(-1)
    )
    relative_players = (
        history_storage["players"][safe_token_ids]
        - observers.unsqueeze(1)
    ) % 4
    history.scatter_(
        2,
        relative_players.unsqueeze(-1),
        valid.to(torch.float32).unsqueeze(-1),
    )
    return history


def _prepare_spec_batches(
    policy,
    rollout: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Featurize dragon/wish decisions once, before the epoch loop.

    The fused features are computed without grad and frozen for all epochs, so
    only the specialist heads receive gradients. That keeps these rare, sparse
    decisions from perturbing the trunk/history encoder the main policy relies
    on, and makes the PPO ratio well-defined: the "state" for these decisions
    is the rollout-time featurization.
    """
    spec_data = {}
    was_training = policy.training
    policy.eval()
    for kind in ("dragon", "wish"):
        if f"{kind}_obs" not in rollout or len(rollout[f"{kind}_obs"]) == 0:
            continue
        with torch.no_grad():
            hl = torch.as_tensor(
                rollout[f"{kind}_history_lens"], dtype=torch.int32, device=device
            )
            max_h = max(int(hl.max().item()), 1)
            hist = torch.as_tensor(
                rollout[f"{kind}_history"][:, :max_h], dtype=torch.float32, device=device
            )
            spec_obs = torch.as_tensor(rollout[f"{kind}_obs"], device=device)
            ps = torch.as_tensor(
                rollout[f"{kind}_pass_source_ids"], dtype=torch.int64, device=device
            )
            pt = torch.as_tensor(
                rollout[f"{kind}_pass_target_ids"], dtype=torch.int64, device=device
            )
            fused = policy.fused_features(spec_obs, hist, hl, ps, pt)
        acts = torch.as_tensor(rollout[f"{kind}_actions"], device=device)
        old_lp = torch.as_tensor(rollout[f"{kind}_log_probs"], device=device)
        adv = torch.as_tensor(
            rollout[f"{kind}_returns"] - rollout[f"{kind}_values"], device=device
        )
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        spec_data[kind] = (fused, acts, old_lp, adv)
    if was_training:
        policy.train()
    return spec_data


def ppo_update_history(
    policy,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, np.ndarray],
    cfg: TrainConfig,
    device: torch.device,
    mmd=None,
) -> dict[str, float]:
    """Perform PPO updates on the history-aware policy.

    ``mmd`` (training.mmd.MMDController) switches the play loss from the
    entropy bonus to alpha * KL(pi || mixed magnet) when cfg.mmd_alpha > 0.
    
    Optimizations:
      1. One-Shot GPU Move: All rollout data is moved to VRAM once at the 
         start of the update. This saturates the PCIe bus and avoids 
         CPU-GPU bottlenecking during mini-batching.
      2. GPU-Side Slicing: Mini-batches are created using advanced indexing
         directly on VRAM tensors.
      3. Vectorized Padding: Legal actions (jagged array) are padded into 
         dense tensors using GPU-parallel indexing.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    update_started = time.perf_counter()
    timing = _PpoStageTimer(
        device,
        enabled=bool(getattr(cfg, "profile_update_timings", False)),
    )
    setup_started = timing.start()

    use_lazy = getattr(cfg, "use_lazy_gpu_loading", False)
    amp_enabled = cfg.use_amp and device.type == "cuda"

    obs = torch.as_tensor(rollout["obs"], device=device)
    action_idxs = torch.as_tensor(rollout["action_idxs"], device=device)
    pass_source_ids = torch.as_tensor(
        rollout["pass_source_ids"], dtype=torch.int64, device=device
    )
    pass_target_ids = torch.as_tensor(
        rollout["pass_target_ids"], dtype=torch.int64, device=device
    )
    old_lp = torch.as_tensor(rollout["log_probs"], device=device)
    returns = torch.as_tensor(rollout["returns"], device=device)
    advs = torch.as_tensor(rollout["advantages"], device=device)
    hist_lens = torch.as_tensor(
        rollout["history_lens"], dtype=torch.int32, device=device
    )

    legal_lens_np = rollout["legal_lens"]
    offsets_np = _legal_offsets(legal_lens_np)
    all_legal_lens = torch.as_tensor(legal_lens_np, dtype=torch.int32, device=device)

    shared_history = "history_tokens" in rollout
    if shared_history and use_lazy:
        history_storage = rollout
    elif shared_history:
        history_storage = {
            "tokens": torch.as_tensor(
                rollout["history_tokens"], device=device
            ),
            "players": torch.as_tensor(
                rollout["history_players"], dtype=torch.int64, device=device
            ),
            "game_offsets": torch.as_tensor(
                rollout["history_game_offsets"],
                dtype=torch.int64,
                device=device,
            ),
            "game_ids": torch.as_tensor(
                rollout["history_game_ids"],
                dtype=torch.int64,
                device=device,
            ),
            "observers": torch.as_tensor(
                rollout["history_observers"],
                dtype=torch.int64,
                device=device,
            ),
        }
    elif use_lazy:
        history_storage = rollout["history"]
    else:
        history_storage = torch.as_tensor(rollout["history"], device=device)

    if use_lazy:
        legal_storage = rollout["legal_flat"]
        offsets_storage = offsets_np
    else:
        legal_storage = torch.as_tensor(rollout["legal_flat"], device=device)
        offsets_storage = torch.as_tensor(offsets_np, dtype=torch.int64, device=device)

    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    N = len(obs)
    idx = np.arange(N)

    stat_pl, stat_vl, stat_ent, stat_kl, stat_clip = [], [], [], [], []
    stat_grad_norm = []  # pre-clip global grad norm per optimizer step
    stat_magnet_kl = []
    use_mmd = mmd is not None and getattr(cfg, "mmd_alpha", 0.0) > 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    spec_data = _prepare_spec_batches(policy, rollout, device)
    spec_stats = {kind: {"loss": [], "entropy": []} for kind in spec_data}

    configured_optimizer_batch = getattr(cfg, "optimizer_batch_size", None)
    if configured_optimizer_batch is not None and configured_optimizer_batch <= 0:
        raise ValueError("optimizer_batch_size must be positive or None")
    optimizer_batch_size = configured_optimizer_batch or N
    microbatch_size = min(cfg.batch_size, optimizer_batch_size)
    optimizer_steps = 0

    policy.train()
    timing.stop("setup", setup_started)
    for _ in range(cfg.n_epochs):
        order_started = timing.start()
        order = _ppo_order(
            idx,
            rollout,
            legal_lens_np,
            microbatch_size,
            cfg.use_ppo_bucketing,
            shuffle_bucket_microbatches=configured_optimizer_batch is not None,
        )
        timing.stop("epoch_order", order_started)

        for optimizer_start in range(0, N, optimizer_batch_size):
            optimizer_stop = min(optimizer_start + optimizer_batch_size, N)
            rows_in_optimizer_batch = optimizer_stop - optimizer_start
            optimizer.zero_grad(set_to_none=True)

            for start in range(optimizer_start, optimizer_stop, microbatch_size):
                prep_started = timing.start()
                stop = min(start + microbatch_size, optimizer_stop)
                b_idx = order[start:stop]
                b_idx_t = torch.as_tensor(b_idx, device=device)

                # ── MICROBATCH PREP ──────────────────────────────────
                obs_b = obs[b_idx_t]
                act_b = action_idxs[b_idx_t]
                ps_b = pass_source_ids[b_idx_t]
                pt_b = pass_target_ids[b_idx_t]
                old_lp_b = old_lp[b_idx_t]
                ret_b = returns[b_idx_t]
                adv_b = advs[b_idx_t]
                hl_b = hist_lens[b_idx_t]
                b_lens = all_legal_lens[b_idx_t]

                if use_lazy:
                    legal_pad = _pad_legal_lazy(
                        legal_storage,
                        offsets_storage,
                        b_idx,
                        legal_lens_np,
                        device,
                    )
                else:
                    legal_pad = _pad_legal_eager(
                        legal_storage,
                        offsets_storage,
                        b_idx_t,
                        b_lens,
                    )

                max_h = max(int(hl_b.max().item()), 1)
                if shared_history and use_lazy:
                    b_hist = torch.as_tensor(
                        _rebuild_shared_history_numpy(rollout, b_idx, max_h),
                        dtype=torch.float32,
                        device=device,
                    )
                elif shared_history:
                    b_hist = _rebuild_shared_history_eager(
                        history_storage,
                        b_idx_t,
                        hl_b,
                        max_h,
                    )
                elif use_lazy:
                    b_hist = torch.as_tensor(
                        history_storage[b_idx, :max_h],
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    b_hist = history_storage[b_idx_t][:, :max_h].to(torch.float32)
                timing.stop("microbatch_prep", prep_started)

                forward_started = timing.start()
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=amp_enabled,
                ):
                    log_prob, value, entropy, logits_b, valid_b = policy.evaluate(
                        obs_b,
                        legal_pad,
                        b_lens,
                        act_b,
                        b_hist,
                        hl_b,
                        ps_b,
                        pt_b,
                        return_logits=True,
                    )
                    if use_mmd:
                        with torch.no_grad():
                            magnet_logits = None
                            if mmd.magnet_policy is not None:
                                magnet_logits, _ = mmd.magnet_policy._forward_scores(
                                    obs_b, legal_pad, b_hist, hl_b, ps_b, pt_b
                                )
                            log_rho = magnet_log_probs(magnet_logits, valid_b, cfg.mmd_epsilon)
                        magnet_kl = kl_to_magnet(logits_b, valid_b, log_rho).mean()
                        regularizer = cfg.mmd_alpha * magnet_kl
                    else:
                        regularizer = -cfg.entropy_coef * entropy.mean()

                    ratio = torch.exp(log_prob - old_lp_b)
                    with torch.no_grad():
                        clamped = (
                            (ratio < 1 - cfg.clip_eps)
                            | (ratio > 1 + cfg.clip_eps)
                        )
                        stat_clip.append(clamped.float().mean().item())

                    pg_loss = torch.max(
                        -adv_b * ratio,
                        -adv_b
                        * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps),
                    ).mean()
                    v_loss = ((value - ret_b) ** 2).mean()
                    ent_loss = entropy.mean()

                    # Sum of scaled microbatch gradients equals the mean
                    # gradient of this effective optimizer batch.
                    scale = len(b_idx) / rows_in_optimizer_batch
                    loss = (
                        pg_loss
                        + cfg.value_coef * v_loss
                        + regularizer
                    ) * scale
                timing.stop("forward_loss", forward_started)

                backward_started = timing.start()
                scaler.scale(loss).backward()
                timing.stop("backward", backward_started)

                metrics_started = timing.start()
                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean()

                stat_pl.append(pg_loss.item())
                stat_vl.append(v_loss.item())
                stat_ent.append(ent_loss.item())
                stat_kl.append(kl.item())
                if use_mmd:
                    stat_magnet_kl.append(magnet_kl.item())
                timing.stop("metrics", metrics_started)

            # Specialist heads historically step once per epoch. Their
            # parameters are disjoint from the play loss, so attach them only
            # to the final main-policy optimizer batch of the epoch.
            specialist_started = timing.start()
            if optimizer_stop == N:
                for kind, (
                    fused,
                    spec_acts,
                    spec_old_lp,
                    spec_adv,
                ) in spec_data.items():
                    head = (
                        policy.dragon_head
                        if kind == "dragon"
                        else policy.wish_head
                    )
                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=amp_enabled,
                    ):
                        spec_logits = head(fused)
                        dist = Categorical(logits=spec_logits)
                        spec_lp = dist.log_prob(spec_acts)
                        spec_ratio = torch.exp(spec_lp - spec_old_lp)
                        spec_pg = torch.max(
                            -spec_adv * spec_ratio,
                            -spec_adv
                            * spec_ratio.clamp(
                                1 - cfg.clip_eps,
                                1 + cfg.clip_eps,
                            ),
                        ).mean()
                        spec_ent = dist.entropy().mean()
                        spec_loss = spec_pg - cfg.entropy_coef * spec_ent
                    scaler.scale(spec_loss).backward()
                    spec_stats[kind]["loss"].append(spec_pg.item())
                    spec_stats[kind]["entropy"].append(spec_ent.item())
            timing.stop("specialists", specialist_started)

            optimizer_started = timing.start()
            scaler.unscale_(optimizer)
            grad_norm = float(nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm))
            if math.isfinite(grad_norm):  # AMP overflow steps are skipped by the scaler
                stat_grad_norm.append(grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            timing.stop("optimizer", optimizer_started)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 2**20
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 2**20
    else:
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0
    stage_seconds = timing.seconds()
    update_seconds = time.perf_counter() - update_started

    del obs, action_idxs, pass_source_ids, pass_target_ids, old_lp, returns, advs
    del hist_lens, all_legal_lens
    del history_storage, legal_storage, offsets_storage

    if device.type == "cuda":
        torch.cuda.empty_cache()

    stats = {
        **({"magnet_kl": float(np.mean(stat_magnet_kl)), "mmd_round": float(mmd.state.round)}
           if use_mmd and stat_magnet_kl else {}),
        "policy_loss": float(np.mean(stat_pl)),
        "value_loss": float(np.mean(stat_vl)),
        "entropy": float(np.mean(stat_ent)),
        "approx_kl": float(np.mean(stat_kl)),
        "clip_frac": float(np.mean(stat_clip)),
        "grad_norm": float(np.mean(stat_grad_norm)) if stat_grad_norm else float("nan"),
        "grad_norm_max": float(np.max(stat_grad_norm)) if stat_grad_norm else float("nan"),
        # share of optimizer steps whose gradient max_grad_norm scaled down
        "grad_clipped_frac": (
            float(np.mean(np.asarray(stat_grad_norm) > cfg.max_grad_norm)) if stat_grad_norm else float("nan")
        ),
        "optimizer_steps": float(optimizer_steps),
        "transitions_per_optimizer_step": float(
            cfg.n_epochs * N / optimizer_steps
        ),
        "ppo_seconds": float(update_seconds),
        "peak_allocated_mb": float(peak_allocated_mb),
        "peak_reserved_mb": float(peak_reserved_mb),
    }
    stats.update(
        {
            f"ppo_stage_{name}_seconds": float(seconds)
            for name, seconds in stage_seconds.items()
        }
    )
    for kind in ("dragon", "wish"):
        losses = spec_stats.get(kind, {}).get("loss", [])
        ents = spec_stats.get(kind, {}).get("entropy", [])
        stats[f"{kind}_loss"] = float(np.mean(losses)) if losses else 0.0
        stats[f"{kind}_entropy"] = float(np.mean(ents)) if ents else 0.0
    return stats
