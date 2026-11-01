"""Magnetic Mirror Descent for the play policy: a moving, mixed magnet.

Background (Sokota et al., ICLR 2023; the outer loop is DeepNash's R-NaD):
the play loss carries  alpha * KL(pi || rho)  instead of an entropy bonus.
rho, the magnet, is

    rho = (1 - eps) * magnet_policy + eps * uniform(legal moves)

In round 0 there is no magnet policy, rho is uniform, and the term equals an
entropy bonus with coefficient alpha (KL(pi || U) = log n - H(pi)). When the
policy has converged against the magnet, the magnet moves to the current
policy and a new round starts. Each round's fixed point is the magnet
re-weighted by move values, so the policy sharpens round by round at the pace
it actually converges; eps caps how far an unplayed move can be suppressed,
so moves keep being sampled and can be rediscovered.

Knobs and how they are set (see MEMORY.md 2026-10-26):
  alpha = 0.03   equals the lineage's validated starting entropy coefficient;
                 with a moving magnet it sets speed/stability, not the ceiling
  eps   = 0.2    calibrated: the lineage at its 0.01 entropy floor spent 15%
                 of its probability on non-preferred moves (4,000 bank states,
                 4.54M and 5.80M); eps * mean(1 - 1/n) = 0.15
  convergence    the per-update KL(pi || rho) stops rising: OLS slope over the
                 last `window` updates is not > 2 SE, after `min_round_games`
  safety check   before moving, the policy plays the magnet (16,384 paired
                 deals); if it is clearly worse (z <= -2) the round restarts
                 from the magnet's weights instead of anchoring a regression
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch


def magnet_log_probs(
    magnet_logits: torch.Tensor | None,
    valid: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """log rho over padded legal moves (invalid entries are 0, never used).

    magnet_logits: (B, A) raw scores of the magnet policy, or None (uniform).
    valid:         (B, A) bool mask of legal moves.
    """
    n = valid.sum(-1, keepdim=True).clamp_min(1).to(torch.float32)
    uniform = valid.to(torch.float32) / n
    if magnet_logits is None:
        rho = uniform
    else:
        masked = magnet_logits.float().masked_fill(~valid, float("-inf"))
        rho = (1.0 - eps) * torch.softmax(masked, dim=-1) + eps * uniform
    return torch.log(rho.clamp_min(1e-30)).masked_fill(~valid, 0.0)


def kl_to_magnet(logits: torch.Tensor, valid: torch.Tensor, log_rho: torch.Tensor) -> torch.Tensor:
    """Per-row KL(pi || rho) over legal moves; differentiable in ``logits``."""
    log_pi = torch.log_softmax(logits.float().masked_fill(~valid, float("-inf")), dim=-1)
    log_pi = log_pi.masked_fill(~valid, 0.0)
    return (log_pi.exp() * (log_pi - log_rho)).sum(-1)


def kl_plateaued(values: list[float], z: float = 2.0) -> tuple[bool, float, float]:
    """True when the OLS slope of ``values`` is not significantly positive.

    Returns (plateaued, slope, slope_se) with the slope per update.
    """
    y = np.asarray(values, dtype=np.float64)
    n = len(y)
    if n < 3:
        return False, float("nan"), float("nan")
    x = np.arange(n, dtype=np.float64)
    x -= x.mean()
    slope = float((x * (y - y.mean())).sum() / (x**2).sum())
    resid = y - y.mean() - slope * x
    se = float(math.sqrt((resid**2).sum() / (n - 2) / (x**2).sum()))
    return slope <= z * se, slope, se


@dataclass
class MMDState:
    round: int = 0
    round_start_games: int = 0
    kl_history: list[float] = field(default_factory=list)
    restarts: int = 0
    events: list[dict] = field(default_factory=list)


class MMDController:
    """Owns the magnet, decides when to move it, and runs the safety check.

    Files in ``checkpoint_dir``:
      mmd_state.json   round, KL window, event log (written every update)
      magnet.pt        full play checkpoint of the current magnet (round >= 1);
                       also the rollback point for a regressed round
      mmd_round_NN.pt  the policy that ended round NN (kept for the curve)
    """

    def __init__(self, cfg, device: torch.device, save_checkpoint, load_checkpoint, policy):
        self.cfg = cfg
        self.device = device
        self.save_checkpoint = save_checkpoint  # (games, path) -> None
        self.load_checkpoint = load_checkpoint  # (path) -> None, restores the training bundle
        self.policy = policy
        self.dir = Path(cfg.checkpoint_dir)
        self.state_path = self.dir / "mmd_state.json"
        self.magnet_path = self.dir / "magnet.pt"
        self.state = MMDState()
        self.magnet_policy: torch.nn.Module | None = None

    # ── state ──────────────────────────────────────────────────────────
    def restore(self, games_played: int) -> None:
        """Load state and magnet when resuming; start round 0 otherwise."""
        if self.state_path.exists():
            self.state = MMDState(**json.loads(self.state_path.read_text()))
            if self.state.round > 0:
                self.magnet_policy = self._frozen_copy_from(self.magnet_path)
            print(f"  [mmd] resumed round {self.state.round} (started at {self.state.round_start_games:,} games, "
                  f"{len(self.state.kl_history)} KL samples, {self.state.restarts} restarts)", flush=True)
        else:
            self.state = MMDState(round_start_games=games_played)
            self._event("start", games_played)
            print(f"  [mmd] round 0 (uniform magnet) from {games_played:,} games, alpha={self.cfg.mmd_alpha} "
                  f"eps={self.cfg.mmd_epsilon}", flush=True)
        self._write_state()

    def _write_state(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=1) + "\n")
        tmp.replace(self.state_path)

    def _event(self, kind: str, games: int, **info) -> None:
        self.state.events.append({"event": kind, "games": int(games), "round": self.state.round,
                                  "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **info})

    def _frozen_copy_from(self, path: Path) -> torch.nn.Module:
        magnet = copy.deepcopy(self.policy)
        state = torch.load(path, map_location=self.device, weights_only=False)["model"]
        magnet.load_state_dict(state)
        return self._freeze(magnet)

    @staticmethod
    def _freeze(module: torch.nn.Module) -> torch.nn.Module:
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
        return module

    # ── per update ─────────────────────────────────────────────────────
    def after_update(self, games_played: int, loss_stats: dict[str, float]) -> None:
        kl = loss_stats.get("magnet_kl")
        if kl is None or not math.isfinite(kl):
            return
        st = self.state
        st.kl_history.append(float(kl))
        st.kl_history = st.kl_history[-self.cfg.mmd_window_updates:]
        round_games = games_played - st.round_start_games
        due = False
        if round_games >= self.cfg.mmd_min_round_games and len(st.kl_history) >= self.cfg.mmd_window_updates:
            plateau, slope, se = kl_plateaued(st.kl_history)
            due = plateau
            reason = {"slope": slope, "slope_se": se, "kl": float(np.mean(st.kl_history[-10:]))}
        if not due and round_games >= self.cfg.mmd_max_round_games:
            due, reason = True, {"forced": True, "kl": float(np.mean(st.kl_history[-10:]))}
        if due:
            self._end_round(games_played, reason)
        self._write_state()

    # ── round end ──────────────────────────────────────────────────────
    def _end_round(self, games: int, reason: dict) -> None:
        st = self.state
        candidate = self.dir / f"mmd_round_{st.round:02d}.pt"
        self.save_checkpoint(games, str(candidate))
        print(f"  [mmd] round {st.round} converged at {games:,} games ({reason}); checking against the magnet",
              flush=True)
        result = None
        if st.round > 0:
            result = self._evaluate(candidate, self.magnet_path, f"round{st.round:02d}_vs_magnet")
        ref = getattr(self.cfg, "mmd_reference", None)
        ref_result = self._evaluate(candidate, Path(ref), f"round{st.round:02d}_vs_reference") if ref else None
        if result is not None and result["z"] <= -2.0:
            # Clearly worse than its own anchor: restart the round from the magnet.
            self.load_checkpoint(str(self.magnet_path))
            st.restarts += 1
            self._event("restart", games, check=result, reference=ref_result, **reason)
            print(f"  [mmd] round {st.round} regressed ({result['mean']:+.2f} +- {result['se']:.2f}); "
                  f"restarting it from the magnet", flush=True)
        else:
            shutil.copyfile(candidate, self.magnet_path)
            self.magnet_policy = self._freeze(copy.deepcopy(self.policy))
            self._event("move", games, check=result, reference=ref_result, **reason)
            st.round += 1
            print(f"  [mmd] magnet moved; round {st.round} starts at {games:,} games"
                  + (f" (vs old magnet {result['mean']:+.2f} +- {result['se']:.2f})" if result else ""), flush=True)
        st.round_start_games = games
        st.kl_history = []

    def _evaluate(self, candidate: Path, opponent: Path, name: str) -> dict:
        """Paired strength of candidate vs opponent on the CPU worker pool."""
        import argparse

        from training.paired_checkpoint_eval import run_experiment

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        out = self.dir / "mmd_evals" / f"{name}.json"
        args = argparse.Namespace(
            candidate=f"candidate={candidate}", opponents=[f"opponent={opponent}"], output=str(out),
            device="cpu", deals=self.cfg.mmd_eval_deals, batch=256, block_size=256,
            bootstrap_samples=10_000, sign_flip_samples=20_000, alpha=0.05,
            seed=20261026 + 7919 * self.state.round + 13 * self.state.restarts, no_gt=False,
            procs=self.cfg.mmd_eval_procs, common_random_numbers=False,
        )
        started = time.perf_counter()
        comparison = run_experiment(args)["comparisons"][0]
        mean = float(comparison["candidate_mean_score_advantage"])
        se = float(comparison["paired_effect_std"]) / math.sqrt(comparison["deal_pairs"])
        summary = {"mean": mean, "se": se, "z": mean / se if se > 0 else 0.0,
                   "deals": int(comparison["deal_pairs"]), "seconds": round(time.perf_counter() - started, 1),
                   "file": str(out)}
        print(f"  [mmd] {name}: {mean:+.2f} +- {se:.2f} (z {summary['z']:+.2f}, {summary['seconds']}s)", flush=True)
        return summary
