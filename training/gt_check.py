"""Grand Tichu head health check.

Reports the average declare probability and, more importantly, whether the
head is *selective*: declare probability bucketed by a simple hand-strength
proxy over random 8-card deals. A healthy head is ~0% on weak hands and
rises monotonically; a collapsed head is 0% everywhere (v29 syndrome), a
degenerate one is flat.

Usage:
    python -m training.gt_check path/to/latest.pt
    python -m training.gt_check path/to/latest.pt --hands 500000
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from tichu_env.cards import (
    DOG_INDEX, DRAGON_INDEX, NUM_CARDS, PHOENIX_INDEX, card_rank,
)
from training.model_gt import TichuGrandTichuPolicy

STRENGTH_BUCKETS = [(-1, 0), (0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 99)]


def hand_strength(cards: list[int] | np.ndarray) -> float:
    """Crude GT-relevance proxy: Dragon/Phoenix +4, A +3, K +2, Q +1, Dog -1."""
    s = 0.0
    for c in cards:
        c = int(c)
        if c in (DRAGON_INDEX, PHOENIX_INDEX):
            s += 4
        elif c == DOG_INDEX:
            s -= 1
        elif c < 52:
            r = card_rank(c).value
            if r == 14:
                s += 3
            elif r == 13:
                s += 2
            elif r == 12:
                s += 1
    return s


def load_gt_policy(ckpt_path: str, device: str = "cpu") -> TichuGrandTichuPolicy | None:
    """Load the GT head from a training checkpoint, or None if absent."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gt_state = ckpt.get("gt_model")
    if gt_state is None:
        return None
    cfg = ckpt.get("config")
    hidden = getattr(cfg, "gt_hidden_dim", 64) if cfg is not None else 64
    layers = getattr(cfg, "gt_n_layers", 2) if cfg is not None else 2
    policy = TichuGrandTichuPolicy(hidden_dim=hidden, n_layers=layers).to(device)
    policy.load_state_dict(gt_state)
    policy.eval()
    return policy


def gt_report(ckpt_path: str, n_hands: int = 100_000, seed: int = 0,
              quiet: bool = False) -> dict | None:
    """Probe declare prob vs hand strength on random 8-card deals.

    Returns a dict with mean_prob, entropy, corr, and per-bucket rates,
    or None if the checkpoint has no GT head.
    """
    policy = load_gt_policy(ckpt_path)
    if policy is None:
        if not quiet:
            print(f"  No gt_model in {ckpt_path}")
        return None

    rng = np.random.default_rng(seed)
    hands = np.zeros((n_hands, NUM_CARDS), dtype=np.float32)
    strength = np.zeros(n_hands)
    for i in range(n_hands):
        cards = rng.choice(NUM_CARDS, 8, replace=False)
        hands[i, cards] = 1.0
        strength[i] = hand_strength(cards)

    probs = np.zeros(n_hands)
    entropy = 0.0
    with torch.no_grad():
        for lo in range(0, n_hands, 50_000):
            chunk = torch.from_numpy(hands[lo:lo + 50_000])
            logits = policy.call_logits(chunk)
            dist = torch.distributions.Bernoulli(logits=logits)
            probs[lo:lo + 50_000] = torch.sigmoid(logits).numpy()
            entropy += dist.entropy().sum().item()
    entropy /= n_hands

    buckets = {}
    for lo, hi in STRENGTH_BUCKETS:
        mask = (strength >= lo) & (strength < hi)
        if mask.sum() > 0:
            buckets[(lo, hi)] = float(probs[mask].mean())

    result = {
        "mean_prob": float(probs.mean()),
        "entropy": float(entropy),
        "corr_strength": float(np.corrcoef(strength, probs)[0, 1]),
        "buckets": buckets,
    }

    if not quiet:
        print(f"  GT head: mean call prob {result['mean_prob']:.1%}  "
              f"entropy {result['entropy']:.3f}  "
              f"corr(strength, prob) {result['corr_strength']:+.3f}")
        print(f"  {'strength':>9}  {'call%':>6}")
        for (lo, hi), p in buckets.items():
            print(f"  {lo:>4}-{hi:<4}  {p:>6.1%}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Grand Tichu head health check")
    parser.add_argument("checkpoint", type=str, help="Path to the .pt checkpoint")
    parser.add_argument("--hands", type=int, default=100_000,
                        help="Number of random 8-card deals to probe")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"Checking GT head: {args.checkpoint}")
    gt_report(args.checkpoint, n_hands=args.hands, seed=args.seed)


if __name__ == "__main__":
    main()
