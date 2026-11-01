"""Digest every legal-action list over many random seeded games.

Plays random full rounds (Grand Tichu, passing, play, dragon gifts, wishes)
and hashes ``get_legal_actions`` and ``get_policy_legal_actions`` at every
decision point. Run before and after an engine change and compare: equal
per-game digests mean identical legal actions at every visited state (the
games themselves follow the same path because choices index into the lists).

Usage:
    python -m training.legal_digest --games 5000 --out before.json
    python -m training.legal_digest --games 5000 --out after.json --compare before.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random

from tichu_env.cards import MAHJONG_INDEX
from tichu_env.game_state import GamePhase, GameState


def play_game(seed: int) -> tuple[str, int]:
    rng = random.Random(seed)
    gs = GameState(track_history=True)
    gs.reset(seed=seed, with_calling=True, with_passing=True)
    h = hashlib.sha256()
    decisions = 0
    for _ in range(5000):
        if gs.round_over:
            break
        if gs.phase == GamePhase.GRAND_TICHU:
            gs.declare_grand_tichu(gs.current_player, rng.random() < 0.15)
            continue
        if gs.phase == GamePhase.PASSING:
            p = gs.current_player
            left, partner, right = rng.sample(gs.get_passing_hand(p), 3)
            assert gs.pass_cards(p, left, partner, right)
            continue
        if gs.dragon_give_pending is not None:
            p = gs.dragon_give_pending
            gs.give_dragon_trick(p, rng.choice(gs.get_dragon_give_options(p)))
            continue
        p = gs.current_player
        legal = gs.get_legal_actions(p)
        policy = gs.get_policy_legal_actions(p)
        h.update(repr((p, legal, policy)).encode())
        decisions += 1
        if not policy:
            break
        action = policy[rng.randrange(len(policy))]
        assert gs.play(p, action), (seed, action)
        if MAHJONG_INDEX in action[0] and gs.mahjong_wish is None and not gs.round_over:
            w = rng.randrange(14)
            gs.set_mahjong_wish(None if w == 0 else w + 1)
    h.update(repr(gs.round_scores).encode())
    return h.hexdigest(), decisions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()
    digests, total = [], 0
    for k in range(args.games):
        d, n = play_game(args.seed * 1_000_003 + k)
        digests.append(d)
        total += n
    json.dump({"games": args.games, "seed": args.seed, "decisions": total, "digests": digests},
              open(args.out, "w"))
    print(f"{args.games} games, {total} decisions -> {args.out}")
    if args.compare:
        ref = json.load(open(args.compare))
        bad = [k for k, (a, b) in enumerate(zip(ref["digests"], digests)) if a != b]
        same = ref["decisions"] == total and not bad and len(ref["digests"]) == len(digests)
        print(f"compare vs {args.compare}: {'IDENTICAL' if same else 'DIFFERENT'} "
              f"(decisions {ref['decisions']} vs {total}, {len(bad)} games differ{', first ' + str(bad[:5]) if bad else ''})")
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
