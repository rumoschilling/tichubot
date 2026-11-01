"""Where and how do the ring agents disagree?

Uses the same fixed decision-state bank as ``training.ring_diversity``. For
every state, takes each agent's most likely move (moves equal up to suit
pooled, as in ``ring_diversity``); for every pair of agents
whose top moves differ, classifies the disagreement and its context, and
prints frequency tables plus a few concrete examples.

Usage:
    python -m training.ring_disagreements CKPT0 CKPT1 CKPT2 CKPT3 [--states 2000]
"""

from __future__ import annotations

import argparse
import collections
import json
import random

import numpy as np
import torch

from tichu_env.cards import card_name
from tichu_env.combinations import CombinationType, classify_cards
from training.ring_diversity import build_state_bank, policy_probs, pooled


def describe(action) -> str:
    cards, tichu = action[0], action[1]
    if not cards:
        return "pass" + (" + Tichu" if tichu else "")
    combo = classify_cards(cards)
    kind = CombinationType(combo.type).name.lower() if combo else "?"
    text = f"{kind} {' '.join(card_name(c) for c in sorted(cards))}"
    return text + (" + Tichu" if tichu else "")


def kind_of(action) -> str:
    if not action[0]:
        return "pass"
    combo = classify_cards(action[0])
    if combo is None:
        return "?"
    return "bomb" if combo.is_bomb else CombinationType(combo.type).name.lower()


def classify(a, b) -> str:
    """One label for how two different top moves differ."""
    ka, kb = kind_of(a), kind_of(b)
    if sorted(a[0]) == sorted(b[0]):
        if a[1] != b[1]:
            return "Tichu call vs no call (same cards)"
        return "Phoenix meaning only"
    if "pass" in (ka, kb):
        return "pass vs play"
    if "bomb" in (ka, kb) and ka != kb:
        return "bomb vs non-bomb"
    if ka == kb:
        ra, rb = classify_cards(a[0]).rank, classify_cards(b[0]).rank
        if ra == rb:
            return f"same {ka} rank, different cards"
        return f"same type ({ka}), different rank"
    return "different combination type"


def context(entry) -> dict[str, str]:
    gs, p = entry["gs"], entry["player"]
    top = gs.trick.leading_combo
    hand = len(gs.hands[p])
    ctx = {
        "situation": "leading" if top is None else "following",
        "hand size": "10-14 cards" if hand >= 10 else "5-9 cards" if hand >= 5 else "1-4 cards",
    }
    if top is not None:
        ctx["trick held by"] = "partner" if gs.trick.leading_player == (p + 2) % 4 else "opponent"
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs=4)
    ap.add_argument("--states", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--out", default=None, help="write the tables as JSON")
    args = ap.parse_args()
    torch.set_num_threads(4)

    bank = build_state_bank(args.states, args.seed)
    probs = [policy_probs(path, bank) for path in args.checkpoints]
    # Top move per agent on suit-pooled moves; the shown move is the agent's
    # most likely card choice within its top group.
    top_group = [[int(np.argmax(pooled(p, e["groups"]))) for p, e in zip(ps, bank)] for ps in probs]
    tops = [[int(np.argmax(np.where(e["groups"] == g, p, -1.0))) for p, e, g in zip(ps, bank, tg)]
            for ps, tg in zip(probs, top_group)]

    kinds = collections.Counter()
    by_context = collections.defaultdict(collections.Counter)  # context value -> [disagree, total]
    passer = collections.Counter()  # in pass-vs-play splits, which agent passed
    passer_total = collections.Counter()
    bomber = collections.Counter()
    bomb_total = collections.Counter()
    examples = []
    n_pairs = 0
    all_agree = 0
    for s, entry in enumerate(bank):
        legal = entry["legal"]
        moves = [legal[tops[a][s]] for a in range(4)]
        if len({top_group[a][s] for a in range(4)}) == 1:
            all_agree += 1
        ctx = context(entry)
        for i in range(4):
            for j in range(i + 1, 4):
                n_pairs += 1
                differ = top_group[i][s] != top_group[j][s]
                for key, value in ctx.items():
                    by_context[f"{key}: {value}"]["pairs"] += 1
                    by_context[f"{key}: {value}"]["disagree"] += differ
                if not differ:
                    continue
                label = classify(moves[i], moves[j])
                kinds[label] += 1
                if label == "pass vs play":
                    for a in (i, j):
                        passer_total[a] += 1
                        passer[a] += kind_of(moves[a]) == "pass"
                if label == "bomb vs non-bomb":
                    for a in (i, j):
                        bomb_total[a] += 1
                        bomber[a] += kind_of(moves[a]) == "bomb"
        if len(set(top_group[a][s] for a in range(4))) >= 3 and len(examples) < 200:
            examples.append(s)

    total_dis = sum(kinds.values())
    print(f"\n{args.states} states, {n_pairs} agent pairs; top moves all agree in {all_agree / args.states:.1%} of states; "
          f"pairs disagree {total_dis / n_pairs:.1%} of the time\n")
    print("How pairs disagree (share of disagreements):")
    for label, n in kinds.most_common():
        print(f"  {n / total_dis:6.1%}  {label}")
    print("\nDisagreement rate by context:")
    for key in sorted(by_context):
        c = by_context[key]
        print(f"  {c['disagree'] / c['pairs']:6.1%}  {key}  ({c['pairs'] // 6} states)")
    print("\nIn pass-vs-play splits, how often each agent is the one passing:")
    for a in range(4):
        if passer_total[a]:
            print(f"  agent{a}: {passer[a] / passer_total[a]:.0%} of {passer_total[a]}")
    print("\nIn bomb-vs-non-bomb splits, how often each agent is the one bombing:")
    for a in range(4):
        if bomb_total[a]:
            print(f"  agent{a}: {bomber[a] / bomb_total[a]:.0%} of {bomb_total[a]}")

    rng = random.Random(1)
    print("\nExamples (states where 3+ agents pick different moves):")
    for s in rng.sample(examples, min(args.examples, len(examples))):
        e = bank[s]
        gs, p = e["gs"], e["player"]
        top = gs.trick.cards_on_table
        print(f"\n  hand ({len(gs.hands[p])}): {' '.join(card_name(c) for c in sorted(gs.hands[p]))}")
        print(f"  table: {' '.join(card_name(c) for c in top) if top else 'empty (leading)'}"
              + (f"  held by {'partner' if gs.trick.leading_player == (p + 2) % 4 else 'opponent'}" if top else ""))
        print(f"  cards left: {[len(gs.hands[(p + k) % 4]) for k in (1, 2, 3)]} (left, partner, right)")
        for a in range(4):
            move = e["legal"][tops[a][s]]
            group_p = pooled(probs[a][s], e["groups"])[top_group[a][s]]
            print(f"    agent{a}: {describe(move):40s} p={group_p:.2f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "states": args.states, "pairs": n_pairs, "all_agree": all_agree / args.states,
                "kinds": {k: v / total_dis for k, v in kinds.items()},
                "by_context": {k: v["disagree"] / v["pairs"] for k, v in by_context.items()},
                "passer": {a: passer[a] / passer_total[a] for a in range(4) if passer_total[a]},
                "bomber": {a: bomber[a] / bomb_total[a] for a in range(4) if bomb_total[a]},
            }, fh, indent=2)


if __name__ == "__main__":
    main()
