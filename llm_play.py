#!/usr/bin/env python3
"""Non-interactive Tichu probe harness: play one seat against a trained bot,
one decision per CLI invocation.

Built for an agent (LLM) driving the game through repeated shell calls:
plain-text output (no ANSI), pickled state between calls, and a
full-information JSON log per round for post-hoc blunder review.

Unlike training/play.py, AI seats use the checkpoint's own dragon/wish
auxiliary heads (when present) instead of random choices, so the bot plays at
its trained strength.

Usage:
    llm_play.py new  [--seed S] [--seat N] [--checkpoint PATH] [--reset]
    llm_play.py show                re-print the pending decision
    llm_play.py act <n>             answer gt (0=no 1=yes), play (index),
                                    wish (0 or 2-14), dragon (opponent seat)
    llm_play.py pass <l> <p> <r>    answer passing decision (positions in the
                                    printed hand: to left, partner, right)
    llm_play.py log [n]             print full-info JSON log of round n
                                    (default: last finished round)

State lives in .llm_play/ (gitignored). `new` after a finished round starts
the next round and keeps the cumulative score.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys

import numpy as np
import torch

from tichu_env.cards import (
    CARD_POINTS,
    DOG_INDEX,
    MAHJONG_INDEX,
    NUM_CARDS,
    Rank,
    card_name,
)
from tichu_env.combinations import classify_cards
from tichu_env.display import COMBO_NAMES, _sort_key
from tichu_env.encoding import (
    encode_history,
    encode_pass_obs,
    encode_pass_source_ids,
    encode_pass_target_ids,
    encode_static,
)
from tichu_env.game_state import GameState, TEAM_A
from training.checkpoints import DEFAULT_PLAY_CHECKPOINT
from training.play import ai_select_action, load_play_bundle

DEFAULT_CHECKPOINT = DEFAULT_PLAY_CHECKPOINT
ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(ROOT, ".llm_play")
STATE_FILE = os.path.join(STATE_DIR, "state.pkl")

RANK_NAMES = {r.value: r.name.title() for r in Rank}
DEVICE = torch.device("cpu")


def pc(cards) -> str:
    """Plain-text card list, rank-sorted."""
    if not cards:
        return "(none)"
    return " ".join(card_name(c) for c in sorted(cards, key=_sort_key))


def combo_name(cards: list[int]) -> str:
    if not cards:
        return "Pass"
    combo = classify_cards(cards)
    return COMBO_NAMES.get(combo.type, "?") if combo else "?"


def _hand_to_binary(hand: list[int]) -> np.ndarray:
    vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for card in hand:
        vec[card] = 1.0
    return vec


def team_points(gs: GameState) -> tuple[int, int]:
    pts = [sum(CARD_POINTS[c] for c in gs.won_cards[p]) for p in range(4)]
    return pts[0] + pts[2], pts[1] + pts[3]


def hands_snapshot(gs: GameState) -> dict[str, str]:
    return {f"P{p}": pc(gs.hands[p]) for p in range(4)}


# ── AI decision functions (mirror rollout_history semantics) ──────────────


@torch.no_grad()
def _fused(policy, gs: GameState, p: int):
    obs = torch.as_tensor(
        encode_static(gs, p), dtype=torch.float32, device=DEVICE
    ).unsqueeze(0)
    ps = torch.as_tensor(
        encode_pass_source_ids(gs, p), dtype=torch.int64, device=DEVICE
    ).unsqueeze(0)
    pt = torch.as_tensor(
        encode_pass_target_ids(gs, p), dtype=torch.int64, device=DEVICE
    ).unsqueeze(0)
    hist, hl = encode_history(gs, p)
    h = torch.as_tensor(
        hist[: max(hl, 1)][np.newaxis], dtype=torch.float32, device=DEVICE
    )
    hlt = torch.tensor([hl], dtype=torch.int32, device=DEVICE)
    return policy.fused_features(obs, h, hlt, ps, pt)


@torch.no_grad()
def ai_gt(gt_policy, h8: list[int]) -> bool:
    if gt_policy is None:
        return False
    h8_bin = torch.as_tensor(_hand_to_binary(h8)[None, :], device=DEVICE)
    decl, _ = gt_policy.act(h8_bin)
    return bool(decl.item())


@torch.no_grad()
def ai_pass(pass_policy, gs: GameState, p: int, rng: random.Random):
    hand = gs.get_passing_hand(p)
    if pass_policy is None:
        left, partner, right = rng.sample(hand, 3)
    else:
        obs = torch.as_tensor(encode_pass_obs(gs, p)[None, :], device=DEVICE)
        hand_t = torch.as_tensor(
            np.array(hand, dtype=np.int64)[None, :], device=DEVICE
        )
        choices_t, _ = pass_policy.act(obs, hand_t)
        idxs = choices_t[0].cpu().numpy()
        left, partner, right = (hand[int(i)] for i in idxs)
    gs.pass_cards(p, left, partner, right)
    return left, partner, right


@torch.no_grad()
def ai_play(
    policy,
    gs: GameState,
    p: int,
    kind: str,
    rng: random.Random,
):
    """Pick and execute a model-aware action, preserving Phoenix semantics."""
    action = ai_select_action(policy, gs, p, DEVICE, rng, kind)
    gs.play(p, action)
    return action


@torch.no_grad()
def ai_wish(policy, gs: GameState, p: int, kind: str, rng: random.Random) -> int | None:
    """Choose a mahjong wish for an AI seat (post-play state). None = no wish."""
    if kind == "history" and hasattr(policy, "get_mahjong_wish_logits"):
        fused = _fused(policy, gs, p)
        logits = policy.get_mahjong_wish_logits(fused)
        w = int(torch.distributions.Categorical(logits=logits).sample().item())
        return None if w == 0 else w + 1
    return rng.randint(2, 14)


@torch.no_grad()
def ai_dragon(policy, gs: GameState, p: int, kind: str, rng: random.Random) -> int:
    """Choose the dragon-trick recipient for an AI seat."""
    options = gs.get_dragon_give_options(p)
    if kind == "history" and hasattr(policy, "get_dragon_gift_logits"):
        fused = _fused(policy, gs, p)
        logits = policy.get_dragon_gift_logits(fused)
        choice = int(torch.distributions.Categorical(logits=logits).sample().item())
        target = (p + (1 if choice == 0 else 3)) % 4
        if target in options:
            return target
    return rng.choice(options)


# ── Session state ──────────────────────────────────────────────────────────


def save_state(st: dict):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(STATE_FILE, "wb") as f:
        pickle.dump(st, f)


def load_state() -> dict | None:
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE, "rb") as f:
        return pickle.load(f)


def load_bundle(st: dict):
    return load_play_bundle(st["checkpoint"], None, DEVICE)


# ── Round advancement ──────────────────────────────────────────────────────


def log_event(st: dict, ev: dict):
    st["round_log"].append(ev)


def sort_legal(legal) -> list[int]:
    """Sorted display order for legal actions (same scheme as training/play.py)."""
    from tichu_env.cards import card_rank

    def key(i):
        cards, call_tichu = legal[i]
        if not cards:
            return (0, 0, 0, 1 if call_tichu else 0)
        combo = classify_cards(cards)
        type_val = combo.type.value if combo else 99
        ranks = [(card_rank(c).value if card_rank(c) else 0) for c in cards]
        return (1, type_val, min(ranks), 1 if call_tichu else 0)

    return sorted(range(len(legal)), key=key)


def advance(st: dict, bundle) -> list[str]:
    """Run AI seats until the human must decide or the round ends.

    Sets st["pending"] and returns transcript lines of public events.
    """
    policy, pass_policy, gt_policy, kind, _ = bundle
    gs: GameState = st["gs"]
    rng: random.Random = st["rng"]
    seat: int = st["seat"]
    out: list[str] = []

    while True:
        if gs.phase.name == "GRAND_TICHU":
            p = gs.current_player
            h8 = sorted(gs.hands[p])[:8]
            if p == seat:
                st["pending"] = ("gt", h8)
                return out
            decl = ai_gt(gt_policy, h8)
            gs.declare_grand_tichu(p, decl)
            log_event(st, {"t": "gt", "p": p, "declare": decl, "h8": pc(h8)})
            if decl:
                out.append(f"P{p} declares GRAND TICHU!")
            continue

        if gs.phase.name == "PASSING":
            p = gs.current_player
            if p == seat:
                st["pending"] = ("pass", gs.get_passing_hand(seat))
                return out
            left, partner, right = ai_pass(pass_policy, gs, p, rng)
            log_event(st, {
                "t": "pass", "p": p,
                "left": pc([left]), "partner": pc([partner]), "right": pc([right]),
            })
            continue

        # Report the human's post-passing hand once play begins
        if st.get("await_pass_report") and gs.phase.name == "PLAY":
            st["await_pass_report"] = False
            kept = st["hand_pre_pass"] - st["outgoing"]
            received = sorted(set(gs.hands[seat]) - kept, key=_sort_key)
            out.append(f"You received: {pc(received)}")
            out.append(f"Your hand after passing: {pc(gs.hands[seat])}")

        if gs.dragon_give_pending is not None:
            p = gs.dragon_give_pending
            if p == seat:
                st["pending"] = ("dragon", gs.get_dragon_give_options(seat))
                return out
            recipient = ai_dragon(policy, gs, p, kind, rng)
            gs.give_dragon_trick(p, recipient)
            log_event(st, {"t": "dragon", "p": p, "to": recipient})
            out.append(f"P{p} gives the Dragon trick to P{recipient}")
            continue

        if gs.round_over:
            finish_round(st, out)
            return out

        p = gs.current_player
        if p == seat:
            legal = gs.get_legal_actions(seat)
            st["pending"] = ("play", legal, sort_legal(legal))
            return out

        tricks_before = len(gs.trick_history)
        ev = {"t": "play", "p": p, "trick": gs.trick_num,
              "table_before": pc(gs.trick.cards_on_table),
              "hands": hands_snapshot(gs)}
        action = ai_play(policy, gs, p, kind, rng)
        cards, call_tichu = action[:2]
        ev.update({"cards": pc(cards), "combo": combo_name(cards),
                   "tichu": call_tichu})
        log_event(st, ev)

        line = f"P{p}: {pc(cards) if cards else 'Pass'}"
        if cards:
            line += f"  ({combo_name(cards)})"
        if cards == [DOG_INDEX]:
            line += "  -> lead passes to partner"
        if call_tichu:
            line += "  + TICHU!"
        out.append(line)

        if MAHJONG_INDEX in cards and not gs.round_over and gs.mahjong_wish is None:
            wish = ai_wish(policy, gs, p, kind, rng)
            gs.set_mahjong_wish(wish)
            log_event(st, {"t": "wish", "p": p, "rank": wish})
            out.append(f"  -> P{p} wishes for {RANK_NAMES.get(wish, wish or 'nothing')}")

        if p in gs.finish_order and gs.finish_order[-1] == p:
            out.append(f"  * P{p} goes out (#{len(gs.finish_order)})")

        if not gs.round_over and len(gs.trick_history) > tricks_before:
            winner, tcards, _ = gs.trick_history[-1]
            pts = sum(CARD_POINTS[c] for c in tcards)
            out.append(f"-- P{winner} wins the trick ({pts:+d} pts) --")


def finish_round(st: dict, out: list[str]):
    gs: GameState = st["gs"]
    seat: int = st["seat"]
    a, b = gs.round_scores
    st["cum"][0] += a
    st["cum"][1] += b
    decls = list(gs.tichu_declarations)
    log_event(st, {
        "t": "end",
        "finish": [f"P{p}" for p in gs.finish_order],
        "scores": [a, b],
        "tichu_declarations": decls,
    })
    os.makedirs(STATE_DIR, exist_ok=True)
    log_path = os.path.join(STATE_DIR, f"round_{st['round_no']:03d}.json")
    with open(log_path, "w") as f:
        json.dump(st["round_log"], f, indent=1)
    st["results"].append((a, b))
    st["pending"] = ("round_over",)

    my_team = 0 if seat in TEAM_A else 1
    my, opp = (a, b) if my_team == 0 else (b, a)
    out.append("")
    out.append(f"ROUND {st['round_no']} OVER")
    out.append(f"  Finish order: {' -> '.join(f'P{p}' for p in gs.finish_order)}")
    for p in range(4):
        d = decls[p]
        if d:
            ok = gs.finish_order[0] == p
            kind_s = "Grand Tichu" if d == 2 else "Tichu"
            bonus = (200 if d == 2 else 100) * (1 if ok else -1)
            out.append(f"  P{p} {kind_s}: {'made' if ok else 'FAILED'} ({bonus:+d})")
    out.append(f"  Round score: Team A {a:+d} | Team B {b:+d}")
    out.append(f"  Your round: {my:+d} vs {opp:+d}  ({'WIN' if my > opp else 'LOSS' if my < opp else 'DRAW'})")
    out.append(f"  Cumulative after {st['round_no']} round(s): "
               f"Team A {st['cum'][0]} | Team B {st['cum'][1]}")
    out.append(f"  Full-info log: {log_path}")
    out.append("Run `new` for the next round.")


# ── Pending-decision display ───────────────────────────────────────────────


def print_pending(st: dict):
    gs: GameState = st["gs"]
    seat: int = st["seat"]
    pending = st["pending"]
    kind = pending[0]

    if kind == "round_over":
        print("[round over — run `new` to start the next round]")
        return

    if kind == "gt":
        print(f"\n== GRAND TICHU decision (you are P{seat}) ==")
        print(f"Your first 8 cards: {pc(pending[1])}")
        print("Reply: act 1 (declare GT) or act 0 (no)")
        return

    if kind == "pass":
        hand = pending[1]
        print(f"\n== PASSING decision (you are P{seat}) ==")
        print(f"Your hand: {pc(gs.hands[seat])}")
        for i, c in enumerate(hand):
            print(f"  {i:2d}) {card_name(c)}")
        left, partner, right = (seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4
        print(f"Reply: pass <to-P{left}(left/opp)> <to-P{partner}(partner)> <to-P{right}(right/opp)>  (positions)")
        return

    if kind == "dragon":
        print(f"\n== DRAGON GIVE decision (you are P{seat}) ==")
        print(f"Give the trick to an opponent: options {pending[1]}")
        print("Reply: act <opponent seat>")
        return

    if kind == "wish":
        print(f"\n== MAHJONG WISH decision (you are P{seat}) ==")
        print("Reply: act <rank 2-14> or act 0 for no wish")
        return

    # play decision
    _, legal, order = pending
    pts_a, pts_b = team_points(gs)
    print(f"\n== PLAY decision — you are P{seat} "
          f"(team {'A' if seat in TEAM_A else 'B'}), round {st['round_no']}, "
          f"trick {gs.trick_num + 1} ==")
    for off, rel in [(1, "left"), (2, "partner"), (3, "right")]:
        p = (seat + off) % 4
        flags = ""
        if gs.tichu_declarations[p] == 1:
            flags += " [T!]"
        elif gs.tichu_declarations[p] == 2:
            flags += " [GT!]"
        if p in gs.finish_order:
            flags += f" (out #{gs.finish_order.index(p) + 1})"
        print(f"  P{p} ({rel}): {len(gs.hands[p])} cards{flags}")
    if gs.trick.cards_on_table:
        combo = gs.trick.leading_combo
        print(f"Table: {pc(gs.trick.cards_on_table)} "
              f"({COMBO_NAMES.get(combo.type, '?') if combo else '?'}, "
              f"led by P{gs.trick.leading_player}, {gs.trick.passes} pass(es))")
    else:
        print("Table: empty — you lead")
    if gs.mahjong_wish:
        print(f"Active wish: {RANK_NAMES.get(gs.mahjong_wish, gs.mahjong_wish)}")
    mine = ""
    if gs.tichu_declarations[seat] == 1:
        mine = " [you called TICHU]"
    elif gs.tichu_declarations[seat] == 2:
        mine = " [you called GRAND TICHU]"
    print(f"Points captured so far: Team A {pts_a}, Team B {pts_b}{mine}")
    print(f"Your hand ({len(gs.hands[seat])}): {pc(gs.hands[seat])}")
    print("Legal plays:")
    for disp, orig in enumerate(order):
        cards, call_tichu = legal[orig]
        label = pc(cards) if cards else "Pass"
        if cards:
            label += f"  ({combo_name(cards)})"
        if call_tichu:
            label += "  + call TICHU"
        print(f"  {disp:3d}) {label}")
    print("Reply: act <n>")


# ── Commands ───────────────────────────────────────────────────────────────


def cmd_new(args):
    st = None if args.reset else load_state()
    if st is None:
        st = {
            "checkpoint": args.checkpoint or DEFAULT_CHECKPOINT,
            "seat": args.seat if args.seat is not None else 0,
            "rng": random.Random(args.seed),
            "round_no": 0,
            "cum": [0, 0],
            "results": [],
        }
    else:
        if args.checkpoint:
            st["checkpoint"] = args.checkpoint
        if args.seat is not None:
            st["seat"] = args.seat
        if st.get("pending", ("round_over",))[0] != "round_over":
            print("[abandoning unfinished round]")

    bundle = load_bundle(st)
    policy, pass_policy, gt_policy, kind, info = bundle
    st["kind"] = kind
    st["round_no"] += 1
    st["round_log"] = []
    st["await_pass_report"] = False

    seed = args.seed if (args.seed is not None and st["round_no"] == 1) else st["rng"].randint(0, 10 ** 9)
    gs = GameState(track_history=(kind == "history"))
    gs.reset(seed=seed, with_calling=True, with_passing=True)
    st["gs"] = gs

    games = info["games_played"]
    games_s = f"{games:,}" if isinstance(games, int) else str(games)
    print(f"[round {st['round_no']} | seed {seed} | you are P{st['seat']} "
          f"(team {'A' if st['seat'] in TEAM_A else 'B'}) | "
          f"bot: {kind}, {games_s} games"
          f"{', pass-net' if pass_policy else ''}{', gt-net' if gt_policy else ''}]")

    for line in advance(st, bundle):
        print(line)
    save_state(st)
    print_pending(st)


def _respond_and_advance(st, bundle):
    for line in advance(st, bundle):
        print(line)
    save_state(st)
    print_pending(st)


def cmd_act(args):
    st = load_state()
    if st is None:
        print("No session. Run `new` first.")
        sys.exit(1)
    bundle = load_bundle(st)
    gs: GameState = st["gs"]
    seat = st["seat"]
    pending = st["pending"]
    kind = pending[0]
    n = args.n

    if kind == "round_over":
        print("[round over — run `new`]")
        return
    if kind == "pass":
        print("Pending decision is passing — use `pass <l> <p> <r>`.")
        sys.exit(1)

    if kind == "gt":
        decl = bool(n)
        gs.declare_grand_tichu(seat, decl)
        log_event(st, {"t": "gt", "p": seat, "declare": decl, "h8": pc(pending[1])})
        print(f"You {'DECLARE GRAND TICHU' if decl else 'pass on Grand Tichu'}.")
    elif kind == "dragon":
        if n not in pending[1]:
            print(f"Pick from {pending[1]}")
            sys.exit(1)
        gs.give_dragon_trick(seat, n)
        log_event(st, {"t": "dragon", "p": seat, "to": n})
        print(f"You give the Dragon trick to P{n}.")
    elif kind == "wish":
        if n != 0 and not (2 <= n <= 14):
            print("Wish must be 0 (none) or 2-14.")
            sys.exit(1)
        wish = None if n == 0 else n
        gs.set_mahjong_wish(wish)
        log_event(st, {"t": "wish", "p": seat, "rank": wish})
        print(f"Wish: {RANK_NAMES.get(wish, 'none')}")
    elif kind == "play":
        _, legal, order = pending
        if not (0 <= n < len(order)):
            print(f"Pick 0-{len(order) - 1}")
            sys.exit(1)
        action = legal[order[n]]
        cards, call_tichu = action
        ev = {"t": "play", "p": seat, "trick": gs.trick_num,
              "table_before": pc(gs.trick.cards_on_table),
              "hands": hands_snapshot(gs)}
        gs.play(seat, action)
        ev.update({"cards": pc(cards), "combo": combo_name(cards),
                   "tichu": call_tichu})
        log_event(st, ev)
        print(f"You play: {pc(cards) if cards else 'Pass'}"
              f"{' + TICHU!' if call_tichu else ''}")
        if seat in gs.finish_order and gs.finish_order[-1] == seat:
            print(f"* You go out (#{len(gs.finish_order)})")
        if MAHJONG_INDEX in cards and not gs.round_over and gs.mahjong_wish is None:
            st["pending"] = ("wish",)
            save_state(st)
            print_pending(st)
            return
    else:
        print(f"Unknown pending kind {kind}")
        sys.exit(1)

    _respond_and_advance(st, bundle)


def cmd_pass(args):
    st = load_state()
    if st is None or st["pending"][0] != "pass":
        print("No pending passing decision.")
        sys.exit(1)
    bundle = load_bundle(st)
    gs: GameState = st["gs"]
    seat = st["seat"]
    hand = st["pending"][1]
    idxs = [args.left, args.partner, args.right]
    if len(set(idxs)) != 3 or not all(0 <= i < len(hand) for i in idxs):
        print(f"Need 3 distinct positions in 0-{len(hand) - 1}.")
        sys.exit(1)
    left, partner, right = (hand[i] for i in idxs)
    st["hand_pre_pass"] = set(gs.hands[seat])
    st["outgoing"] = {left, partner, right}
    st["await_pass_report"] = True
    gs.pass_cards(seat, left, partner, right)
    log_event(st, {"t": "pass", "p": seat, "left": pc([left]),
                   "partner": pc([partner]), "right": pc([right])})
    print(f"You pass: left {card_name(left)}, partner {card_name(partner)}, "
          f"right {card_name(right)}")
    _respond_and_advance(st, bundle)


def cmd_show(_args):
    st = load_state()
    if st is None:
        print("No session. Run `new` first.")
        sys.exit(1)
    print_pending(st)


def cmd_log(args):
    st = load_state()
    n = args.round
    if n is None:
        if st is None:
            print("No session.")
            sys.exit(1)
        n = st["round_no"] if st["pending"][0] == "round_over" else st["round_no"] - 1
    path = os.path.join(STATE_DIR, f"round_{n:03d}.json")
    if not os.path.exists(path):
        print(f"No log for round {n} ({path})")
        sys.exit(1)
    with open(path) as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new", help="start a new round (or session)")
    p_new.add_argument("--seed", type=int, default=None)
    p_new.add_argument("--seat", type=int, default=None, choices=[0, 1, 2, 3])
    p_new.add_argument("--checkpoint", type=str, default=None)
    p_new.add_argument("--reset", action="store_true",
                       help="wipe the session (cumulative score, round counter)")
    p_new.set_defaults(fn=cmd_new)

    p_act = sub.add_parser("act", help="answer the pending decision")
    p_act.add_argument("n", type=int)
    p_act.set_defaults(fn=cmd_act)

    p_pass = sub.add_parser("pass", help="answer the pending passing decision")
    p_pass.add_argument("left", type=int)
    p_pass.add_argument("partner", type=int)
    p_pass.add_argument("right", type=int)
    p_pass.set_defaults(fn=cmd_pass)

    p_show = sub.add_parser("show", help="re-print the pending decision")
    p_show.set_defaults(fn=cmd_show)

    p_log = sub.add_parser("log", help="print full-info log of a finished round")
    p_log.add_argument("round", type=int, nargs="?", default=None)
    p_log.set_defaults(fn=cmd_log)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
