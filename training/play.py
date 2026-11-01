"""Play against a trained Tichu AI in the terminal.

You are player 0 by default (team A with player 2).
The AI controls the other three seats.

Usage:
    python -m training.play
    python -m training.play --checkpoint kaggle_runs/2026-08-01_110222_standard/tichubot_outputs/checkpoints/history_standard_gpu/latest.pt
    python -m training.play --checkpoint kaggle_runs/2026-08-01_110222_standard/tichubot_outputs/checkpoints/history_standard_gpu/latest.pt --seat 2
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from collections.abc import Mapping

import numpy as np
import torch

from tichu_env.cards import MAHJONG_INDEX, NUM_CARDS, Rank
from tichu_env.combinations import classify_cards
from tichu_env.display import fmt_cards, fmt_hand, COMBO_NAMES, _RESET, _BOLD
from tichu_env.encoding import (
    encode_pass_obs,
    encode_pass_source_ids,
    encode_pass_target_ids,
    encode_static,
    encode_history,
    PLAY_TOKEN_SIZE,
)
from tichu_env.constants import MAX_ACTIONS
from tichu_env.game_state import GameState, TEAM_A, TEAM_B

from training.checkpoints import DEFAULT_PLAY_CHECKPOINT
from training.model_embed import ACTION_DIM
from training.model_passing import TichuPassingPolicy
from training.observations import encode_policy_static


DEFAULT_CHECKPOINT = DEFAULT_PLAY_CHECKPOINT
DIM    = "\033[2m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
RED    = "\033[31m"
UNDO = "__undo__"
PassChoice = tuple[int, int, int]


def _team(p: int) -> str:
    return "A" if p in TEAM_A else "B"


def _divider():
    print(f"{DIM}{'─' * 60}{_RESET}")


def print_table(gs: GameState, human: int):
    print()
    _divider()

    for offset in [1, 2, 3]:
        p = (human + offset) % 4
        n = len(gs.hands[p])
        rel = ["", "Left", "Partner", "Right"][offset]
        status = ""
        if p in gs.finish_order:
            pos = gs.finish_order.index(p) + 1
            status = f" {GREEN}(out #{pos}){_RESET}"
        tichu = ""
        if gs.tichu_declarations[p] == 1:
            tichu = f" {RED}[T!]{_RESET}"
        elif gs.tichu_declarations[p] == 2:
            tichu = f" {RED}[GT!]{_RESET}"
        bar = "█" * n + "░" * (14 - n)
        print(f"  P{p} {rel} (team {_team(p)}){tichu}: {DIM}{bar}{_RESET} {n} cards{status}")

    print()
    if gs.trick.cards_on_table:
        combo = gs.trick.leading_combo
        type_name = COMBO_NAMES.get(combo.type, "") if combo else ""
        leader = gs.trick.leading_player
        print(f"  {YELLOW}Table:{_RESET} {fmt_cards(gs.trick.cards_on_table)}  ({type_name}, by P{leader})")
        print(f"  {DIM}Passes: {gs.trick.passes}{_RESET}")
    else:
        print(f"  {YELLOW}Table:{_RESET} empty — new trick")

    if gs.mahjong_wish:
        rank_names = {r.value: r.name.title() for r in Rank}
        print(f"  {CYAN}Wish: {rank_names.get(gs.mahjong_wish, gs.mahjong_wish)}{_RESET}")

    print()
    tichu = ""
    if gs.tichu_declarations[human] == 1:
        tichu = f" {RED}[T!]{_RESET}"
    elif gs.tichu_declarations[human] == 2:
        tichu = f" {RED}[GT!]{_RESET}"
    print(f"  {_BOLD}You (P{human}, team {_team(human)}){tichu}:{_RESET}")
    print(f"  {fmt_hand(gs.hands[human])}")
    _divider()


def choose_action(actions: list[tuple[list[int], bool]]) -> int | str:
    # Sort actions for better display
    # Key: (is_not_pass, combo_type, min_rank, call_tichu)
    def sort_key(idx):
        cards, call_tichu = actions[idx]
        if not cards:
            return (0, 0, 0, 1 if call_tichu else 0)
        
        combo = classify_cards(cards)
        from tichu_env.cards import card_rank
        
        # Use integer values for robust sorting (Rank and ComboType are Enums)
        type_val = combo.type.value if combo else 99
        
        # Get ranks for all cards in combo, using 0 for special cards
        # (Actually card_rank returns None for specials, we convert to int)
        ranks = []
        for c in cards:
            r = card_rank(c)
            ranks.append(r.value if r else 0)
        min_rank = min(ranks)
        
        return (1, type_val, min_rank, 1 if call_tichu else 0)

    sorted_indices = sorted(range(len(actions)), key=sort_key)

    print(f"\n  {_BOLD}Legal plays:{_RESET}")
    for i, original_idx in enumerate(sorted_indices):
        cards, call_tichu = actions[original_idx]
        tichu_tag = f"  {RED}+ TICHU{_RESET}" if call_tichu else ""
        if not cards:
            label = f"{DIM}Pass{_RESET}{tichu_tag}"
        else:
            combo = classify_cards(cards)
            type_name = COMBO_NAMES.get(combo.type, "") if combo else ""
            if combo and combo.is_bomb:
                type_name = f"{RED}{_BOLD}{type_name}{_RESET}"
            label = f"{fmt_cards(cards)}  {DIM}({type_name}){_RESET}{tichu_tag}"
        print(f"    {CYAN}{i:3d}{_RESET}) {label}")

    while True:
        try:
            raw = input(f"\n  {_BOLD}Your play [0-{len(actions)-1}]: {_RESET}").strip()
            if not raw:
                continue
            if raw.lower() in {"u", "undo"}:
                return UNDO
            choice = int(raw)
            if 0 <= choice < len(actions):
                return sorted_indices[choice]
            print(f"  {RED}Pick 0-{len(actions)-1}{_RESET}")
        except ValueError:
            print(f"  {RED}Enter a number{_RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)


def choose_wish() -> int | None | str:
    print(f"\n  {CYAN}You played the Mahjong! Declare a wish?{_RESET}")
    print(f"    {DIM}Enter rank 2-14 (J=11, Q=12, K=13, A=14) or 0 for no wish:{_RESET}")
    while True:
        try:
            raw = input(f"  {_BOLD}Wish: {_RESET}").strip()
            if not raw:
                continue
            if raw.lower() in {"u", "undo"}:
                return UNDO
            val = int(raw)
            if val == 0:
                return None
            if 2 <= val <= 14:
                return val
            print(f"  {RED}Enter 2-14 or 0{_RESET}")
        except ValueError:
            print(f"  {RED}Enter a number{_RESET}")
        except (EOFError, KeyboardInterrupt):
            return None


def choose_gt(hand_8: list[int]) -> bool | str:
    print(f"\n  {RED}{_BOLD}Grand Tichu Phase{_RESET}")
    print(f"  Your first 8 cards: {fmt_cards(hand_8)}")
    while True:
        raw = input(f"  {_BOLD}Declare Grand Tichu? (y/n): {_RESET}").strip().lower()
        if raw in {"y", "yes"}: return True
        if raw in {"n", "no", ""}: return False
        if raw in {"u", "undo"}: return UNDO
        print(f"  {RED}Enter y or n{_RESET}")


@torch.no_grad()
def ai_gt(gt_policy, hand_8: list[int], device: torch.device) -> bool:
    if gt_policy is None: return False
    h8_bin = torch.as_tensor(_hand_to_binary(hand_8)[None, :], device=device)
    decl, _ = gt_policy.act(h8_bin)
    return bool(decl.item())


def choose_dragon_recipient(gs: GameState, human: int) -> int | str:
    options = gs.get_dragon_give_options(human)
    print(f"\n  {RED}{_BOLD}You won a trick with the Dragon! Give it to an opponent:{_RESET}")
    for opp in options:
        print(f"    {CYAN}{opp}{_RESET}) P{opp} (team {_team(opp)})")
    while True:
        try:
            raw = input(f"  {_BOLD}Give to: {_RESET}").strip()
            if raw.lower() in {"u", "undo"}:
                return UNDO
            choice = int(raw)
            if choice in options:
                return choice
            print(f"  {RED}Pick from {options}{_RESET}")
        except (ValueError, EOFError, KeyboardInterrupt):
            print(f"  {RED}Pick from {options}{_RESET}")


def _hand_to_binary(hand: list[int]) -> np.ndarray:
    vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for card in hand:
        vec[card] = 1.0
    return vec


def choose_pass_cards(hand: list[int]) -> tuple[int, int, int] | str:
    labels = ("left", "partner", "right")
    chosen: list[int] = []

    print(f"\n  {CYAN}{_BOLD}Passing Phase{_RESET}")
    print(f"  Choose one card to pass to each recipient.")

    while len(chosen) < 3:
        print(f"\n  Your hand: {fmt_cards(hand)}")
        for idx, card in enumerate(hand):
            tag = f" {DIM}[chosen]{_RESET}" if idx in chosen else ""
            print(f"    {CYAN}{idx:2d}{_RESET}) {fmt_cards([card])}{tag}")
        try:
            raw = input(
                f"\n  {_BOLD}Pick card for {labels[len(chosen)]} [0-{len(hand)-1}]: {_RESET}"
            ).strip()
            if raw.lower() in {"u", "undo"}:
                return UNDO
            choice = int(raw)
            if not (0 <= choice < len(hand)):
                print(f"  {RED}Pick 0-{len(hand)-1}{_RESET}")
                continue
            if choice in chosen:
                print(f"  {RED}Card already chosen{_RESET}")
                continue
            chosen.append(choice)
        except ValueError:
            print(f"  {RED}Enter a number{_RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)

    left, partner, right = (hand[i] for i in chosen)
    print(
        f"\n  Passing: left {fmt_cards([left])}, "
        f"partner {fmt_cards([partner])}, right {fmt_cards([right])}"
    )
    return left, partner, right


@torch.no_grad()
def ai_pass(pass_policy, gs: GameState, player: int, device: torch.device,
            rng: random.Random) -> PassChoice:
    hand = gs.get_passing_hand(player)
    if pass_policy is None:
        left, partner, right = rng.sample(hand, 3)
    else:
        pass_obs = torch.as_tensor(
            encode_pass_obs(gs, player)[None, :], device=device
        )
        hand_cards = torch.as_tensor(
            np.array(hand, dtype=np.int64)[None, :], device=device
        )
        choices_t, _ = pass_policy.act(pass_obs, hand_cards)
        idxs = choices_t[0].cpu().numpy()
        left, partner, right = (hand[int(i)] for i in idxs)

    if not gs.pass_cards(player, left, partner, right):
        raise RuntimeError(f"AI produced an illegal pass for P{player}")
    print(f"  P{player} submits passing cards")
    return left, partner, right


@torch.no_grad()
def ai_select_action(policy, gs: GameState, player: int, device: torch.device,
                     rng: random.Random, kind: str = "embed"):
    """Choose one legal AI action without changing ``gs``.

    Keeping selection separate from application lets non-terminal clients
    precompute an action for display while retaining the exact policy path
    used by :func:`ai_act`.
    """
    legal = (
        gs.get_legal_actions(player)
        if kind == "fixed"
        else gs.get_policy_legal_actions(player)
    )
    if policy is None:
        return rng.choice(legal)
    obs = torch.as_tensor(
        encode_policy_static(policy, gs, player),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    pass_src = torch.as_tensor(
        encode_pass_source_ids(gs, player), dtype=torch.int64, device=device
    ).unsqueeze(0)
    pass_tgt = torch.as_tensor(
        encode_pass_target_ids(gs, player), dtype=torch.int64, device=device
    ).unsqueeze(0)

    if kind == "fixed":
        n_legal = min(len(legal), MAX_ACTIONS)
        mask = (torch.arange(MAX_ACTIONS, device=device) < n_legal).unsqueeze(0)
        action_t, _, _ = policy.act(obs, mask)
    else:
        from training.rollout_embed import _action_vec
        n = len(legal)
        pad = np.zeros((1, n, ACTION_DIM), dtype=np.float32)
        for k, a in enumerate(legal):
            pad[0, k] = _action_vec(a)
        act_args = [
            obs,
            torch.as_tensor(pad, device=device),
            torch.tensor([n], dtype=torch.int32, device=device),
            pass_src,
            pass_tgt,
        ]
        if kind == "history":
            hist, hist_len = encode_history(gs, player)
            max_h = max(hist_len, 1)
            hist_t = torch.as_tensor(
                hist[:max_h][np.newaxis], device=device
            )
            pass_tgt = act_args.pop()
            pass_src = act_args.pop()
            act_args += [
                hist_t,
                torch.tensor([hist_len], dtype=torch.int32, device=device),
                pass_src,
                pass_tgt,
            ]
        action_t, _, _ = policy.act(*act_args)

    idx = int(action_t.item())
    return legal[idx] if idx < len(legal) else legal[0]


def _fused_seat_features(policy, gs: GameState, player: int,
                         device: torch.device) -> torch.Tensor:
    obs = encode_static(gs, player)
    pass_src = encode_pass_source_ids(gs, player)
    pass_tgt = encode_pass_target_ids(gs, player)
    history, history_len = encode_history(gs, player)
    max_history = max(int(history_len), 1)
    return policy.fused_features(
        torch.as_tensor(obs[None, :], dtype=torch.float32, device=device),
        torch.as_tensor(
            history[None, :max_history], dtype=torch.float32, device=device
        ),
        torch.as_tensor([history_len], dtype=torch.int32, device=device),
        torch.as_tensor(pass_src[None, :], dtype=torch.int64, device=device),
        torch.as_tensor(pass_tgt[None, :], dtype=torch.int64, device=device),
    )


@torch.no_grad()
def ai_select_dragon_recipient(policy, gs: GameState, player: int,
                               device: torch.device, rng: random.Random,
                               kind: str = "embed") -> int:
    """Use the deployed checkpoint's Dragon head, matching evaluation."""
    options = gs.get_dragon_give_options(player)
    if kind == "history" and policy is not None and hasattr(policy, "dragon_head"):
        logits = policy.get_dragon_gift_logits(
            _fused_seat_features(policy, gs, player, device)
        )
        choice = int(torch.distributions.Categorical(logits=logits).sample().item())
        recipient = (player + (1 if choice == 0 else 3)) % 4
        if recipient in options:
            return recipient
    return rng.choice(options)


@torch.no_grad()
def ai_select_mahjong_wish(policy, gs: GameState, player: int,
                           device: torch.device, rng: random.Random,
                           kind: str = "embed") -> int | None:
    """Use the deployed checkpoint's Mahjong head, matching evaluation."""
    if kind == "history" and policy is not None and hasattr(policy, "wish_head"):
        logits = policy.get_mahjong_wish_logits(
            _fused_seat_features(policy, gs, player, device)
        )
        choice = int(torch.distributions.Categorical(logits=logits).sample().item())
        return None if choice == 0 else choice + 1
    return rng.randint(2, 14) if rng.random() < 0.5 else None


@torch.no_grad()
def ai_act(policy, gs: GameState, player: int, device: torch.device,
           rng: random.Random, kind: str = "embed"):
    """Choose and apply one AI action (the terminal-play behavior)."""
    action = ai_select_action(policy, gs, player, device, rng, kind)
    cards, call_tichu = action[:2]

    gs.play(player, action)

    tichu_tag = f"  {RED}+ TICHU!{_RESET}" if call_tichu else ""
    cards_str = fmt_cards(cards) if cards else f"{DIM}Pass{_RESET}"
    combo = gs.trick.leading_combo if cards else None
    type_name = ""
    if combo:
        type_name = COMBO_NAMES.get(combo.type, "")
        if combo.is_bomb:
            type_name = f"{RED}{_BOLD}{type_name}{_RESET}"
    print(f"  P{player}: {cards_str}  {DIM}{type_name}{_RESET}{tichu_tag}")

    if MAHJONG_INDEX in cards:
        wish = ai_select_mahjong_wish(policy, gs, player, device, rng, kind)
        gs.set_mahjong_wish(wish)
        rank_names = {r.value: r.name.title() for r in Rank}
        wish_label = rank_names.get(wish, wish) if wish is not None else "nothing"
        print(f"    {CYAN}-> P{player} wishes for {wish_label}{_RESET}")

    if player in gs.finish_order and gs.finish_order[-1] == player:
        pos = len(gs.finish_order)
        print(f"    {GREEN}* P{player} goes out (#{pos})!{_RESET}")

    return cards


def run_replay_passing_phase(
    gs: GameState,
    pass_policy,
    device: torch.device,
    replayed_player: int,
    recorded_passes: Mapping[int, PassChoice] | None,
    rng: random.Random,
    *,
    standardize_opponent_passes: bool,
    announce: bool = True,
) -> dict[int, PassChoice]:
    """Complete replay passing, retaining recorded opponent choices when requested.

    A replayed seat always samples the current pass policy.  In recorded mode,
    the other seats submit the directed cards they used in the original round;
    this keeps the post-pass deal comparable to the played round apart from the
    replayed seat's own decision.
    """
    if standardize_opponent_passes and recorded_passes:
        required_players = set(range(4)) - {replayed_player}
        missing_players = sorted(required_players - recorded_passes.keys())
        if missing_players:
            missing = ", ".join(f"P{player}" for player in missing_players)
            raise ValueError(
                f"Recorded opponent passes are incomplete; missing {missing}"
            )

    applied: dict[int, PassChoice] = {}
    while gs.phase.name == "PASSING":
        player = gs.current_player
        fixed_choice = (
            recorded_passes.get(player)
            if standardize_opponent_passes
            and player != replayed_player
            and recorded_passes is not None
            else None
        )
        if fixed_choice is None:
            applied[player] = ai_pass(pass_policy, gs, player, device, rng)
            continue

        left, partner, right = fixed_choice
        if not gs.pass_cards(player, left, partner, right):
            raise ValueError(
                f"Recorded pass for P{player} is not legal for this replay deal"
            )
        applied[player] = fixed_choice
        if announce:
            print(f"  P{player} reuses recorded passing cards")

    return applied


def run_ai_replay(
    policy,
    pass_policy,
    initial_hands,
    device,
    kind,
    *,
    replayed_player: int | None = None,
    recorded_passes: Mapping[int, PassChoice] | None = None,
    tichu_declarations: list[int] | None = None,
    standardize_opponent_passes: bool = True,
):
    """Simulate a full round with AI in all seats using the original deal.

    When a replayed player and recorded passes are supplied, opponents retain
    the choices from the live round by default.  ``standardize_opponent_passes``
    can be disabled for a fresh all-AI pass phase.
    """
    print(f"\n{DIM}{'='*20} AI REPLAY {'='*20}{_RESET}")
    gs = GameState(track_history=(kind == "history"))
    gs.reset(with_passing=True)
    gs.hands = [set(h) for h in initial_hands]
    gs._invalidate_cache()
    if tichu_declarations is not None:
        gs.tichu_declarations = list(tichu_declarations)

    replay_rng = random.Random()
    if standardize_opponent_passes and replayed_player is not None and recorded_passes:
        print(f"  Reusing recorded opponent passes; P{replayed_player} uses the AI pass.")
    elif standardize_opponent_passes:
        print(f"  No recorded passes available; all seats use AI passes.")

    run_replay_passing_phase(
        gs,
        pass_policy,
        device,
        replayed_player if replayed_player is not None else -1,
        recorded_passes,
        replay_rng,
        standardize_opponent_passes=standardize_opponent_passes,
    )
    
    # AI Play Phase
    while not gs.round_over:
        if gs.dragon_give_pending is not None:
            giver = gs.dragon_give_pending
            recipient = ai_select_dragon_recipient(
                policy, gs, giver, device, replay_rng, kind
            )
            gs.give_dragon_trick(giver, recipient)
            print(f"  AI P{giver} gives Dragon to P{recipient}")
            continue
            
        p = gs.current_player
        # Standard AI act
        ai_act(policy, gs, p, device, replay_rng, kind)
        
    print(f"\n  {_BOLD}AI REPLAY RESULT:{_RESET}")
    print(f"  Finish order: {' -> '.join(f'P{p}' for p in gs.finish_order)}")
    a, b = gs.round_scores
    print(f"  Team A (P0+P2): {a:+d}  |  Team B (P1+P3): {b:+d}")
    print(f"{DIM}{'='*51}{_RESET}\n")


def play_game(policy, pass_policy, gt_policy, human: int, device: torch.device,
              rng: random.Random, seed: int, kind: str = "embed",
              replay_opponent_passes: str = "recorded"):
    gs = GameState(track_history=(kind == "history"))
    gs.reset(seed=seed, with_calling=True, with_passing=True)
    
    # DEALING / GT PHASE
    # In GameState.reset(with_calling=True), the game starts in GRAND_TICHU phase.
    
    undo_stack: list[tuple[GameState, object]] = []

    # Collect GT declarations in a proper phase loop
    while gs.phase.name == "GRAND_TICHU":
        p = gs.current_player
        if p == human:
            # We simulate the "see 8" by filtering the full hand
            # The hand is already full in gs.hands, but we only show 8
            # (In a real game you'd get 8 then 6, but gs deals all 14)
            h14 = sorted(list(gs.hands[human]))
            # To match the "deal 8" exactly, we use the first 8 from the shuffled deck
            # for that player. But for simplicity, we just take the first 8 from the sorted hand.
            h8 = h14[:8]
            choice = choose_gt(h8)
            if choice == UNDO:
                return UNDO
            gs.declare_grand_tichu(human, choice)
        else:
            # AI GT Decision
            h14 = sorted(list(gs.hands[p]))
            h8 = h14[:8]
            decl = ai_gt(gt_policy, h8, device)
            gs.declare_grand_tichu(p, decl)
            if decl:
                print(f"  P{p} declared GRAND TICHU!")

    # At this point, phase has transitioned to PASSING
    
    # CAPTURE INITIAL STATE
    initial_hands = [list(h) for h in gs.hands]
    initial_tichu_declarations = list(gs.tichu_declarations)
    recorded_passes: dict[int, PassChoice] = {}
    
    initial_hand = set(gs.hands[human])
    outgoing_cards: set[int] = set()

    print(f"\n{'#' * 60}")
    print(f"  {_BOLD}TICHU{_RESET} — You are P{human} (team {_team(human)})")
    print(f"  Partner: P{(human+2)%4}  |  Opponents: P{(human+1)%4}, P{(human+3)%4}")
    print(f"{'#' * 60}")
    print(f"  {DIM}Type 'undo' at your prompts to revert your last manual action.{_RESET}")

    while gs.phase.name == "PASSING":
        player = gs.current_player
        if player == human:
            print(f"\n  {_BOLD}Initial hand:{_RESET} {fmt_cards(gs.hands[human])}")
            choice = choose_pass_cards(gs.get_passing_hand(human))
            if choice == UNDO:
                if undo_stack:
                    gs, rng_state = undo_stack.pop()
                    rng.setstate(rng_state)
                    print(f"  {YELLOW}Undid last action.{_RESET}")
                else:
                    print(f"  {RED}Nothing to undo yet.{_RESET}")
                continue
            left, partner, right = choice
            outgoing_cards = {left, partner, right}
            undo_stack.append((copy.deepcopy(gs), rng.getstate()))
            gs.pass_cards(human, left, partner, right)
            recorded_passes[human] = choice
        else:
            recorded_passes[player] = ai_pass(pass_policy, gs, player, device, rng)

    kept_cards = initial_hand - outgoing_cards
    received_cards = sorted(set(gs.hands[human]) - kept_cards)
    print(f"\n  {_BOLD}You received:{_RESET} {fmt_cards(received_cards)}")
    print(f"  {_BOLD}Hand After Passing:{_RESET} {fmt_cards(gs.hands[human])}")

    while not gs.round_over:
        if gs.dragon_give_pending is not None:
            giver = gs.dragon_give_pending
            if giver == human:
                print_table(gs, human)
                recipient = choose_dragon_recipient(gs, human)
                if recipient == UNDO:
                    if undo_stack:
                        gs, rng_state = undo_stack.pop()
                        rng.setstate(rng_state)
                        print(f"  {YELLOW}Undid last action.{_RESET}")
                    else:
                        print(f"  {RED}Nothing to undo yet.{_RESET}")
                    continue
                undo_stack.append((copy.deepcopy(gs), rng.getstate()))
            else:
                recipient = ai_select_dragon_recipient(
                    policy, gs, giver, device, rng, kind
                )
                print(f"  P{giver} gives Dragon trick to P{recipient}")
            gs.give_dragon_trick(giver, recipient)
            continue

        player = gs.current_player

        if player == human:
            print_table(gs, human)
            actions = gs.get_legal_actions(human)
            choice = choose_action(actions)
            if choice == UNDO:
                if undo_stack:
                    gs, rng_state = undo_stack.pop()
                    rng.setstate(rng_state)
                    print(f"  {YELLOW}Undid last action.{_RESET}")
                else:
                    print(f"  {RED}Nothing to undo yet.{_RESET}")
                continue
            action = actions[choice]
            cards, call_tichu = action
            undo_stack.append((copy.deepcopy(gs), rng.getstate()))
            gs.play(human, action)

            tichu_tag = f" {RED}+ TICHU!{_RESET}" if call_tichu else ""
            if cards:
                combo = classify_cards(cards)
                type_name = COMBO_NAMES.get(combo.type, "") if combo else ""
                print(f"\n  {GREEN}You played: {fmt_cards(cards)} ({type_name}){_RESET}{tichu_tag}")
            else:
                print(f"\n  {DIM}You passed{_RESET}{tichu_tag}")

            if MAHJONG_INDEX in cards:
                wish = choose_wish()
                if wish == UNDO:
                    gs, rng_state = undo_stack.pop()
                    rng.setstate(rng_state)
                    print(f"  {YELLOW}Undid last action.{_RESET}")
                    continue
                if wish:
                    gs.set_mahjong_wish(wish)
                    print(f"  {CYAN}Wish set: rank {wish}{_RESET}")

            if human in gs.finish_order and gs.finish_order[-1] == human:
                pos = len(gs.finish_order)
                print(f"\n  {GREEN}{_BOLD}* You went out (#{pos})!{_RESET}")
        else:
            action = ai_act(policy, gs, player, device, rng, kind)

            if gs.dragon_give_pending is not None and gs.dragon_give_pending != human:
                giver = gs.dragon_give_pending
                recipient = ai_select_dragon_recipient(
                    policy, gs, giver, device, rng, kind
                )
                gs.give_dragon_trick(giver, recipient)
                print(f"  P{giver} gives Dragon trick to P{recipient}")

    print()
    _divider()
    print(f"  {_BOLD}ROUND OVER{_RESET}")
    print(f"  Finish order: {' -> '.join(f'P{p}' for p in gs.finish_order)}")
    a, b = gs.round_scores
    print(f"  Team A (P0+P2): {a:+d}")
    print(f"  Team B (P1+P3): {b:+d}")

    your_team = 0 if human in TEAM_A else 1
    your_score = gs.round_scores[your_team]
    opp_score  = gs.round_scores[1 - your_team]
    if your_score > opp_score:
        print(f"\n  {GREEN}{_BOLD}You win!{_RESET}")
    elif your_score < opp_score:
        print(f"\n  {RED}You lose.{_RESET}")
    else:
        print(f"\n  {YELLOW}Draw.{_RESET}")
    _divider()

    # Offer AI replay
    ans = input(f"\n  Show AI replay for this deal? (y/n): ").lower().strip()
    if ans == 'y':
        run_ai_replay(
            policy,
            pass_policy,
            initial_hands,
            device,
            kind,
            replayed_player=human,
            recorded_passes=recorded_passes,
            tichu_declarations=initial_tichu_declarations,
            standardize_opponent_passes=(replay_opponent_passes == "recorded"),
        )

    return a, b


def load_play_bundle(checkpoint: str, pass_checkpoint: str | None,
                     device: torch.device):
    """Load a play policy plus bundled pass/GT sub-models from a checkpoint.

    Returns (policy, pass_policy, gt_policy, kind, info) where kind is
    "history" | "embed" | "fixed" and info is a dict with games_played and
    pass_source for display.
    """
    from training.checkpoints import load_policy_checkpoint, load_raw_checkpoint

    loaded = load_policy_checkpoint(checkpoint, device, strict=False)
    ckpt = loaded.checkpoint
    policy = loaded.policy
    kind = loaded.kind
    pass_policy = None
    pass_source = None
    pass_state = ckpt.get("pass_model")
    if pass_checkpoint is not None:
        pass_ckpt, _ = load_raw_checkpoint(pass_checkpoint, device)
        pass_state = pass_ckpt.get("pass_model")
        pass_source = pass_checkpoint
    elif pass_state is not None:
        pass_source = "bundled"
    if pass_state is not None:
        pass_policy = TichuPassingPolicy().to(device)
        pass_policy.load_state_dict(pass_state)
        pass_policy.eval()

    gt_policy = None
    gt_state = ckpt.get("gt_model")
    if gt_state is not None:
        from training.model_gt import TichuGrandTichuPolicy
        cfg = ckpt.get("config")
        cfg_value = (
            (lambda key, default: cfg.get(key, default))
            if isinstance(cfg, dict)
            else (lambda key, default: getattr(cfg, key, default))
        )
        gt_policy = TichuGrandTichuPolicy(
            hidden_dim=cfg_value("gt_hidden_dim", 64),
            n_layers=cfg_value("gt_n_layers", 2),
        ).to(device)
        gt_policy.load_state_dict(gt_state)
        gt_policy.eval()

    info = {
        "games_played": ckpt.get("games_played", "?"),
        "pass_source": pass_source,
    }
    return policy, pass_policy, gt_policy, kind, info


def main():
    parser = argparse.ArgumentParser(description="Play Tichu against a trained AI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--pass-checkpoint", type=str, default=None)
    parser.add_argument("--seat",   type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--replay-opponent-passes",
        choices=("recorded", "ai"),
        default="recorded",
        help="Use the live round's opponent passes in AI replay (default: recorded).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, pass_policy, gt_policy, kind, info = load_play_bundle(
        args.checkpoint, args.pass_checkpoint, device
    )
    pass_source = info["pass_source"]

    games_trained = info["games_played"]
    trained_str = f"{games_trained:,}" if isinstance(games_trained, int) else str(games_trained)
    print(f"Loaded {args.checkpoint}  (trained on {trained_str} games, {kind})")
    if pass_policy is not None:
        print(f"Loaded passing model ({pass_source})")
    else:
        print("No passing model found; AI will pass randomly")
    if gt_policy is not None:
        print("Loaded Grand Tichu model")

    rng = random.Random(args.seed)
    total_a = total_b = 0
    rounds = 0

    try:
        while True:
            seed = rng.randint(0, 99999)
            res = play_game(
                policy,
                pass_policy,
                gt_policy,
                args.seat,
                device,
                rng,
                seed,
                kind,
                args.replay_opponent_passes,
            )
            if res == UNDO:
                print(f"  {YELLOW}Restarting round...{_RESET}")
                continue
            a, b = res
            total_a += a
            total_b += b
            rounds += 1
            print(f"\n  Cumulative ({rounds} rounds):  Team A {total_a}  vs  Team B {total_b}")
            try:
                input("\n  Press Enter for next round (Ctrl-C to quit)...")
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
