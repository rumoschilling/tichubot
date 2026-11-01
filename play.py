#!/usr/bin/env python3
"""Play Tichu interactively against 3 AI opponents.

Usage: python play.py [seed] [--seat N]
  seat: 0-3, which player you control (default 0)
"""

import random
import sys

from tichu_env.cards import (
    DRAGON_INDEX,
    DOG_INDEX,
    MAHJONG_INDEX,
    PHOENIX_INDEX,
    Rank,
    card_name,
    card_rank,
)
from tichu_env.display import color_card, fmt_cards, fmt_hand, COMBO_NAMES, _RESET, _BOLD
from tichu_env.combinations import classify_cards, CombinationType
from tichu_env.game_state import GameState, TEAM_A, TEAM_B


DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"


def team_label(p: int) -> str:
    return "A" if p in TEAM_A else "B"


def print_divider():
    print(f"{DIM}{'─' * 60}{_RESET}")


def print_table(gs: GameState, human: int):
    """Print the current game state from the human's perspective."""
    print()
    print_divider()

    # Other players' card counts and status
    for offset in [1, 2, 3]:
        p = (human + offset) % 4
        team = team_label(p)
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
        print(f"  P{p} {rel} (team {team}){tichu}: {DIM}{bar}{_RESET} {n} cards{status}")

    # Table
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

    # Human's hand
    team = team_label(human)
    tichu = ""
    if gs.tichu_declarations[human] == 1:
        tichu = f" {RED}[T!]{_RESET}"
    elif gs.tichu_declarations[human] == 2:
        tichu = f" {RED}[GT!]{_RESET}"
    print(f"  {_BOLD}You (P{human}, team {team}){tichu}:{_RESET}")
    print(f"  {fmt_hand(gs.hands[human])}")
    print_divider()


def choose_action(
    actions: list[tuple[list[int], bool]], gs: GameState, human: int
) -> int:
    """Let the human pick from legal actions."""
    print(f"\n  {_BOLD}Legal plays:{_RESET}")

    for i, (cards, call_tichu) in enumerate(actions):
        tichu_tag = f"  {RED}+ TICHU{_RESET}" if call_tichu else ""
        if not cards:
            label = f"{DIM}Pass{_RESET}{tichu_tag}"
        else:
            cards_str = fmt_cards(cards)
            combo = classify_cards(cards)
            type_name = COMBO_NAMES.get(combo.type, "") if combo else ""
            if combo and combo.is_bomb:
                type_name = f"{RED}{_BOLD}{type_name}{_RESET}"
            label = f"{cards_str}  {DIM}({type_name}){_RESET}{tichu_tag}"
        print(f"    {CYAN}{i:3d}{_RESET}) {label}")

    while True:
        try:
            raw = input(f"\n  {_BOLD}Your play [0-{len(actions)-1}]: {_RESET}").strip()
            if not raw:
                continue
            choice = int(raw)
            if 0 <= choice < len(actions):
                return choice
            print(f"  {RED}Pick 0-{len(actions)-1}{_RESET}")
        except ValueError:
            print(f"  {RED}Enter a number{_RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)


def choose_wish() -> int | None:
    """Let the human declare a Mahjong wish."""
    print(f"\n  {CYAN}You played the Mahjong! Declare a wish?{_RESET}")
    print(f"    {DIM}Enter rank 2-14 (J=11, Q=12, K=13, A=14) or 0 for no wish:{_RESET}")
    while True:
        try:
            raw = input(f"  {_BOLD}Wish: {_RESET}").strip()
            if not raw:
                continue
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


def choose_dragon_recipient(gs: GameState, human: int) -> int:
    """Let the human choose who to give the Dragon trick to."""
    options = gs.get_dragon_give_options(human)
    print(f"\n  {RED}{_BOLD}You won a trick with the Dragon! Give it to an opponent:{_RESET}")
    for opp in options:
        team = team_label(opp)
        print(f"    {CYAN}{opp}{_RESET}) P{opp} (team {team})")
    while True:
        try:
            raw = input(f"  {_BOLD}Give to: {_RESET}").strip()
            choice = int(raw)
            if choice in options:
                return choice
            print(f"  {RED}Pick from {options}{_RESET}")
        except (ValueError, EOFError, KeyboardInterrupt):
            print(f"  {RED}Pick from {options}{_RESET}")


def ai_play(gs: GameState, rng: random.Random):
    """AI plays randomly."""
    player = gs.current_player
    actions = gs.get_legal_actions(player)
    action = rng.choice(actions)
    cards, call_tichu = action
    gs.play(player, action)

    cards_str = fmt_cards(cards) if cards else f"{DIM}Pass{_RESET}"
    combo = classify_cards(cards) if cards else None
    type_name = ""
    if combo:
        type_name = COMBO_NAMES.get(combo.type, "")
        if combo.is_bomb:
            type_name = f"{RED}{_BOLD}{type_name}{_RESET}"
    tichu_tag = f"  {RED}+ TICHU!{_RESET}" if call_tichu else ""
    print(f"  P{player}: {cards_str}  {DIM}{type_name}{_RESET}{tichu_tag}")

    # AI mahjong wish: random
    if MAHJONG_INDEX in cards:
        if rng.random() < 0.5:
            wish = rng.randint(2, 14)
            gs.set_mahjong_wish(wish)
            print(f"    {CYAN}-> P{player} wishes for rank {wish}{_RESET}")

    # Check if player went out
    if player in gs.finish_order and gs.finish_order[-1] == player:
        pos = len(gs.finish_order)
        print(f"    {GREEN}* P{player} goes out (#{pos})!{_RESET}")


def main():
    seed = None
    human = 0

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--seat" and i + 1 < len(args):
            human = int(args[i + 1])
        elif arg.isdigit() and (i == 0 or args[i - 1] != "--seat"):
            seed = int(arg)

    if seed is None:
        seed = random.randint(0, 99999)

    rng = random.Random(seed)
    gs = GameState()
    gs.reset(seed=seed)

    print(f"\n{'#' * 60}")
    print(f"  {_BOLD}TICHU{_RESET} — You are P{human} (team {team_label(human)})")
    print(f"  Partner: P{(human + 2) % 4}  |  Opponents: P{(human + 1) % 4}, P{(human + 3) % 4}")
    print(f"  Seed: {seed}")
    print(f"{'#' * 60}")

    while not gs.round_over:
        # Dragon give
        if gs.dragon_give_pending is not None:
            player = gs.dragon_give_pending
            if player == human:
                print_table(gs, human)
                recipient = choose_dragon_recipient(gs, human)
            else:
                options = gs.get_dragon_give_options(player)
                recipient = rng.choice(options)
                print(f"  P{player} gives Dragon trick to P{recipient}")
            gs.give_dragon_trick(player, recipient)
            continue

        player = gs.current_player

        if player == human:
            print_table(gs, human)
            actions = gs.get_legal_actions(human)

            if not actions:
                continue

            choice = choose_action(actions, gs, human)
            action = actions[choice]
            cards, call_tichu = action

            gs.play(human, action)

            tichu_tag = f" {RED}+ TICHU!{_RESET}" if call_tichu else ""
            if cards:
                combo = classify_cards(cards)
                type_name = COMBO_NAMES.get(combo.type, "") if combo else ""
                print(
                    f"\n  {GREEN}You played: {fmt_cards(cards)} "
                    f"({type_name}){_RESET}{tichu_tag}"
                )
            else:
                print(f"\n  {DIM}You passed{_RESET}{tichu_tag}")

            # Mahjong wish
            if MAHJONG_INDEX in cards:
                wish = choose_wish()
                gs.set_mahjong_wish(wish)
                if wish:
                    print(f"  {CYAN}Wish set: rank {wish}{_RESET}")

            # Check if went out
            if human in gs.finish_order and gs.finish_order[-1] == human:
                pos = len(gs.finish_order)
                print(f"\n  {GREEN}{_BOLD}* You went out (#{pos})!{_RESET}")

        else:
            ai_play(gs, rng)

            # AI dragon give
            if gs.dragon_give_pending is not None and gs.dragon_give_pending != human:
                player = gs.dragon_give_pending
                options = gs.get_dragon_give_options(player)
                recipient = rng.choice(options)
                gs.give_dragon_trick(player, recipient)
                print(f"  P{player} gives Dragon trick to P{recipient}")

    # Game over
    print()
    print_divider()
    print(f"  {_BOLD}ROUND OVER{_RESET}")
    print(f"  Finish order: {' -> '.join(f'P{p}' for p in gs.finish_order)}")
    a, b = gs.round_scores
    print(f"  Team A (P0+P2): {a:+d}")
    print(f"  Team B (P1+P3): {b:+d}")

    your_team = 0 if human in TEAM_A else 1
    your_score = gs.round_scores[your_team]
    opp_score = gs.round_scores[1 - your_team]
    if your_score > opp_score:
        print(f"\n  {GREEN}{_BOLD}You win!{_RESET}")
    elif your_score < opp_score:
        print(f"\n  {RED}You lose.{_RESET}")
    else:
        print(f"\n  {YELLOW}Draw.{_RESET}")
    print_divider()


if __name__ == "__main__":
    main()
