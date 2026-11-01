"""Human-readable game display and play-by-play logging with colored suits."""

from __future__ import annotations

from tichu_env.cards import (
    DRAGON_INDEX,
    DOG_INDEX,
    MAHJONG_INDEX,
    PHOENIX_INDEX,
    Suit,
    card_name,
    card_rank,
    card_suit,
)
from tichu_env.combinations import CombinationType, classify_cards


# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_SUIT_COLORS = {
    Suit.JADE: "\033[32m",     # green
    Suit.PAGODA: "\033[34m",   # blue
    Suit.STAR: "\033[31m",     # red
    Suit.SWORD: "\033[33m",    # yellow/orange (black invisible on dark terminals)
}
_SPECIAL_COLORS = {
    MAHJONG_INDEX: "\033[36m",           # cyan
    DOG_INDEX: "\033[90m",               # gray
    PHOENIX_INDEX: "\033[35m",           # magenta
    DRAGON_INDEX: "\033[31m\033[1m",     # bold red
}

COMBO_NAMES = {
    CombinationType.PASS: "Pass",
    CombinationType.DOG: "Dog",
    CombinationType.SINGLE: "Single",
    CombinationType.PAIR: "Pair",
    CombinationType.TRIPLE: "Triple",
    CombinationType.STRAIGHT: "Straight",
    CombinationType.FULL_HOUSE: "Full House",
    CombinationType.STEPS: "Steps",
    CombinationType.FOUR_BOMB: "BOMB (4)",
    CombinationType.STRAIGHT_BOMB: "BOMB (SF)",
}


def color_card(index: int) -> str:
    """Return a colored card name string."""
    name = card_name(index)
    if index in _SPECIAL_COLORS:
        return f"{_SPECIAL_COLORS[index]}{name}{_RESET}"
    suit = card_suit(index)
    if suit is not None:
        return f"{_SUIT_COLORS[suit]}{name}{_RESET}"
    return name


def _sort_key(c: int) -> tuple[int, int]:
    """Sort by rank first, then suit. Specials go: Mahjong first, Dog/Phoenix/Dragon last."""
    if c == MAHJONG_INDEX:
        return (0, 0)
    if c == DOG_INDEX:
        return (99, 0)
    if c == PHOENIX_INDEX:
        return (99, 1)
    if c == DRAGON_INDEX:
        return (99, 2)
    r = card_rank(c)
    s = card_suit(c)
    return (r.value, s.value)


def fmt_cards(cards: list[int] | set[int]) -> str:
    """Format a collection of cards as a colored string, sorted by rank."""
    if not cards:
        return "---"
    return " ".join(color_card(c) for c in sorted(cards, key=_sort_key))


def fmt_hand(cards: set[int]) -> str:
    """Format a hand with colored cards."""
    if not cards:
        return "[empty]"
    return f"[{fmt_cards(cards)}] ({len(cards)})"


def fmt_play(player: int, cards: list[int]) -> str:
    """Describe a single play action."""
    if not cards:
        return f"Player {player}: Pass"

    if cards == [DOG_INDEX]:
        return f"Player {player}: Dog → lead passes to partner"

    combo = classify_cards(cards)
    type_name = COMBO_NAMES.get(combo.type, "???") if combo else "???"
    card_str = fmt_cards(cards)

    if combo and combo.is_bomb:
        return f"Player {player}: ** {type_name} ** {card_str}"
    return f"Player {player}: {type_name} {card_str}"


def print_state(game_state) -> None:
    """Print the full current game state."""
    gs = game_state
    print("=" * 60)
    for i in range(4):
        tag = ""
        if i == gs.current_player and i not in gs.finish_order:
            tag = " ◄"
        if i in gs.finish_order:
            pos = gs.finish_order.index(i) + 1
            tag = f" (out #{pos})"
        tichu = ""
        if gs.tichu_declarations[i] == 1:
            tichu = " [T!]"
        elif gs.tichu_declarations[i] == 2:
            tichu = " [GT!]"
        team = "A" if i in (0, 2) else "B"
        print(f"  P{i} (team {team}){tichu}{tag}: {fmt_hand(gs.hands[i])}")

    if gs.trick.cards_on_table:
        combo = gs.trick.leading_combo
        type_name = COMBO_NAMES.get(combo.type, "") if combo else ""
        print(f"  Table: {fmt_cards(gs.trick.cards_on_table)} ({type_name}, by P{gs.trick.leading_player})")
    else:
        print("  Table: empty (new trick)")

    if gs.mahjong_wish:
        from tichu_env.cards import Rank
        rank_names = {v: k for k, v in Rank.__members__.items()}
        wish_name = rank_names.get(gs.mahjong_wish, str(gs.mahjong_wish))
        print(f"  Wish: {wish_name}")
    if gs.dragon_give_pending is not None:
        print(f"  ** Dragon give pending (P{gs.dragon_give_pending} must choose) **")
    print("=" * 60)


def print_scores(game_state) -> None:
    """Print end-of-round scoring summary."""
    gs = game_state
    print()
    print("─" * 40)
    print("ROUND OVER")
    print(f"  Finish order: {' → '.join(f'P{p}' for p in gs.finish_order)}")
    a, b = gs.round_scores
    print(f"  Team A (P0+P2): {a:+d}")
    print(f"  Team B (P1+P3): {b:+d}")
    for i in range(4):
        d = gs.tichu_declarations[i]
        if d == 1:
            ok = gs.finish_order[0] == i
            print(f"  P{i} Tichu: {'success +100' if ok else 'FAILED -100'}")
        elif d == 2:
            ok = gs.finish_order[0] == i
            print(f"  P{i} Grand Tichu: {'success +200' if ok else 'FAILED -200'}")
    print("─" * 40)


def play_random_game(seed: int = 0, show_hands: bool = True, pause: bool = False) -> None:
    """Play a full random game with printed play-by-play."""
    import random
    from tichu_env.game_state import GameState

    rng = random.Random(seed)
    gs = GameState()
    gs.reset(seed=seed)

    print(f"\n{'#' * 60}")
    print(f"  TICHU — Random game (seed={seed})")
    print(f"  Teams: A = P0+P2, B = P1+P3")
    print(f"{'#' * 60}\n")

    if show_hands:
        print("Dealt hands:")
        for i in range(4):
            print(f"  P{i}: {fmt_hand(gs.hands[i])}")
        print()

    tricks_seen = len(gs.trick_history)
    print(f"\n--- Trick 1 (P{gs.current_player} leads) ---")

    steps = 0
    while not gs.round_over and steps < 500:
        steps += 1

        if gs.dragon_give_pending is not None:
            player = gs.dragon_give_pending
            opts = gs.get_dragon_give_options(player)
            recipient = rng.choice(opts)
            gs.give_dragon_trick(player, recipient)
            print(f"  Player {player}: gives Dragon trick to P{recipient}")
            if not gs.round_over and len(gs.trick_history) > tricks_seen:
                tricks_seen = len(gs.trick_history)
                print(f"\n--- Trick {tricks_seen + 1} (P{gs.current_player} leads) ---")
            continue

        player = gs.current_player
        actions = gs.get_legal_actions(player)
        action = rng.choice(actions)
        cards, call_tichu = action

        old_finish = len(gs.finish_order)

        gs.play(player, action)

        # Print the play
        tichu_tag = " + TICHU!" if call_tichu else ""
        print(f"  {fmt_play(player, cards)}{tichu_tag}")

        # Mahjong wish (random)
        if MAHJONG_INDEX in cards:
            if rng.random() < 0.5:
                wish = rng.randint(2, 14)
                gs.set_mahjong_wish(wish)
                print(f"    -> wishes for rank {wish}")

        # Player went out
        if len(gs.finish_order) > old_finish:
            who = gs.finish_order[-1]
            pos = len(gs.finish_order)
            print(f"    * P{who} goes out (#{pos})!")

        # New trick started
        if not gs.round_over and len(gs.trick_history) > tricks_seen:
            tricks_seen = len(gs.trick_history)
            print(f"\n--- Trick {tricks_seen + 1} (P{gs.current_player} leads) ---")

        if pause and not gs.round_over:
            input()

    print_scores(gs)
