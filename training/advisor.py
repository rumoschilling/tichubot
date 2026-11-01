"""Real-life Tichu advisor: input your hand and table plays, get AI suggestions.

You play a real game of Tichu and input what happens. The network
tells you what to play on your turn.

Usage:
    python -m training.advisor
    python -m training.advisor --checkpoint kaggle_runs/2026-08-01_110222_standard/tichubot_outputs/checkpoints/history_standard_gpu/latest.pt --seat 0
"""

from __future__ import annotations

import argparse
import copy
import random
import sys

import numpy as np
import torch

from tichu_env.cards import (
    DRAGON_INDEX,
    DOG_INDEX,
    MAHJONG_INDEX,
    NUM_CARDS,
    PHOENIX_INDEX,
    Rank,
    Suit,
    card_index,
    card_name,
)
from tichu_env.combinations import (
    Combination,
    CombinationType,
    binary_phoenix_interpretations,
    classify_cards,
)
from tichu_env.display import (
    COMBO_NAMES,
    _RESET,
    _BOLD,
    color_card,
    fmt_cards,
    fmt_hand,
)
from tichu_env.encoding import (
    encode_history,
    encode_pass_source_ids,
    encode_pass_target_ids,
    encode_static,
    PASS_OBS_SIZE,
    PLAY_TOKEN_SIZE,
)
from tichu_env.game_state import GamePhase, GameState, TEAM_A
from training.checkpoints import DEFAULT_PLAY_CHECKPOINT
from training.model_embed import ACTION_DIM
from training.observations import encode_policy_static
from training.play import load_play_bundle
from training.rollout_embed import _action_vec

MAX_ACTIONS = 2000

DEFAULT_CHECKPOINT = DEFAULT_PLAY_CHECKPOINT
DIM    = "\033[2m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
RED    = "\033[31m"
UNDO = "__undo__"


# ── Card parsing ─────────────────────────────────────────────────────

_SUIT_MAP = {"g": Suit.JADE, "b": Suit.PAGODA, "r": Suit.STAR, "s": Suit.SWORD}
_RANK_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "j": 11, "q": 12, "k": 13, "a": 14,
}
_SPECIAL_MAP = {
    "mahjong": MAHJONG_INDEX, "m": MAHJONG_INDEX, "1": MAHJONG_INDEX,
    "dog": DOG_INDEX, "w": DOG_INDEX,
    "phoenix": PHOENIX_INDEX, "ph": PHOENIX_INDEX, "p": PHOENIX_INDEX,
    "dragon": DRAGON_INDEX, "dr": DRAGON_INDEX, "d": DRAGON_INDEX,
}


def parse_card(s: str) -> int | None:
    """Parse a card string (e.g. '3g', 'Jb', 'Mahjong') into a card index.

    Supports shorthand aliases for special cards (m=Mahjong, d=Dragon,
    w=Dog, p=Phoenix)
    and standard rank+suit notation (r=Star, s=Sword, g=Jade, b=Pagoda).
    """
    s = s.strip().lower()
    if s in _SPECIAL_MAP:
        return _SPECIAL_MAP[s]
    # Normal card: rank + suit letter (last char)
    if len(s) < 2:
        return None
    suit_ch = s[-1]
    rank_str = s[:-1]
    if suit_ch not in _SUIT_MAP or rank_str not in _RANK_MAP:
        return None
    return card_index(_SUIT_MAP[suit_ch], Rank(_RANK_MAP[rank_str]))


def _expand_rank_token(token: str) -> list[int] | None:
    """Expand one bare rank or inclusive rank range."""
    token = token.strip().lower()
    if token in _RANK_MAP:
        return [_RANK_MAP[token]]
    if token.count("-") != 1:
        return None
    start_s, end_s = token.split("-", 1)
    if start_s not in _RANK_MAP or end_s not in _RANK_MAP:
        return None
    start = _RANK_MAP[start_s]
    end = _RANK_MAP[end_s]
    if end < start:
        return None
    return list(range(start, end + 1))


def _expand_suit_token(token: str) -> list[Suit] | None:
    """Expand ``r``, ``rgb`` and similar compact suit sequences."""
    token = token.strip().lower()
    if not token or any(char not in _SUIT_MAP for char in token):
        return None
    return [_SUIT_MAP[char] for char in token]


def parse_cards(text: str) -> list[int] | None:
    """Parse compact card input, returning ``None`` on an invalid sequence.

    Besides ordinary tokens such as ``2r`` and ``Phoenix``, consecutive bare
    ranks can be paired positionally with consecutive suits:

    - ``2 2 3 3 r g s b`` -> ``2r 2g 3s 3b``
    - ``2-7 r g b s g s`` -> ``2r 3g 4b 5s 6g 7s``
    - ``2 rgb`` -> ``2r 2g 2b``
    - ``J bomb`` -> all four Jacks
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.replace(",", " ").split()
    cards: list[int] = []
    i = 0
    while i < len(tokens):
        card = parse_card(tokens[i])
        if card is not None:
            cards.append(card)
            i += 1
            continue

        ranks: list[int] = []
        while i < len(tokens):
            expanded = _expand_rank_token(tokens[i])
            if expanded is None:
                break
            ranks.extend(expanded)
            i += 1
        if not ranks:
            return None

        if i < len(tokens) and tokens[i].lower() == "bomb":
            if len(ranks) != 1:
                return None
            cards.extend(
                card_index(suit, Rank(ranks[0]))
                for suit in (Suit.STAR, Suit.JADE, Suit.PAGODA, Suit.SWORD)
            )
            i += 1
            continue

        suits: list[Suit] = []
        while i < len(tokens):
            expanded_suits = _expand_suit_token(tokens[i])
            if expanded_suits is None:
                break
            suits.extend(expanded_suits)
            i += 1
        if len(ranks) == 1 and suits:
            ranks *= len(suits)
        elif len(suits) == 1 and ranks:
            suits *= len(ranks)
        elif len(suits) != len(ranks):
            return None
        cards.extend(card_index(suit, Rank(rank)) for rank, suit in zip(ranks, suits))
    return cards


def input_cards(
    prompt: str, *, allow_empty: bool = False, allow_undo: bool = False
) -> list[int] | str:
    """Prompt user for cards until valid input is given."""
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)
        if allow_undo and raw.lower() in {"u", "undo"}:
            return UNDO
        if not raw and allow_empty:
            return []
        cards = parse_cards(raw)
        if cards is None or (not cards and not allow_empty):
            print(f"  {RED}Could not parse. Use: 3g Jb 10r As Phoenix Dragon Dog Mahjong{_RESET}")
            continue
        return cards


# ── Game state helpers ───────────────────────────────────────────────

def setup_game(hand_cards: list[int], seat: int, track_history: bool = False) -> GameState:
    """Create a GameState with a known hand for the human player.

    Remaining 42 cards are distributed randomly to opponents.
    """
    gs = GameState(track_history=track_history)
    # Real-table game: every seat is a human/unknown player, so no seat
    # shares weights with any other (partner same-weights bit = 0).
    gs.player_model_ids = [0, 1, 2, 3]
    # Minimal reset without dealing
    gs.finish_order = []
    gs.finished = set()
    gs.won_cards = [[] for _ in range(4)]
    from tichu_env.game_state import TrickState
    gs.trick = TrickState()
    gs.tichu_declarations = [0, 0, 0, 0]
    gs.mahjong_wish = None
    gs.dragon_give_pending = None
    gs.played_cards = set()
    gs.trick_history = []
    gs.play_log = []
    gs.trick_num = 0
    gs._pass_pending = {}
    gs._incoming_pass_source = [dict() for _ in range(4)]
    gs._outgoing_pass_target = [dict() for _ in range(4)]
    gs._with_passing = False
    gs.round_over = False
    gs.round_scores = (0, 0)
    gs._has_played = set()
    gs.phase = GamePhase.PLAY

    hand_set = set(hand_cards)
    remaining = [c for c in range(NUM_CARDS) if c not in hand_set]
    random.shuffle(remaining)

    gs.hands = [set() for _ in range(4)]
    gs.hands[seat] = hand_set
    others = [p for p in range(4) if p != seat]
    for i, p in enumerate(others):
        gs.hands[p] = set(remaining[i * 14 : (i + 1) * 14])

    # Mahjong holder leads
    for i in range(4):
        if MAHJONG_INDEX in gs.hands[i]:
            gs.current_player = i
            break

    return gs


def force_play(
    gs: GameState,
    player: int,
    cards: list[int],
    seat: int,
    call_tichu: bool = False,
    *,
    phoenix_as: int | None = None,
) -> bool:
    """Force an opponent's play into the game state transactionally.

    If the opponent doesn't "have" a card in our simulated state, reconcile it
    from another hidden opponent hand. A rejected move leaves both ``gs`` and
    the global random stream exactly as they were before the attempt.
    """
    # Real cards in your hand are known ground truth and cannot be played by opponents.
    for c in cards:
        if c in gs.hands[seat]:
            return False

    # Cards passed by the human have a known recipient. Hidden-hand
    # reconciliation must not contradict that observed table fact.
    known_outgoing = {
        card: target
        for target, card in gs._outgoing_pass_target[seat].items()
    }
    if any(
        card in known_outgoing and known_outgoing[card] != player
        for card in cards
    ):
        return False

    trial = copy.deepcopy(gs)
    random_state = random.getstate()
    protected = {
        card
        for card in trial._incoming_pass_source[player].values()
        if card in trial.hands[player]
    }
    for c in cards:
        if c in trial.hands[player]:
            continue
        # Find it among other opponents (not the human).
        for other in range(4):
            if other == seat or other == player:
                continue
            if c in trial.hands[other]:
                # Give the simulated owner an unknown replacement from player.
                swappable = [
                    x
                    for x in trial.hands[player]
                    if x not in cards and x not in protected
                ]
                if not swappable:
                    random.setstate(random_state)
                    return False
                swap_card = random.choice(swappable)
                trial.hands[other].remove(c)
                trial.hands[other].add(swap_card)
                trial.hands[player].remove(swap_card)
                trial.hands[player].add(c)
                break

    # Reconciliation can affect every hidden hand, so discard all cached sorts.
    trial._invalidate_cache()
    # The wish obligation depends on the opponent's real hand, which the
    # advisor only guesses. A real move that leaves the wish open shows they
    # could not fulfil it, so the simulated hand must not veto that move.
    wish = trial.mahjong_wish
    suspend_wish = wish is not None and not any(
        c < 52 and c % 13 + 2 == wish for c in cards
    )
    if suspend_wish:
        trial.mahjong_wish = None
    try:
        accepted = trial.play(
            player,
            (cards, call_tichu),
            phoenix_as=phoenix_as,
        )
    except Exception:
        random.setstate(random_state)
        raise
    if not accepted:
        random.setstate(random_state)
        return False
    if suspend_wish:
        trial.mahjong_wish = wish

    # Keep the caller-visible GameState identity while committing every field
    # from the validated trial, including history and passing provenance.
    gs.__dict__.clear()
    gs.__dict__.update(trial.__dict__)
    return True


def known_mahjong_holder_from_passing(gs: GameState, seat: int) -> int | None:
    """Return the known holder when ``seat`` passed the Mahjong away."""
    return next(
        (
            target
            for target, card in gs._outgoing_pass_target[seat].items()
            if card == MAHJONG_INDEX
        ),
        None,
    )


def legal_phoenix_meanings(
    gs: GameState,
    player: int,
    seat: int,
    cards: list[int],
    call_tichu: bool = False,
) -> list[Combination]:
    """Return the legal meanings of an ambiguous Phoenix straight/full house.

    Unambiguous card sets return an empty list. Each meaning is tried on a
    copy of ``gs`` (through ``force_play`` for opponents), so neither the
    state nor the global random stream changes.
    """
    meanings = binary_phoenix_interpretations(cards)
    if meanings is None:
        return []
    legal = []
    random_state = random.getstate()
    try:
        for combo in meanings:
            trial = copy.deepcopy(gs)
            if player == seat:
                accepted = trial.play(
                    player, (list(cards), call_tichu), phoenix_as=combo.phoenix_as
                )
            else:
                accepted = force_play(
                    trial, player, list(cards), seat, call_tichu,
                    phoenix_as=combo.phoenix_as,
                )
            if accepted:
                legal.append(combo)
    finally:
        random.setstate(random_state)
    return legal


def _rank_label(rank: int) -> str:
    return {11: "J", 12: "Q", 13: "K", 14: "A"}.get(rank, str(rank))


def ask_phoenix_meaning(meanings: list[Combination]) -> Combination | None:
    """Ask which of two legal Phoenix meanings was played; None cancels."""
    low, high = meanings

    def describe(combo: Combination) -> str:
        what = "straight to" if combo.type == CombinationType.STRAIGHT else "triple"
        return f"{what} {_rank_label(int(combo.rank))}, Phoenix as {_rank_label(combo.phoenix_as)}"

    print(f"  {CYAN}Phoenix meaning? [l]ow: {describe(low)} | [h]igh: {describe(high)}{_RESET}")
    while True:
        try:
            raw = input(f"  {_BOLD}Phoenix (l/h, u=cancel): {_RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)
        if raw in {"l", "low"}:
            return low
        if raw in {"h", "high"}:
            return high
        if raw in {"u", "undo"}:
            return None
        print(f"  {RED}Enter l or h{_RESET}")


def parse_wish(raw: str) -> int | None | str:
    """Parse a wish answer: rank 2-14 or J/Q/K/A, None for no wish.

    Returns UNDO for an undo request and "" for unparseable input.
    """
    raw = raw.strip().lower()
    if raw in {"u", "undo"}:
        return UNDO
    if raw in {"", "0", "n", "no", "none"}:
        return None
    if raw in _RANK_MAP:
        return _RANK_MAP[raw]
    if raw.isdigit() and 2 <= int(raw) <= 14:
        return int(raw)
    return ""


def input_wish(prompt: str) -> int | None | str:
    """Prompt until a valid wish (or undo) is entered."""
    while True:
        try:
            raw = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)
        wish = parse_wish(raw)
        if wish != "":
            return wish
        print(f"  {RED}Enter a rank 2-10, J, Q, K, A, or 0 for no wish{_RESET}")


def choose_pass_cards(hand: list[int]) -> tuple[int, int, int] | str:
    labels = ("left", "partner", "right")
    chosen: list[int] = []

    print(f"\n  {CYAN}{_BOLD}Passing Phase{_RESET}")
    print("  Choose one card to pass to each recipient.")

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

    return tuple(hand[i] for i in chosen)


def suggest_pass(pass_policy, gs: GameState, seat: int,
                 device: torch.device) -> tuple[int, int, int] | None:
    if pass_policy is None:
        return None
    hand = sorted(gs.hands[seat])
    pass_obs = np.zeros((1, PASS_OBS_SIZE), dtype=np.float32)
    for c in hand:
        pass_obs[0, c] = 1.0
    for i in range(4):
        if gs.tichu_declarations[(seat + i) % 4] == 2:
            pass_obs[0, NUM_CARDS + i] = 1.0
    hand_cards = np.array(hand, dtype=np.int64)[None, :]
    with torch.no_grad():
        choices_t, _ = pass_policy.act(
            torch.as_tensor(pass_obs, device=device),
            torch.as_tensor(hand_cards, device=device),
        )
    idxs = choices_t[0].cpu().numpy()
    return tuple(hand[int(i)] for i in idxs)


def apply_manual_passing(
    gs: GameState,
    seat: int,
    outgoing: tuple[int, int, int],
    incoming: tuple[int, int, int],
) -> None:
    """Apply the real table's directed exchange to an advisor state.

    Both tuples use left, partner, right order from ``seat``. Opponent hands
    remain approximate, but their sizes are reconciled and every known pass
    edge is retained for the policy's persistent provenance embeddings.
    """
    if len(set(outgoing)) != 3 or not set(outgoing).issubset(gs.hands[seat]):
        raise ValueError("Outgoing cards must be three distinct cards in your hand.")
    if len(set(incoming)) != 3:
        raise ValueError("Received cards must be three distinct cards.")
    if set(incoming) & gs.hands[seat]:
        raise ValueError("A received card was already in your original hand.")

    related = (
        (seat + 1) % 4,
        (seat + 2) % 4,
        (seat + 3) % 4,
    )
    outgoing_set = set(outgoing)
    incoming_set = set(incoming)

    gs.hands[seat].difference_update(outgoing_set)
    for target, card in zip(related, outgoing):
        gs.hands[target].add(card)

    # The advisor's initial opponent deal is random. Move each known incoming
    # card to its actual source first, swapping an unknown card back to the
    # simulated owner so all opponent hand counts remain at 14.
    for source, card in zip(related, incoming):
        if card not in gs.hands[source]:
            owner = next(
                (
                    player
                    for player in range(4)
                    if player != seat and card in gs.hands[player]
                ),
                None,
            )
            if owner is None:
                raise ValueError("A received card is missing from the simulated deal.")
            swappable = [
                candidate
                for candidate in gs.hands[source]
                if candidate not in outgoing_set and candidate not in incoming_set
            ]
            if not swappable:
                raise ValueError("Could not reconcile the received cards with opponent hands.")
            swap = random.choice(swappable)
            gs.hands[source].remove(swap)
            gs.hands[source].add(card)
            gs.hands[owner].remove(card)
            gs.hands[owner].add(swap)
        gs.hands[source].remove(card)
        gs.hands[seat].add(card)

    gs._incoming_pass_source = [dict() for _ in range(4)]
    gs._outgoing_pass_target = [dict() for _ in range(4)]
    for target, card in zip(related, outgoing):
        gs._outgoing_pass_target[seat][target] = card
        gs._incoming_pass_source[target][seat] = card
    for source, card in zip(related, incoming):
        gs._incoming_pass_source[seat][source] = card
        gs._outgoing_pass_target[source][seat] = card
    gs._pass_pending = {}
    gs._invalidate_cache()
    gs.phase = GamePhase.PLAY
    gs._start_play()


def run_passing_phase(
    gs: GameState,
    seat: int,
    device: torch.device,
    pass_policy=None,
):
    """Apply a manual passing phase for advisor mode.

    We keep your hand exact and update opponent hands approximately.
    """
    hand = sorted(gs.hands[seat])
    suggestion = suggest_pass(pass_policy, gs, seat, device)
    if suggestion is not None:
        print(
            f"\n  {_BOLD}AI passing suggestion:{_RESET} "
            f"left {fmt_cards([suggestion[0]])}, "
            f"partner {fmt_cards([suggestion[1]])}, "
            f"right {fmt_cards([suggestion[2]])}"
        )
    res = choose_pass_cards(hand)
    if res == UNDO:
        return UNDO
    left, partner, right = res
    outgoing = {left, partner, right}

    print("\n  Enter cards received from left, partner, then right.")
    while True:
        incoming = input_cards(f"  {_BOLD}Received (L P R): {_RESET}")
        if len(incoming) != 3:
            print(f"  {RED}Need exactly 3 cards{_RESET}")
            continue
        if len(set(incoming)) != 3:
            print(f"  {RED}Duplicate cards{_RESET}")
            continue
        if any(c in gs.hands[seat] for c in incoming):
            print(f"  {RED}Received cards overlap with your original hand{_RESET}")
            continue
        break

    apply_manual_passing(
        gs,
        seat,
        (left, partner, right),
        tuple(incoming),
    )

    print(f"\n  {_BOLD}Your hand after passing:{_RESET} {fmt_hand(gs.hands[seat])}")


# ── Display ──────────────────────────────────────────────────────────

def print_status(gs: GameState, seat: int):
    """Print current game state from the advisor's perspective."""
    print()
    print(f"{'─' * 60}")
    for offset in [1, 2, 3]:
        p = (seat + offset) % 4
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
        print(f"  P{p} {rel}{tichu}: {n} cards{status}")

    print()
    if gs.trick.cards_on_table:
        combo = gs.trick.leading_combo
        type_name = COMBO_NAMES.get(combo.type, "") if combo else ""
        print(f"  {YELLOW}Table:{_RESET} {fmt_cards(gs.trick.cards_on_table)} ({type_name}, by P{gs.trick.leading_player})")
    else:
        print(f"  {YELLOW}Table:{_RESET} empty — new trick")

    if gs.mahjong_wish:
        rank_names = {r.value: r.name.title() for r in Rank}
        print(f"  {CYAN}Wish: {rank_names.get(gs.mahjong_wish, gs.mahjong_wish)}{_RESET}")

    print()
    tichu = ""
    if gs.tichu_declarations[seat] == 1:
        tichu = f" {RED}[T!]{_RESET}"
    print(f"  {_BOLD}Your hand{tichu}:{_RESET} {fmt_hand(gs.hands[seat])}")
    print(f"{'─' * 60}")


# ── AI suggestion ────────────────────────────────────────────────────

def suggest_gt(gt_policy, hand_8: list[int], device: torch.device) -> float | None:
    """Return the model's Grand Tichu probability for the first 8 cards."""
    if gt_policy is None:
        return None
    h8_bin = np.zeros((1, NUM_CARDS), dtype=np.float32)
    h8_bin[0, hand_8] = 1.0
    with torch.no_grad():
        log_p, _ = gt_policy.evaluate(
            torch.as_tensor(h8_bin, device=device),
            torch.ones(1, device=device),
        )
    return float(log_p.exp().item())


def _fused_seat_features(policy, gs: GameState, player: int, device: torch.device):
    obs = encode_static(gs, player)
    ps = encode_pass_source_ids(gs, player)
    pt = encode_pass_target_ids(gs, player)
    hist, hist_len = encode_history(gs, player)
    max_h = max(int(hist_len), 1)
    return policy.fused_features(
        torch.as_tensor(obs[None, :], dtype=torch.float32, device=device),
        torch.as_tensor(hist[None, :max_h], dtype=torch.float32, device=device),
        torch.as_tensor(np.array([hist_len], dtype=np.int32), device=device),
        torch.as_tensor(ps[None, :], dtype=torch.int64, device=device),
        torch.as_tensor(pt[None, :], dtype=torch.int64, device=device),
    )


@torch.no_grad()
def suggest_dragon(policy, kind: str, gs: GameState, giver: int,
                   device: torch.device) -> tuple[int, float] | None:
    """Dragon-head advice: (absolute target player, confidence) or None."""
    if kind != "history" or not hasattr(policy, "dragon_head"):
        return None
    logits = policy.get_dragon_gift_logits(
        _fused_seat_features(policy, gs, giver, device))[0]
    probs = torch.softmax(logits, dim=-1)
    choice = int(probs.argmax())
    target_abs = (giver + (1 if choice == 0 else 3)) % 4
    return target_abs, float(probs[choice])


@torch.no_grad()
def suggest_wish(policy, kind: str, gs: GameState, player: int,
                 device: torch.device, top_n: int = 3) -> list[tuple[str, float]] | None:
    """Wish-head advice: top (rank label, prob) pairs, or None."""
    if kind != "history" or not hasattr(policy, "wish_head"):
        return None
    logits = policy.get_mahjong_wish_logits(
        _fused_seat_features(policy, gs, player, device))[0]
    probs = torch.softmax(logits, dim=-1)
    rank_names = {11: "J", 12: "Q", 13: "K", 14: "A"}
    out = []
    for idx in probs.argsort(descending=True)[:top_n]:
        i = int(idx)
        label = "no wish" if i == 0 else rank_names.get(i + 1, str(i + 1))
        out.append((label, float(probs[i])))
    return out


@torch.no_grad()
def suggest(policy, gs: GameState, player: int, device: torch.device,
            top_n: int = 5, kind: str = "embed"):
    """Return top-N suggested actions with probabilities."""
    legal = (
        gs.get_legal_actions(player)
        if kind == "fixed"
        else gs.get_policy_legal_actions(player)
    )
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

    if kind in ("embed", "history"):
        import numpy as np
        n = len(legal)
        pad = np.zeros((1, n, ACTION_DIM), dtype=np.float32)
        for k, action in enumerate(legal):
            pad[0, k] = _action_vec(action)
        action_t = torch.as_tensor(pad, device=device)
        if kind == "history":
            hist, hist_len = encode_history(gs, player)
            max_h = max(int(hist_len), 1)
            hist_t = torch.as_tensor(
                hist[:max_h][np.newaxis], dtype=torch.float32, device=device
            )
            logits = policy._forward_scores(
                obs, action_t, hist_t,
                torch.tensor([hist_len], dtype=torch.int32, device=device),
                pass_src,
                pass_tgt,
            )[0]
        else:
            logits = policy._forward_scores(obs, action_t, pass_src, pass_tgt)[0]
        probs = torch.softmax(logits[0, :n], dim=0)
    else:
        n_legal = min(len(legal), MAX_ACTIONS)
        mask = (torch.arange(MAX_ACTIONS, device=device) < n_legal).unsqueeze(0)
        forward_result = policy.forward(obs, mask)
        logits = (
            forward_result[0]
            if isinstance(forward_result, tuple)
            else forward_result
        )
        probs = torch.softmax(logits[0, :n_legal], dim=0)

    # Top-N by probability
    values, indices = torch.topk(probs, min(top_n, len(probs)))
    results = []
    for prob, idx in zip(values.tolist(), indices.tolist()):
        action = legal[idx]
        cards, call_tichu = action[:2]
        phoenix_high = bool(action[2]) if len(action) == 3 else False
        results.append((cards, call_tichu, phoenix_high, prob))
    return results


def print_suggestions(suggestions):
    """Print the model's top suggestions."""
    print(f"\n  {_BOLD}AI suggests:{_RESET}")
    for i, (cards, call_tichu, phoenix_high, prob) in enumerate(suggestions):
        tichu_tag = f" {RED}+ TICHU{_RESET}" if call_tichu else ""
        if not cards:
            label = f"{DIM}Pass{_RESET}{tichu_tag}"
        else:
            combo = classify_cards(cards)
            type_name = COMBO_NAMES.get(combo.type, "") if combo else ""
            if phoenix_high:
                type_name = f"{type_name}, higher Phoenix"
            label = f"{fmt_cards(cards)} {DIM}({type_name}){_RESET}{tichu_tag}"
        bar_len = int(prob * 30)
        bar = f"{GREEN}{'█' * bar_len}{DIM}{'░' * (30 - bar_len)}{_RESET}"
        print(f"    {CYAN}{i}{_RESET}) {label}  {bar} {prob:.1%}")


# ── Main loop ────────────────────────────────────────────────────────

def run(policy, seat: int, device: torch.device, kind: str = "history", 
        pass_policy=None, gt_policy=None):
    """Main advisor loop: manages real-time game tracking and AI suggestions.

    Phases:
      1. GT Setup: User inputs 8 cards; AI suggests GT call.
      2. Full Hand: User inputs remaining 6 cards.
      3. Passing: AI suggests passes, user inputs choices and received cards.
      4. Play: Cyclic turn loop until the round is over.
    """

    print(f"\n{'#' * 60}")
    print(f"  {_BOLD}TICHU ADVISOR{_RESET} — You are P{seat}")
    print(f"  Partner: P{(seat+2)%4}  |  Opponents: P{(seat+1)%4}, P{(seat+3)%4}")
    print(f"{'#' * 60}")
    print(f"  {DIM}Type 'undo' at prompts to revert your last manual state change.{_RESET}")

    while True: # Setup loop for UNDO support
        # 1. First 8 cards (Grand Tichu Phase)
        print(f"\n  {_BOLD}Enter your first 8 cards (for Grand Tichu decision):{_RESET}")
        print(f"  {DIM}(e.g. 3g 5b 10r Jb Qs Kg As Phoenix){_RESET}")
        while True:
            cards_8 = input_cards(f"  {_BOLD}First 8: {_RESET}")
            if len(cards_8) != 8:
                print(f"  {RED}Need exactly 8 cards, got {len(cards_8)}{_RESET}")
                continue
            if len(set(cards_8)) != 8:
                print(f"  {RED}Duplicate cards{_RESET}")
                continue
            break
        
        gt_prob = suggest_gt(gt_policy, cards_8, device)
        if gt_prob is None:
            print(f"\n  {DIM}No GT model loaded — no suggestion.{_RESET}")
        else:
            sugg_text = (f"{RED}CALL GT!{_RESET}" if gt_prob > 0.5
                         else f"{DIM}Do not call GT.{_RESET}")
            print(f"\n  {_BOLD}AI GT Suggestion:{_RESET} {sugg_text} "
                  f"{DIM}(p={gt_prob:.1%}){_RESET}")

        # 2. GT Declaration and remaining cards
        print(f"\n  {_BOLD}GT Declarations?{_RESET} {DIM}Enter seats who called GT (e.g. '0 2', or enter for none):{_RESET}")
        gt_callers = []
        while True:
            raw = input(f"  {_BOLD}GT Callers: {_RESET}").strip()
            if not raw: break
            try:
                gt_callers = [int(x) for x in raw.split()]
                if all(0 <= c <= 3 for c in gt_callers): break
            except ValueError: pass
            print(f"  {RED}Enter player numbers 0-3 separated by spaces.{_RESET}")

        print(f"\n  {_BOLD}Enter your remaining 6 cards:{_RESET}")
        while True:
            cards_6 = input_cards(f"  {_BOLD}Next 6: {_RESET}")
            if len(cards_6) != 6:
                print(f"  {RED}Need exactly 6 cards, got {len(cards_6)}{_RESET}")
                continue
            full_hand = cards_8 + cards_6
            if len(set(full_hand)) != 14:
                print(f"  {RED}Duplicate cards found between the two sets!{_RESET}")
                continue
            break

        gs = setup_game(full_hand, seat, track_history=(kind == "history"))
        for c in gt_callers:
            gs.tichu_declarations[c] = 2
            
        undo_stack: list[GameState] = []
        if run_passing_phase(gs, seat, device, pass_policy) == UNDO:
            print(f"  {YELLOW}Undoing setup... restarting from 8-card phase.{_RESET}")
            continue # RESTART setup
        
        break # Setup successful

    if MAHJONG_INDEX in gs.hands[seat]:
        print(f"\n  {GREEN}You have the Mahjong — you lead.{_RESET}")
    else:
        holder = known_mahjong_holder_from_passing(gs, seat)
        if holder is not None:
            gs.current_player = holder
            print(
                f"\n  {CYAN}You passed the Mahjong to P{holder}; "
                f"P{holder} leads.{_RESET}"
            )
        else:
            print(f"\n  {CYAN}Who has the Mahjong? Enter player number (0-3):{_RESET}")
            while True:
                try:
                    raw = input(f"  {_BOLD}Mahjong holder: {_RESET}").strip()
                    holder = int(raw)
                    if 0 <= holder <= 3 and holder != seat:
                        gs.current_player = holder
                        # Ensure the holder "has" the Mahjong in our simulated state
                        for other in range(4):
                            if other != seat and MAHJONG_INDEX in gs.hands[other]:
                                if other != holder:
                                    # Swap a random card between holder and current owner
                                    swap = random.choice([c for c in gs.hands[holder] if c != MAHJONG_INDEX])
                                    gs.hands[other].remove(MAHJONG_INDEX)
                                    gs.hands[other].add(swap)
                                    gs.hands[holder].remove(swap)
                                    gs.hands[holder].add(MAHJONG_INDEX)
                                    gs._invalidate_cache()
                                break
                        break
                    print(f"  {RED}Enter an opponent's seat (0-3, not {seat}){_RESET}")
                except (ValueError, EOFError, KeyboardInterrupt):
                    print(f"  {RED}Enter a number 0-3{_RESET}")

    print(f"\n  {GREEN}Game started!{_RESET} P{gs.current_player} leads.")
    print(f"  Your hand: {fmt_hand(gs.hands[seat])}")

    while not gs.round_over:
        # Dragon give
        if gs.dragon_give_pending is not None:
            giver = gs.dragon_give_pending
            opts = gs.get_dragon_give_options(giver)
            if giver == seat:
                print(f"\n  {RED}Dragon give! Choose opponent: {opts}{_RESET}")
                ds = suggest_dragon(policy, kind, gs, giver, device)
                if ds is not None and ds[0] in opts:
                    print(f"  {_BOLD}AI suggests:{_RESET} give to P{ds[0]} "
                          f"{DIM}({ds[1]:.0%}){_RESET}")
                undone = False
                while True:
                    try:
                        raw = input(f"  {_BOLD}Give to: {_RESET}").strip()
                        if raw.lower() in {"u", "undo"}:
                            if undo_stack:
                                gs = undo_stack.pop()
                                print(f"  {YELLOW}Undid last action.{_RESET}")
                                undone = True
                                break
                            print(f"  {RED}Nothing to undo yet.{_RESET}")
                            continue
                        choice = int(raw)
                        if choice in opts:
                            break
                    except (ValueError, EOFError, KeyboardInterrupt):
                        pass
                    print(f"  {RED}Pick from {opts}{_RESET}")
                if undone:
                    continue
                undo_stack.append(copy.deepcopy(gs))
                gs.give_dragon_trick(giver, choice)
            else:
                print(f"\n  Dragon give pending for P{giver}. Who do they give to? {opts}")
                undone = False
                while True:
                    try:
                        raw = input(f"  {_BOLD}P{giver} gives to: {_RESET}").strip()
                        if raw.lower() in {"u", "undo"}:
                            if undo_stack:
                                gs = undo_stack.pop()
                                print(f"  {YELLOW}Undid last action.{_RESET}")
                                undone = True
                                break
                            print(f"  {RED}Nothing to undo yet.{_RESET}")
                            continue
                        choice = int(raw)
                        if choice in opts:
                            break
                    except (ValueError, EOFError, KeyboardInterrupt):
                        pass
                    print(f"  {RED}Pick from {opts}{_RESET}")
                if undone:
                    continue
                undo_stack.append(copy.deepcopy(gs))
                gs.give_dragon_trick(giver, choice)
            continue

        player = gs.current_player

        if player == seat:
            # Our turn — show suggestions
            print_status(gs, seat)
            suggestions = suggest(policy, gs, player, device, kind=kind)
            print_suggestions(suggestions)

            print(f"\n  {DIM}Enter cards to play (e.g. '3g 3b', empty=pass, '+t' to call Tichu, or pick suggestion #):{_RESET}")
            while True:
                try:
                    raw = input(f"  {_BOLD}Play: {_RESET}").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye!")
                    sys.exit(0)
                
                if raw.lower() in {"u", "undo"}:
                    if undo_stack:
                        gs = undo_stack.pop()
                        print(f"  {YELLOW}Undid last action.{_RESET}")
                    else:
                        print(f"  {RED}Nothing to undo yet.{_RESET}")
                    chosen_cards = None
                    break

                # Handle "tichu" command for others
                if raw.lower().startswith("tichu "):
                    try:
                        p_tichu = int(raw.split()[1])
                        if 0 <= p_tichu <= 3:
                            undo_stack.append(copy.deepcopy(gs))
                            gs.tichu_declarations[p_tichu] = 1
                            print(f"  {RED}P{p_tichu} called TICHU!{_RESET}")
                            chosen_cards = None # Refresh loop
                            break
                    except: pass

                # Check if it's a suggestion number
                call_tichu = False
                phoenix_high = False
                picked_suggestion = False
                if raw.isdigit() and int(raw) < len(suggestions):
                    chosen_cards, call_tichu, phoenix_high, _ = suggestions[int(raw)]
                    picked_suggestion = True
                    break

                # Parse manual cards + optional tichu call
                if raw.lower() in ("pass", "p", ""):
                    chosen_cards = []
                    break

                if raw.endswith("+t"):
                    call_tichu = True
                    raw = raw[:-2].strip()

                parsed = parse_cards(raw)
                if parsed is None:
                    print(f"  {RED}Could not parse. Use '+t' suffix for Tichu (e.g. '3g 3b +t'){_RESET}")
                    continue
                chosen_cards = parsed
                break

            if chosen_cards is None:
                continue

            # Typed Phoenix straights/full houses carry no explicit meaning;
            # suggestions already do (phoenix_high).
            phoenix_as = None
            if not picked_suggestion:
                meanings = legal_phoenix_meanings(gs, player, seat, chosen_cards, call_tichu)
                if len(meanings) == 2:
                    meaning = ask_phoenix_meaning(meanings)
                    if meaning is None:
                        continue
                    phoenix_as = meaning.phoenix_as
                elif len(meanings) == 1:
                    phoenix_as = meanings[0].phoenix_as

            undo_stack.append(copy.deepcopy(gs))
            action = (
                (chosen_cards, call_tichu)
                if kind == "fixed"
                else (chosen_cards, call_tichu, phoenix_high)
            )
            ok = gs.play(player, action, phoenix_as=phoenix_as)
            if not ok:
                undo_stack.pop()
                print(f"  {RED}Illegal play! Try again.{_RESET}")
                continue

            if chosen_cards:
                combo = classify_cards(chosen_cards)
                type_name = COMBO_NAMES.get(combo.type, "") if combo else ""
                tichu_tag = f" {RED}+ TICHU!{_RESET}" if call_tichu else ""
                print(f"  {GREEN}Played: {fmt_cards(chosen_cards)} ({type_name}){tichu_tag}{_RESET}")
            else:
                print(f"  {DIM}Passed{_RESET}")

            # Mahjong wish
            if MAHJONG_INDEX in chosen_cards and not gs.round_over:
                ws = suggest_wish(policy, kind, gs, player, device)
                if ws is not None:
                    pretty = ", ".join(f"{lbl} {DIM}{pr:.0%}{_RESET}" for lbl, pr in ws)
                    print(f"  {_BOLD}AI wish suggestion:{_RESET} {pretty}")
                print(f"  {CYAN}Wish? Enter rank 2-10, J, Q, K, A (0=none):{_RESET}")
                wish = input_wish(f"  {_BOLD}Wish: {_RESET}")
                if wish == UNDO:
                    gs = undo_stack.pop()
                    print(f"  {YELLOW}Undid last action.{_RESET}")
                    continue
                if wish is not None:
                    gs.set_mahjong_wish(wish)

        else:
            # Opponent's turn — user inputs what they played
            print(f"\n  {YELLOW}P{player}'s turn{_RESET}", end="")
            if gs.trick.leading_combo is None:
                print(" (leads)")
            else:
                print()
            print(f"  {DIM}Enter cards (empty=pass, 'tichu <p>' to record a call, '+t' if they called on this play):{_RESET}")

            while True:
                raw_input = input(f"  {_BOLD}P{player} plays: {_RESET}").strip()
                
                if raw_input.lower() in {"u", "undo"}:
                    if undo_stack:
                        gs = undo_stack.pop()
                        print(f"  {YELLOW}Undid last action.{_RESET}")
                    else:
                        print(f"  {RED}Nothing to undo yet.{_RESET}")
                    cards = None
                    break

                # Handle "tichu" command for any player
                if raw_input.lower().startswith("tichu "):
                    try:
                        p_tichu = int(raw_input.split()[1])
                        if 0 <= p_tichu <= 3:
                            undo_stack.append(copy.deepcopy(gs))
                            gs.tichu_declarations[p_tichu] = 1
                            print(f"  {RED}P{p_tichu} called TICHU!{_RESET}")
                            cards = None # Refresh loop
                            break
                    except: pass

                # Pass
                if raw_input.lower() in ("pass", "p", ""):
                    undo_stack.append(copy.deepcopy(gs))
                    ok = force_play(gs, player, [], seat)
                    if ok:
                        print(f"  P{player}: {DIM}Pass{_RESET}")
                        cards = []
                        break
                    undo_stack.pop()
                    print(f"  {RED}Can't pass here{_RESET}")
                    continue

                # Play with optional Tichu
                call_tichu = False
                if raw_input.endswith("+t"):
                    call_tichu = True
                    raw_input = raw_input[:-2].strip()

                cards = parse_cards(raw_input)
                if cards is None:
                    print(f"  {RED}Could not parse. Use '+t' suffix for Tichu (e.g. '3g 3b +t'){_RESET}")
                    continue

                phoenix_as = None
                meanings = legal_phoenix_meanings(gs, player, seat, cards, call_tichu)
                if len(meanings) == 2:
                    meaning = ask_phoenix_meaning(meanings)
                    if meaning is None:
                        continue
                    phoenix_as = meaning.phoenix_as
                elif len(meanings) == 1:
                    phoenix_as = meanings[0].phoenix_as

                undo_stack.append(copy.deepcopy(gs))
                ok = force_play(
                    gs,
                    player,
                    cards,
                    seat,
                    call_tichu=call_tichu,
                    phoenix_as=phoenix_as,
                )
                if not ok:
                    undo_stack.pop()
                    print(
                        f"  {RED}Illegal or inconsistent play. "
                        f"If those cards are in your hand, the advisor state is wrong.{_RESET}"
                    )
                    continue
                
                combo = classify_cards(cards)
                type_name = COMBO_NAMES.get(combo.type, "") if combo else ""
                tichu_tag = f" {RED}+ TICHU!{_RESET}" if call_tichu else ""
                print(f"  P{player}: {fmt_cards(cards)} ({type_name}){tichu_tag}")

                # Mahjong wish
                if MAHJONG_INDEX in cards and not gs.round_over:
                    print(f"  {CYAN}Did P{player} wish? Enter rank 2-10, J, Q, K, A (0=none):{_RESET}")
                    wish = input_wish(f"  {_BOLD}Wish: {_RESET}")
                    if wish == UNDO:
                        gs = undo_stack.pop()
                        print(f"  {YELLOW}Undid last action.{_RESET}")
                        cards = None
                        break
                    if wish is not None:
                        gs.set_mahjong_wish(wish)
                break
            if cards is None:
                continue

        # Player went out
        if gs.finish_order and gs.finish_order[-1] == player:
            pos = len(gs.finish_order)
            who = "You" if player == seat else f"P{player}"
            print(f"  {GREEN}* {who} went out (#{pos})!{_RESET}")

    # Game over
    print(f"\n{'─' * 60}")
    print(f"  {_BOLD}ROUND OVER{_RESET}")
    print(f"  Finish order: {' → '.join(f'P{p}' for p in gs.finish_order)}")
    a, b = gs.round_scores
    print(f"  Team A (P0+P2): {a:+d}")
    print(f"  Team B (P1+P3): {b:+d}")
    print(f"{'─' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Tichu advisor for real-life games")
    parser.add_argument(
        "--checkpoint", type=str,
        default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pass-checkpoint", type=str, default=None)
    parser.add_argument("--seat", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, pass_policy, gt_policy, kind, info = load_play_bundle(
        args.checkpoint, args.pass_checkpoint, device
    )
    if gt_policy is not None:
        print("Loaded Grand Tichu model")

    games_trained = info["games_played"]
    trained_str = f"{games_trained:,}" if isinstance(games_trained, int) else str(games_trained)
    print(f"Loaded {args.checkpoint} ({kind}, trained on {trained_str} games)")
    if pass_policy is not None:
        print("Loaded passing model")
    else:
        print("No passing model loaded; passing suggestions disabled")

    run(policy, args.seat, device, kind=kind, pass_policy=pass_policy, gt_policy=gt_policy)


if __name__ == "__main__":
    main()
