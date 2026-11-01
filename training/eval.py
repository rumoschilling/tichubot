"""Evaluate two checkpoints by pitting them against each other.

Team A (players 0, 2) uses policy_a.
Team B (players 1, 3) uses policy_b.

Usage:
    python -m training.eval checkpoints/ckpt_00010240.pt checkpoints/ckpt_00050176.pt
    python -m training.eval checkpoints/ckpt_00010240.pt checkpoints/ckpt_00050176.pt --games 1000
    python -m training.eval tiny-kaggle tiny-history-fresh --games 1000
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
from torch.distributions import Categorical

from training.checkpoints import load_policy_checkpoint, resolve_checkpoint
from training.model_embed import ACTION_DIM
from training.model_passing import TichuPassingPolicy
from training.observations import encode_policy_static
from training.rollout_embed import _action_vec
from tichu_env.cards import MAHJONG_INDEX, NUM_CARDS
from tichu_env.encoding import (
    encode_history,
    encode_pass_obs,
    encode_pass_source_ids,
    encode_pass_target_ids,
    encode_static,
)
from tichu_env.constants import MAX_ACTIONS
from tichu_env.game_state import GameState, TEAM_A


TEAM_A_PLAYERS = {0, 2}
TEAM_B_PLAYERS = {1, 3}


def _load_pass_policy(
    policy_ckpt: dict,
    pass_ckpt_path: str | None,
    device: torch.device,
    label: str,
):
    pass_state = None
    source = None

    if pass_ckpt_path is not None:
        resolved_pass_path = resolve_checkpoint(pass_ckpt_path)
        pass_ckpt = torch.load(resolved_pass_path, map_location=device, weights_only=False)
        pass_state = pass_ckpt.get("pass_model")
        source = str(resolved_pass_path)
    else:
        pass_state = policy_ckpt.get("pass_model")
        if pass_state is not None:
            source = "bundled"

    # Fallback to global passing model if still missing
    if pass_state is None:
        import os
        fallback_path = "checkpoints/passing_fallback/latest.pt"
        if os.path.exists(fallback_path):
            try:
                pass_ckpt = torch.load(fallback_path, map_location=device, weights_only=False)
                pass_state = pass_ckpt.get("pass_model")
                source = f"fallback: {fallback_path}"
            except Exception:
                pass_state = None

    if pass_state is None:
        print(f"  No passing model for {label}; using random passing", flush=True)
        return None

    pass_policy = TichuPassingPolicy()
    pass_policy.load_state_dict(pass_state)
    pass_policy.to(device)
    pass_policy.eval()
    print(f"  Loaded passing model for {label}  ({source})", flush=True)
    return pass_policy


def _spec_head_deciders(policy, kind: str, device: torch.device, label: str):
    """Build (decide_dragon, decide_wish) callables from a history model's
    specialist heads. Returns (None, None) for models without them, in which
    case the caller falls back to random decisions."""
    if kind != "history" or not hasattr(policy, "dragon_head"):
        return None, None

    def _fused(gs: GameState, player: int) -> torch.Tensor:
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

    def _sample(logits: torch.Tensor, noise) -> int:
        if noise is None:
            return int(Categorical(logits=logits).sample())
        u = torch.as_tensor(noise, dtype=logits.dtype, device=logits.device).clamp(1e-20, 1.0 - 1e-7)
        return int((logits.reshape(-1) - torch.log(-torch.log(u))).argmax())

    @torch.no_grad()
    def decide_dragon(gs: GameState, player: int, options, rng: random.Random, noise=None) -> int:
        """``noise``: 3 uniforms (2 Gumbel slots + fallback) for common random numbers."""
        logits = policy.get_dragon_gift_logits(_fused(gs, player))
        choice = _sample(logits, None if noise is None else noise[:2])
        target_abs = (player + (1 if choice == 0 else 3)) % 4
        if target_abs in options:
            return target_abs
        if noise is None:
            return rng.choice(options)
        return options[min(int(noise[2] * len(options)), len(options) - 1)]

    @torch.no_grad()
    def decide_wish(gs: GameState, player: int, noise=None) -> int | None:
        logits = policy.get_mahjong_wish_logits(_fused(gs, player))
        w_idx = _sample(logits, noise)
        return None if w_idx == 0 else w_idx + 1

    print(f"  Using trained dragon/wish heads for {label}", flush=True)
    return decide_dragon, decide_wish


def _hand_to_binary(hand: list[int]) -> np.ndarray:
    vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for card in hand:
        vec[card] = 1.0
    return vec


def _load_gt_policy(policy_ckpt: dict, device: torch.device, label: str):
    """Build the GT head bundled in a checkpoint, or None if absent."""
    gt_state = policy_ckpt.get("gt_model")
    if gt_state is None:
        print(f"  No GT model for {label}; never declaring grand tichu", flush=True)
        return None
    from training.model_gt import TichuGrandTichuPolicy
    cfg = policy_ckpt.get("config")
    hidden = getattr(cfg, "gt_hidden_dim", 64) if cfg is not None else 64
    layers = getattr(cfg, "gt_n_layers", 2) if cfg is not None else 2
    gt_policy = TichuGrandTichuPolicy(hidden_dim=hidden, n_layers=layers).to(device)
    gt_policy.load_state_dict(gt_state)
    gt_policy.eval()
    print(f"  Using trained GT head for {label}", flush=True)
    return gt_policy


def _run_gt_phase(
    games: list[GameState],
    gt_pol_a,
    gt_pol_b,
    device: torch.device,
    threshold: float | None = None,
    act_a: bool = True,
    act_b: bool = True,
) -> list[dict[str, bool]]:
    """Resolve the GRAND_TICHU phase (mirrors rollout_history, no explore eps).

    Teams without a GT policy always decline. With a threshold, declare
    deterministically iff sigmoid(logit) > threshold instead of sampling.

    For a team whose head exists but is disabled (act_*=False), the head's
    choice is still computed as a hypothetical — those rounds are the
    hand-matched control group for measuring the value of declaring.
    Returns one dict per game: "a"/"b" = head wanted to declare, and
    "a_players"/"b_players" = which seats wanted it."""
    wanted = [{"a": False, "b": False, "a_players": [], "b_players": []}
              for _ in games]
    for gi, gs in enumerate(games):
        while gs.phase.name == "GRAND_TICHU":
            p = gs.current_player
            team = "a" if p in TEAM_A_PLAYERS else "b"
            gt_policy = gt_pol_a if team == "a" else gt_pol_b
            act = act_a if team == "a" else act_b
            declare = False
            if gt_policy is not None:
                h8 = sorted(list(gs.hands[p]))[:8]
                h8_bin = torch.as_tensor(_hand_to_binary(h8)[None, :], device=device)
                with torch.no_grad():
                    if threshold is not None:
                        logit = gt_policy.call_logits(h8_bin).squeeze(-1)
                        declare = bool(torch.sigmoid(logit).item() > threshold)
                    else:
                        decl, _ = gt_policy.act(h8_bin)
                        declare = bool(decl.item())
            if declare:
                wanted[gi][team] = True
                wanted[gi][team + "_players"].append(p)
            gs.declare_grand_tichu(p, declare and act)
    return wanted


def _run_passing_phase(
    games: list[GameState],
    pass_a,
    pass_b,
    device: torch.device,
    rng: random.Random,
):
    for gs in games:
        while gs.phase.name == "PASSING":
            player = gs.current_player
            hand = gs.get_passing_hand(player)
            pass_policy = pass_a if player in TEAM_A_PLAYERS else pass_b

            if pass_policy is None:
                choices = rng.sample(hand, 3)
            else:
                pass_obs = torch.as_tensor(
                    encode_pass_obs(gs, player)[None, :], device=device
                )
                hand_cards = torch.as_tensor(
                    np.array(hand, dtype=np.int64)[None, :], device=device
                )
                with torch.no_grad():
                    choices_t, _ = pass_policy.act(pass_obs, hand_cards)
                idxs = choices_t[0].cpu().numpy()
                choices = [hand[int(i)] for i in idxs]

            gs.pass_cards(player, choices[0], choices[1], choices[2])


def evaluate(
    path_a: str,
    path_b: str,
    pass_path_a: str | None = None,
    pass_path_b: str | None = None,
    n_games: int = 500,
    batch: int = 64,
    device: torch.device = torch.device("cpu"),
    progress_every: int = 256,
    gt_a: bool = False,
    gt_b: bool = False,
    gt_threshold: float | None = None,
    post_pass_hook=None,
    decision_hook=None,
    post_action_hook=None,
    post_round_hook=None,
) -> dict[str, float]:
    """post_pass_hook(games, gt_wanted) fires after passing, before play;
    decision_hook(games, team, active, values) fires after each policy forward
    pass and before those actions are applied. ``team`` is ``"a"`` or ``"b"``;
    ``values`` is in the acting players' team-relative frame.
    post_action_hook(game, team, player, action) fires immediately after the
    selected main-play action is applied.
    post_round_hook(games, gt_wanted) after the same batch's rounds end.
    gt_wanted is None unless a GT flag is set (see _run_gt_phase)."""
    loaded_a = load_policy_checkpoint(path_a, device)
    loaded_b = load_policy_checkpoint(path_b, device)
    for label, loaded in (("A", loaded_a), ("B", loaded_b)):
        games = loaded.checkpoint.get("games_played", "?")
        games_label = f"{games:,}" if isinstance(games, int) else str(games)
        print(
            f"  Loaded {label}: {loaded.path}  ({games_label} games, {loaded.kind})",
            flush=True,
        )

    policy_a = loaded_a.policy
    policy_b = loaded_b.policy
    kind_a = loaded_a.kind
    kind_b = loaded_b.kind
    pass_a = _load_pass_policy(loaded_a.checkpoint, pass_path_a, device, "A")
    pass_b = _load_pass_policy(loaded_b.checkpoint, pass_path_b, device, "B")
    needs_history = kind_a == "history" or kind_b == "history"
    dragon_a, wish_a = _spec_head_deciders(policy_a, kind_a, device, "A")
    dragon_b, wish_b = _spec_head_deciders(policy_b, kind_b, device, "B")
    with_calling = gt_a or gt_b
    # Load both heads whenever the GT phase runs: a disabled team's head still
    # produces hypothetical declares (the hand-matched control group).
    gt_pol_a = _load_gt_policy(loaded_a.checkpoint, device, "A") if with_calling else None
    gt_pol_b = _load_gt_policy(loaded_b.checkpoint, device, "B") if with_calling else None

    wins_a = 0
    wins_b = 0
    score_diffs: list[float] = []
    gt_declares = {"a": 0, "b": 0}
    gt_made = {"a": 0, "b": 0}
    # Small tichu (declared via the action space during play, +/-100)
    tichu_declares = {"a": 0, "b": 0}
    tichu_made = {"a": 0, "b": 0}
    # per team: score diff (from that team's perspective) in rounds where its
    # head wanted to declare — actual declares if enabled, control if not
    diffs_wanted = {"a": [], "b": []}

    rng = random.Random()
    games_done = 0
    t0 = time.perf_counter()
    next_progress = min(progress_every, n_games) if progress_every > 0 else None

    while games_done < n_games:
        n = min(batch, n_games - games_done)

        games: list[GameState] = []
        for _ in range(n):
            gs = GameState(track_history=needs_history)
            # Team A (seats 0/2) vs team B (seats 1/3) are different models:
            # partner same-weights bit stays 1, opponents are foreign.
            gs.player_model_ids = [0, 1, 0, 1]
            gs.reset(with_calling=with_calling, with_passing=True)
            games.append(gs)

        gt_wanted = None
        if with_calling:
            gt_wanted = _run_gt_phase(games, gt_pol_a, gt_pol_b, device,
                                      gt_threshold, act_a=gt_a, act_b=gt_b)
        _run_passing_phase(games, pass_a, pass_b, device, rng)
        if post_pass_hook is not None:
            post_pass_hook(games, gt_wanted)

        done = [False] * n

        while not all(done):
            # Dragon-give decisions (trained head if available, else random)
            for i, gs in enumerate(games):
                if done[i]:
                    continue
                while gs.dragon_give_pending is not None and not gs.round_over:
                    giver = gs.dragon_give_pending
                    opts = gs.get_dragon_give_options(giver)
                    decide = dragon_a if giver in TEAM_A_PLAYERS else dragon_b
                    if decide is not None:
                        gs.give_dragon_trick(giver, decide(gs, giver, opts, rng))
                    else:
                        gs.give_dragon_trick(giver, rng.choice(opts))
                if gs.round_over:
                    done[i] = True

            if all(done):
                break

            # Collect active steps, split by team
            active_a: list[tuple[int, int, np.ndarray, int, list]] = []
            active_b: list[tuple[int, int, np.ndarray, int, list]] = []

            for i, gs in enumerate(games):
                if done[i]:
                    continue
                player = gs.current_player
                acting_kind = kind_a if player in TEAM_A_PLAYERS else kind_b
                legal = (
                    gs.get_legal_actions(player)
                    if acting_kind == "fixed"
                    else gs.get_policy_legal_actions(player)
                )
                acting_policy = (
                    policy_a if player in TEAM_A_PLAYERS else policy_b
                )
                obs = encode_policy_static(acting_policy, gs, player)
                pass_src = encode_pass_source_ids(gs, player)
                pass_tgt = encode_pass_target_ids(gs, player)
                if needs_history:
                    hist, hist_len = encode_history(gs, player)
                    entry = (i, player, obs, pass_src, pass_tgt, len(legal), legal, hist, hist_len)
                else:
                    entry = (i, player, obs, pass_src, pass_tgt, len(legal), legal, None, 0)
                if player in TEAM_A_PLAYERS:
                    active_a.append(entry)
                else:
                    active_b.append(entry)

            # Forward pass for each team — supports fixed, embed, and history models
            def _act(policy, kind, active):
                if not active:
                    return {}, np.empty(0, dtype=np.float32)
                obs_t = torch.as_tensor(
                    np.stack([x[2] for x in active]), dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    if kind == "fixed":
                        n_legal_arr = np.array([x[5] for x in active], dtype=np.int16)
                        idx_t  = torch.arange(MAX_ACTIONS, device=device)
                        mask_t = idx_t.unsqueeze(0) < torch.as_tensor(
                            n_legal_arr, device=device
                        ).unsqueeze(1)
                        actions_t, _, values_t = policy.act(obs_t, mask_t)
                    else:
                        # Embed or history: build action vectors
                        B = len(active)
                        n_arr  = np.array([len(x[6]) for x in active], dtype=np.int32)
                        max_n  = int(n_arr.max())
                        pad    = np.zeros((B, max_n, ACTION_DIM), dtype=np.float32)
                        for j, entry in enumerate(active):
                            for k, a in enumerate(entry[6]):
                                pad[j, k] = _action_vec(a)
                        action_args = [
                            obs_t,
                            torch.as_tensor(pad, device=device),
                            torch.as_tensor(n_arr, device=device),
                            torch.as_tensor(
                                np.stack([x[3] for x in active]),
                                dtype=torch.int64,
                                device=device,
                            ),
                            torch.as_tensor(
                                np.stack([x[4] for x in active]),
                                dtype=torch.int64,
                                device=device,
                            ),
                        ]
                        if kind == "history":
                            from tichu_env.encoding import PLAY_TOKEN_SIZE
                            pass_tgt = action_args.pop()
                            pass_src = action_args.pop()
                            hist_list = [x[7] for x in active]  # (200, 63) each
                            hist_lens = np.array([x[8] for x in active], dtype=np.int32)
                            max_h = int(hist_lens.max()) if hist_lens.max() > 0 else 1
                            hist_batch = np.zeros((B, max_h, PLAY_TOKEN_SIZE), dtype=np.float32)
                            for j, h in enumerate(hist_list):
                                hist_batch[j] = h[:max_h]
                            action_args += [
                                torch.as_tensor(hist_batch, device=device),
                                torch.as_tensor(hist_lens, device=device),
                                pass_src,
                                pass_tgt,
                            ]
                        actions_t, _, values_t = policy.act(*action_args)
                chosen = {
                    entry[0]: int(actions_t[j].item())
                    for j, entry in enumerate(active)
                }
                return chosen, values_t.detach().cpu().numpy()

            chosen_a, values_a = _act(policy_a, kind_a, active_a)
            chosen_b, values_b = _act(policy_b, kind_b, active_b)
            if decision_hook is not None:
                if active_a:
                    decision_hook(games, "a", active_a, values_a)
                if active_b:
                    decision_hook(games, "b", active_b, values_b)
            chosen   = {**chosen_a, **chosen_b}

            # Execute actions
            for team, active in (("a", active_a), ("b", active_b)):
                for (game_idx, player, obs, _ps, _pt, n_legal, legal, _h, _hl) in active:
                    gs         = games[game_idx]
                    action_idx = chosen[game_idx]
                    action     = legal[action_idx] if action_idx < len(legal) else legal[0]

                    gs.play(player, action)
                    if post_action_hook is not None:
                        post_action_hook(gs, team, player, action)

                    cards = action[0]
                    if (MAHJONG_INDEX in cards and not gs.round_over
                            and gs.mahjong_wish is None):
                        decide = wish_a if player in TEAM_A_PLAYERS else wish_b
                        if decide is not None:
                            gs.set_mahjong_wish(decide(gs, player))
                        elif rng.random() < 0.5:
                            gs.set_mahjong_wish(rng.randint(2, 14))

                    if gs.round_over:
                        done[game_idx] = True

        # Tally results
        for gi, gs in enumerate(games):
            sa, sb = gs.round_scores
            score_diffs.append(sa - sb)
            if sa > sb:
                wins_a += 1
            elif sb > sa:
                wins_b += 1
            for p in range(4):
                if gs.tichu_declarations[p] == 1:
                    team = "a" if p in TEAM_A_PLAYERS else "b"
                    tichu_declares[team] += 1
                    if gs.finish_order[0] == p:
                        tichu_made[team] += 1
            if with_calling:
                for p in range(4):
                    if gs.tichu_declarations[p] == 2:
                        team = "a" if p in TEAM_A_PLAYERS else "b"
                        gt_declares[team] += 1
                        if gs.finish_order[0] == p:
                            gt_made[team] += 1
                if gt_wanted[gi]["a"]:
                    diffs_wanted["a"].append(sa - sb)
                if gt_wanted[gi]["b"]:
                    diffs_wanted["b"].append(sb - sa)

        if post_round_hook is not None:
            post_round_hook(games, gt_wanted)

        games_done += n

        if next_progress is not None and games_done >= next_progress:
            elapsed = time.perf_counter() - t0
            gps = games_done / elapsed if elapsed > 0 else 0.0
            remaining = n_games - games_done
            eta = remaining / gps if gps > 0 else float("inf")
            print(
                f"  progress: {games_done}/{n_games} games"
                f"  |  {gps:.1f} g/s"
                f"  |  elapsed {elapsed:.0f}s"
                f"  |  eta {eta:.0f}s"
            , flush=True)
            while next_progress is not None and next_progress <= games_done:
                next_progress += progress_every

    total = wins_a + wins_b + (n_games - wins_a - wins_b)
    results = {
        "games":      n_games,
        "win_rate_a": wins_a / n_games,
        "win_rate_b": wins_b / n_games,
        "draws":      (n_games - wins_a - wins_b) / n_games,
        "mean_score_diff": float(np.mean(score_diffs)),
    }
    for team in ("a", "b"):
        results[f"tichu_declares_{team}"] = tichu_declares[team]
        results[f"tichu_made_{team}"] = tichu_made[team]
    if with_calling:
        for team in ("a", "b"):
            d, m = gt_declares[team], gt_made[team]
            results[f"gt_declares_{team}"] = d
            results[f"gt_made_{team}"] = m
            # each made GT is +200, each missed one -200
            results[f"gt_points_{team}"] = 200 * m - 200 * (d - m)
            wanted = diffs_wanted[team]
            results[f"gt_wanted_n_{team}"] = len(wanted)
            results[f"gt_wanted_diff_{team}"] = (
                float(np.mean(wanted)) if wanted else 0.0)
    return results


def main():
    parser = argparse.ArgumentParser(description="Pit two checkpoints against each other")
    parser.add_argument("policy_a", type=str, help="Path to checkpoint A (team 0,2)")
    parser.add_argument("policy_b", type=str, help="Path to checkpoint B (team 1,3)")
    parser.add_argument("--pass-a", type=str, default=None,
                        help="Optional separate passing checkpoint for A")
    parser.add_argument("--pass-b", type=str, default=None,
                        help="Optional separate passing checkpoint for B")
    parser.add_argument("--games",  type=int, default=500)
    parser.add_argument("--batch",  type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--progress-every", type=int, default=256,
        help="Print progress after this many completed games (0 disables progress output)",
    )
    parser.add_argument("--gt-a", action="store_true",
                        help="Team A declares grand tichu via its trained GT head")
    parser.add_argument("--gt-b", action="store_true",
                        help="Team B declares grand tichu via its trained GT head")
    parser.add_argument("--gt-threshold", type=float, default=None,
                        help="Declare iff P(GT) > this, instead of sampling "
                             "(applies to both GT-enabled teams)")
    args = parser.parse_args()

    print(f"\nEvaluating over {args.games} games:", flush=True)
    results = evaluate(
        args.policy_a, args.policy_b,
        pass_path_a=args.pass_a,
        pass_path_b=args.pass_b,
        n_games=args.games,
        batch=args.batch,
        device=torch.device(args.device),
        progress_every=args.progress_every,
        gt_a=args.gt_a,
        gt_b=args.gt_b,
        gt_threshold=args.gt_threshold,
    )

    print(f"\nResults ({results['games']} games):", flush=True)
    print(f"  A win rate: {results['win_rate_a']:.1%}  ({args.policy_a})", flush=True)
    print(f"  B win rate: {results['win_rate_b']:.1%}  ({args.policy_b})", flush=True)
    print(f"  Draws:      {results['draws']:.1%}", flush=True)
    print(f"  Mean score diff (A - B): {results['mean_score_diff']:+.1f}", flush=True)
    n_rounds = results["games"]
    for team in ("a", "b"):
        d, m = results[f"tichu_declares_{team}"], results[f"tichu_made_{team}"]
        made_pct = f"{m / d:.1%}" if d else "n/a"
        net = 100 * m - 100 * (d - m)
        print(f"  Tichu {team.upper()}: called {d} ({d / n_rounds:.1%} of rounds), "
              f"made {made_pct}, net {net / n_rounds:+.1f} pts/round", flush=True)
    if args.gt_a or args.gt_b:
        n = results["games"]
        for team, flag in (("a", args.gt_a), ("b", args.gt_b)):
            d, m = results[f"gt_declares_{team}"], results[f"gt_made_{team}"]
            pts = results[f"gt_points_{team}"]
            made_pct = f"{m / d:.1%}" if d else "n/a"
            print(f"  GT {team.upper()}: {'on' if flag else 'off'}  "
                  f"declared {d} ({d / n:.1%} of rounds), made {made_pct}, "
                  f"net {pts / n:+.1f} pts/round", flush=True)
        for team, flag in (("a", args.gt_a), ("b", args.gt_b)):
            nw = results[f"gt_wanted_n_{team}"]
            dw = results[f"gt_wanted_diff_{team}"]
            label = ("declared" if flag else "control: head wanted, not declared")
            print(f"  {team.upper()} monster-hand rounds ({label}): "
                  f"{dw:+.1f}/round from {team.upper()}'s view  (n={nw})",
                  flush=True)


if __name__ == "__main__":
    main()
