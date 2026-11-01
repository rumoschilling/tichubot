"""Model arena: pit N checkpoints against and alongside each other.

Pure self-play "inbreeds": a policy only ever coordinates with its own
weights. The arena measures how checkpoints perform against foreign models
AND partnered with them, and sets GameState.player_model_ids truthfully for
every seating so the partner same-weights bit (obs[195]) reflects reality.

Seating convention: seating[k] = model index controlling seat k.
Team A = seats (0, 2), team B = seats (1, 3); score diff = A - B.

Matchups per unordered pair (i, j), each pooled with its team-swapped
seating (sign-corrected) so seat/first-leader bias cancels:

  versus  [i,j,i,j] + [j,i,j,i]   pure-i team against pure-j team
  mixed   [i,j,j,i] + [j,i,i,j]   both teams are one i + one j; the pooled
                                  diff should be ~0 (balance sanity) while
                                  the games exercise cross-model play
  coop    [i,i,j,i] + [i,i,i,j]   the mixed pair (i+j) against the pure pair
                                  (i+i): what partnering with a foreign model
                                  costs relative to inbred coordination.
                                  Ordered — C[i][j] uses the pure-i baseline.

Primary metric: mean score diff per round (± stderr); win rate secondary.

Usage:
  python -m training.arena v29=path/a.pt v28=path/b.pt [more ...] \
      --games 200 --batch 64 --device cpu --json arena.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tichu_env.cards import MAHJONG_INDEX
from tichu_env.encoding import (
    PLAY_TOKEN_SIZE,
    encode_history,
    encode_pass_obs,
    encode_pass_source_ids,
    encode_pass_target_ids,
    encode_static,
)
from tichu_env.constants import MAX_ACTIONS
from tichu_env.game_state import GameState
from training.checkpoints import load_policy_checkpoint, tag_legacy_label
from training.eval import (
    _hand_to_binary,
    _load_gt_policy,
    _load_pass_policy,
    _spec_head_deciders,
)
from training.model_embed import ACTION_DIM
from training.observations import encode_policy_static
from training.rollout_embed import _action_vec


@dataclass
class ArenaModel:
    label: str
    policy: torch.nn.Module
    kind: str
    pass_policy: object | None
    gt_policy: object | None
    decide_dragon: object | None
    decide_wish: object | None


def _short_label(ref: str) -> str:
    path = Path(ref)
    if path.stem in ("latest", "final") and len(path.parts) >= 2:
        for part in reversed(path.parts[:-1]):
            if part not in ("checkpoints", "tichubot_outputs", "history_standard_gpu"):
                return part
    return path.stem


def load_arena_model(spec: str, device: torch.device) -> ArenaModel:
    label, sep, path = spec.partition("=")
    if not sep:
        label, path = _short_label(spec), spec
    loaded = load_policy_checkpoint(path, device)
    label = tag_legacy_label(label, loaded.checkpoint)
    games = loaded.checkpoint.get("games_played", "?")
    games_s = f"{games:,}" if isinstance(games, int) else str(games)
    print(f"  Loaded {label}: {loaded.path}  ({games_s} games, {loaded.kind})", flush=True)
    pass_policy = _load_pass_policy(loaded.checkpoint, None, device, label)
    gt_policy = _load_gt_policy(loaded.checkpoint, device, label)
    decide_dragon, decide_wish = _spec_head_deciders(
        loaded.policy, loaded.kind, device, label
    )
    return ArenaModel(
        label=label,
        policy=loaded.policy,
        kind=loaded.kind,
        pass_policy=pass_policy,
        gt_policy=gt_policy,
        decide_dragon=decide_dragon,
        decide_wish=decide_wish,
    )


class CRNStreams:
    """Common random numbers for one deal.

    Every random decision draws its uniforms from a stream keyed by
    (deal seed, seat, decision type), consumed in that seat's decision order.
    Both seatings of a paired comparison replay the same deal, so a decision
    that is the same in both games gets the same random numbers: identical
    policies play identical games, and similar policies stay coupled until
    they first choose differently. Each draw is still an exact sample from
    the acting policy, so expectations are unchanged; only the variance of
    the paired difference drops.
    """

    GT, PASS, PLAY, DRAGON, WISH = range(5)

    def __init__(self, deal_seed: int):
        self.deal_seed = int(deal_seed) % (2**63)
        self._gens: dict[tuple[int, int], np.random.Generator] = {}

    def uniforms(self, seat: int, kind: int, size: int) -> np.ndarray:
        key = (seat, kind)
        gen = self._gens.get(key)
        if gen is None:
            gen = np.random.default_rng(np.random.SeedSequence([self.deal_seed, seat, kind]))
            self._gens[key] = gen
        return gen.random(size)


@torch.no_grad()
def _act_batch(policy, kind: str, active: list[tuple], device: torch.device,
               noise: list[np.ndarray] | None = None) -> dict[int, int]:
    """Batched action selection for one policy. Mirrors training/eval.py.

    Entries: (game_idx, player, obs, pass_src, pass_tgt, n_legal, legal, hist, hist_len)
    Returns {game_idx: action_idx}.
    """
    if not active:
        return {}
    obs_t = torch.as_tensor(
        np.stack([x[2] for x in active]), dtype=torch.float32, device=device
    )
    if kind == "fixed":
        n_legal_arr = np.array([x[5] for x in active], dtype=np.int16)
        idx_t = torch.arange(MAX_ACTIONS, device=device)
        mask_t = idx_t.unsqueeze(0) < torch.as_tensor(
            n_legal_arr, device=device
        ).unsqueeze(1)
        actions_t, _, _ = policy.act(obs_t, mask_t)
    else:
        B = len(active)
        n_arr = np.array([len(x[6]) for x in active], dtype=np.int32)
        max_n = int(n_arr.max())
        pad = np.zeros((B, max_n, ACTION_DIM), dtype=np.float32)
        for j, entry in enumerate(active):
            for k, a in enumerate(entry[6]):
                pad[j, k] = _action_vec(a)
        action_args = [
            obs_t,
            torch.as_tensor(pad, device=device),
            torch.as_tensor(n_arr, device=device),
        ]
        pass_src_t = torch.as_tensor(
            np.stack([x[3] for x in active]), dtype=torch.int64, device=device
        )
        pass_tgt_t = torch.as_tensor(
            np.stack([x[4] for x in active]), dtype=torch.int64, device=device
        )
        if kind == "history":
            hist_lens = np.array([x[8] for x in active], dtype=np.int32)
            max_h = int(hist_lens.max()) if hist_lens.max() > 0 else 1
            hist_batch = np.zeros((B, max_h, PLAY_TOKEN_SIZE), dtype=np.float32)
            for j, entry in enumerate(active):
                hist_batch[j] = entry[7][:max_h]
            action_args += [
                torch.as_tensor(hist_batch, device=device),
                torch.as_tensor(hist_lens, device=device),
                pass_src_t,
                pass_tgt_t,
            ]
        else:
            action_args += [pass_src_t, pass_tgt_t]
        if noise is not None:
            if kind != "history":
                raise ValueError("common random numbers need history policies")
            noise_t = np.full((B, max_n), 0.5, dtype=np.float32)
            for j, row in enumerate(noise):
                noise_t[j, : len(row)] = row
            actions_t, _, _ = policy.act(*action_args, noise=torch.as_tensor(noise_t, device=device))
        else:
            actions_t, _, _ = policy.act(*action_args)
    # One device-to-host copy, not one sync per element.
    actions = actions_t.cpu().tolist()
    return {entry[0]: int(actions[j]) for j, entry in enumerate(active)}


@torch.no_grad()
def _run_arena_gt_phase(
    models: list[ArenaModel],
    seating: tuple[int, int, int, int],
    games: list[GameState],
    device: torch.device,
    crn: list[CRNStreams] | None = None,
) -> None:
    """Resolve deployed GT heads in batches grouped by controlling model."""
    while any(gs.phase.name == "GRAND_TICHU" for gs in games):
        by_model: dict[int, list[tuple[int, int]]] = {}
        for game_idx, gs in enumerate(games):
            if gs.phase.name == "GRAND_TICHU":
                player = gs.current_player
                by_model.setdefault(seating[player], []).append(
                    (game_idx, player)
                )

        decisions: dict[int, bool] = {}
        for model_idx, active in by_model.items():
            gt_policy = models[model_idx].gt_policy
            if gt_policy is None:
                decisions.update({game_idx: False for game_idx, _ in active})
                continue
            hands = np.stack(
                [
                    _hand_to_binary(
                        sorted(games[game_idx].hands[player])[:8]
                    )
                    for game_idx, player in active
                ]
            )
            hands_t = torch.as_tensor(hands, dtype=torch.float32, device=device)
            if crn is None:
                declares, _ = gt_policy.act(hands_t)
                declares = declares.cpu().tolist()
            else:
                probs = torch.sigmoid(gt_policy.call_logits(hands_t).reshape(-1)).cpu().numpy()
                u = np.array([crn[g].uniforms(p, CRNStreams.GT, 1)[0] for g, p in active])
                declares = (u < probs).tolist()
            decisions.update(
                {
                    game_idx: bool(declares[row])
                    for row, (game_idx, _player) in enumerate(active)
                }
            )

        for active in by_model.values():
            for game_idx, player in active:
                games[game_idx].declare_grand_tichu(
                    player,
                    decisions[game_idx],
                )


@torch.no_grad()
def _run_arena_passing_phase(
    models: list[ArenaModel],
    seating: tuple[int, int, int, int],
    games: list[GameState],
    device: torch.device,
    rng: random.Random,
    crn: list[CRNStreams] | None = None,
) -> None:
    """Resolve passing in a few model-batched calls instead of per game."""
    while any(gs.phase.name == "PASSING" for gs in games):
        by_model: dict[int, list[tuple[int, int, list[int]]]] = {}
        for game_idx, gs in enumerate(games):
            if gs.phase.name == "PASSING":
                player = gs.current_player
                by_model.setdefault(seating[player], []).append(
                    (game_idx, player, gs.get_passing_hand(player))
                )

        choices_by_game: dict[int, list[int]] = {}
        for model_idx, active in by_model.items():
            pass_policy = models[model_idx].pass_policy
            if pass_policy is None:
                for game_idx, player, hand in active:
                    if crn is None:
                        choices_by_game[game_idx] = rng.sample(hand, 3)
                    else:
                        order = np.argsort(crn[game_idx].uniforms(player, CRNStreams.PASS, len(hand)))
                        choices_by_game[game_idx] = [hand[int(i)] for i in order[:3]]
                continue
            pass_obs = torch.as_tensor(
                np.stack(
                    [encode_pass_obs(games[game_idx], player)
                     for game_idx, player, _hand in active]
                ),
                dtype=torch.float32,
                device=device,
            )
            hand_cards = torch.as_tensor(
                np.asarray([hand for _game_idx, _player, hand in active]),
                dtype=torch.int64,
                device=device,
            )
            if crn is None:
                choice_rows, _ = pass_policy.act(pass_obs, hand_cards)
            else:
                width = hand_cards.shape[1]
                noise = np.stack([
                    crn[game_idx].uniforms(player, CRNStreams.PASS, 3 * width).reshape(3, width)
                    for game_idx, player, _hand in active
                ]).astype(np.float32)
                choice_rows, _ = pass_policy.act(
                    pass_obs, hand_cards, noise=torch.as_tensor(noise, device=device))
            for row, (game_idx, _player, hand) in enumerate(active):
                choices_by_game[game_idx] = [
                    hand[int(index)] for index in choice_rows[row].cpu().numpy()
                ]

        for active in by_model.values():
            for game_idx, player, _hand in active:
                choices = choices_by_game[game_idx]
                games[game_idx].pass_cards(
                    player,
                    choices[0],
                    choices[1],
                    choices[2],
                )


@torch.no_grad()
def play_matchup(
    models: list[ArenaModel],
    seating: tuple[int, int, int, int],
    n_games: int,
    batch: int = 64,
    device: torch.device = torch.device("cpu"),
    rng: random.Random | None = None,
    game_seeds: list[int] | np.ndarray | None = None,
    with_calling: bool = False,
    common_random_numbers: bool = False,
) -> np.ndarray:
    """Play n_games with models[seating[k]] controlling seat k.

    Returns per-game score diffs (team A: seats 0/2, minus team B: seats 1/3).
    ``common_random_numbers`` draws every random decision from per-deal
    streams (CRNStreams), so two seatings of the same deals are coupled.
    """
    rng = rng or random.Random()
    if game_seeds is not None and len(game_seeds) != n_games:
        raise ValueError("game_seeds must contain exactly n_games seeds")
    if common_random_numbers and game_seeds is None:
        raise ValueError("common random numbers need explicit game_seeds")
    seat_model = [models[m] for m in seating]
    needs_history = any(sm.kind == "history" for sm in seat_model)

    diffs: list[float] = []
    games_done = 0
    while games_done < n_games:
        n = min(batch, n_games - games_done)

        games: list[GameState] = []
        for offset in range(n):
            gs = GameState(track_history=needs_history)
            # Truthful identity annotation: equal seating entries share
            # weights. Drives the partner same-weights bit obs[195].
            gs.player_model_ids = list(seating)
            game_seed = (
                None
                if game_seeds is None
                else int(game_seeds[games_done + offset])
            )
            gs.reset(
                seed=game_seed,
                with_calling=with_calling,
                with_passing=True,
            )
            games.append(gs)
        crn = (
            [CRNStreams(game_seeds[games_done + k]) for k in range(n)]
            if common_random_numbers else None
        )

        if with_calling:
            _run_arena_gt_phase(models, seating, games, device, crn)
        _run_arena_passing_phase(models, seating, games, device, rng, crn)

        done = [False] * n
        while not all(done):
            # Dragon-give decisions, per-seat head (random fallback)
            for i, gs in enumerate(games):
                if done[i]:
                    continue
                while gs.dragon_give_pending is not None and not gs.round_over:
                    giver = gs.dragon_give_pending
                    opts = gs.get_dragon_give_options(giver)
                    decide = seat_model[giver].decide_dragon
                    noise = None if crn is None else crn[i].uniforms(giver, CRNStreams.DRAGON, 3)
                    if decide is not None:
                        if noise is None:
                            gs.give_dragon_trick(giver, decide(gs, giver, opts, rng))
                        else:
                            gs.give_dragon_trick(giver, decide(gs, giver, opts, rng, noise=noise))
                    elif noise is None:
                        gs.give_dragon_trick(giver, rng.choice(opts))
                    else:
                        gs.give_dragon_trick(giver, opts[min(int(noise[2] * len(opts)), len(opts) - 1)])
                if gs.round_over:
                    done[i] = True

            if all(done):
                break

            # Collect active steps, grouped by which model acts
            by_model: dict[int, list[tuple]] = {}
            for i, gs in enumerate(games):
                if done[i]:
                    continue
                player = gs.current_player
                acting_model = models[seating[player]]
                legal = (
                    gs.get_legal_actions(player)
                    if acting_model.kind == "fixed"
                    else gs.get_policy_legal_actions(player)
                )
                acting_policy = acting_model.policy
                obs = encode_policy_static(acting_policy, gs, player)
                pass_src = encode_pass_source_ids(gs, player)
                pass_tgt = encode_pass_target_ids(gs, player)
                if needs_history:
                    hist, hist_len = encode_history(gs, player)
                else:
                    hist, hist_len = None, 0
                entry = (i, player, obs, pass_src, pass_tgt, len(legal), legal, hist, hist_len)
                by_model.setdefault(seating[player], []).append(entry)

            chosen: dict[int, int] = {}
            for m, active in by_model.items():
                noise = None if crn is None else [
                    crn[entry[0]].uniforms(entry[1], CRNStreams.PLAY, len(entry[6])) for entry in active
                ]
                chosen.update(_act_batch(models[m].policy, models[m].kind, active, device, noise))

            # Execute actions
            for active in by_model.values():
                for (game_idx, player, _obs, _ps, _pt, _n, legal, _h, _hl) in active:
                    gs = games[game_idx]
                    action_idx = chosen[game_idx]
                    action = legal[action_idx] if action_idx < len(legal) else legal[0]
                    gs.play(player, action)

                    cards = action[0]
                    if (MAHJONG_INDEX in cards and not gs.round_over
                            and gs.mahjong_wish is None):
                        decide = seat_model[player].decide_wish
                        noise = None if crn is None else crn[game_idx].uniforms(player, CRNStreams.WISH, 14)
                        if decide is not None:
                            if noise is None:
                                gs.set_mahjong_wish(decide(gs, player))
                            else:
                                gs.set_mahjong_wish(decide(gs, player, noise=noise))
                        elif noise is None:
                            if rng.random() < 0.5:
                                gs.set_mahjong_wish(rng.randint(2, 14))
                        elif noise[0] < 0.5:
                            gs.set_mahjong_wish(2 + min(int(noise[1] * 13), 12))

                    if gs.round_over:
                        done[game_idx] = True

        for gs in games:
            sa, sb = gs.round_scores
            diffs.append(float(sa - sb))
        games_done += n

    return np.asarray(diffs, dtype=np.float64)


@dataclass
class MatchupResult:
    diffs: np.ndarray  # sign-corrected: positive favors the first-named side

    @property
    def n(self) -> int:
        return len(self.diffs)

    @property
    def mean(self) -> float:
        return float(np.mean(self.diffs))

    @property
    def stderr(self) -> float:
        if len(self.diffs) < 2:
            return float("nan")
        return float(np.std(self.diffs, ddof=1) / np.sqrt(len(self.diffs)))

    @property
    def win_rate(self) -> float:
        return float(np.mean(self.diffs > 0))

    @property
    def draw_rate(self) -> float:
        return float(np.mean(self.diffs == 0))

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_score_diff": self.mean,
            "stderr": self.stderr,
            "win_rate": self.win_rate,
            "draw_rate": self.draw_rate,
        }

    def __str__(self) -> str:
        return f"{self.mean:+7.1f} ±{self.stderr:5.1f}  (win {self.win_rate:5.1%}, n={self.n})"


def _pooled(
    models: list[ArenaModel],
    seating_fwd: tuple[int, int, int, int],
    seating_swp: tuple[int, int, int, int],
    n_games: int,
    batch: int,
    device: torch.device,
    rng: random.Random,
) -> MatchupResult:
    """Run a seating and its team-swapped twin; pool with sign correction."""
    d_fwd = play_matchup(models, seating_fwd, n_games, batch, device, rng)
    d_swp = play_matchup(models, seating_swp, n_games, batch, device, rng)
    return MatchupResult(np.concatenate([d_fwd, -d_swp]))


def _team_swap(seating: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Swap teams: old team B (seats 1,3) becomes team A (seats 0,2)."""
    return (seating[1], seating[0], seating[3], seating[2])


def run_arena(
    models: list[ArenaModel],
    n_games: int = 200,
    batch: int = 64,
    device: torch.device = torch.device("cpu"),
    include_coop: bool = True,
    include_self: bool = False,
    seed: int | None = None,
    progress: bool = True,
) -> dict:
    """Run all matchups. n_games is per seating (each matchup pools 2 seatings)."""
    rng = random.Random(seed)
    if seed is not None:
        torch.manual_seed(seed)

    M = len(models)
    versus: dict[tuple[int, int], MatchupResult] = {}
    mixed: dict[tuple[int, int], MatchupResult] = {}
    coop: dict[tuple[int, int], MatchupResult] = {}
    selfplay: dict[int, MatchupResult] = {}

    def _log(msg: str) -> None:
        if progress:
            print(msg, flush=True)

    t0 = time.perf_counter()
    for i in range(M):
        for j in range(i + 1, M):
            li, lj = models[i].label, models[j].label
            _log(f"  versus  {li} vs {lj} ...")
            versus[(i, j)] = _pooled(
                models, (i, j, i, j), (j, i, j, i), n_games, batch, device, rng
            )
            _log(f"    {versus[(i, j)]}")
            _log(f"  mixed   {li}+{lj} vs {lj}+{li} ...")
            mixed[(i, j)] = _pooled(
                models, (i, j, j, i), (j, i, i, j), n_games, batch, device, rng
            )
            _log(f"    {mixed[(i, j)]}")
            if include_coop:
                for a, b in ((i, j), (j, i)):
                    la, lb = models[a].label, models[b].label
                    _log(f"  coop    {la}+{lb} vs {la}+{la} ...")
                    fwd = (a, a, b, a)
                    coop[(a, b)] = _pooled(
                        models, fwd, _team_swap(fwd), n_games, batch, device, rng
                    )
                    _log(f"    {coop[(a, b)]}")
    if include_self:
        for i in range(M):
            _log(f"  self    {models[i].label} sanity ...")
            selfplay[i] = MatchupResult(
                play_matchup(models, (i, i, i, i), n_games, batch, device, rng)
            )
            _log(f"    {selfplay[i]}")
    _log(f"  arena finished in {time.perf_counter() - t0:.0f}s")

    return {
        "labels": [m.label for m in models],
        "n_games_per_seating": n_games,
        "versus": versus,
        "mixed": mixed,
        "coop": coop,
        "self": selfplay,
    }


def format_report(results: dict) -> str:
    labels: list[str] = results["labels"]
    M = len(labels)
    versus = results["versus"]
    width = max(12, max(len(l) for l in labels) + 2)

    lines = ["", "Head-to-head mean score diff per round (row vs col, ± stderr):", ""]
    header = " " * width + "".join(f"{l:>{width}}" for l in labels)
    lines.append(header)
    for i in range(M):
        cells = []
        for j in range(M):
            if i == j:
                r = results["self"].get(i)
                cells.append(f"{r.mean:+.1f}±{r.stderr:.1f}".rjust(width) if r else "—".rjust(width))
            else:
                r = versus[(i, j)] if (i, j) in versus else versus[(j, i)]
                mean = r.mean if (i, j) in versus else -r.mean
                cells.append(f"{mean:+.1f}±{r.stderr:.1f}".rjust(width))
        lines.append(f"{labels[i]:<{width}}" + "".join(cells))

    if results["versus"]:
        lines += ["", "Win rates (row beats col):"]
        for (i, j), r in versus.items():
            lines.append(
                f"  {labels[i]} vs {labels[j]}: {r.win_rate:.1%} / "
                f"{1 - r.win_rate - r.draw_rate:.1%}  (draws {r.draw_rate:.1%}, n={r.n})"
            )

    if results["mixed"]:
        lines += ["", "Mixed-team balance (both teams i+j; should be ~0):"]
        for (i, j), r in results["mixed"].items():
            lines.append(f"  {labels[i]}+{labels[j]}: {r}")

    if results["coop"]:
        lines += [
            "",
            "Cross-model cooperation: team (i+j) vs pure team (i+i).",
            "Negative = partnering with the foreign model loses points",
            "relative to inbred coordination:",
        ]
        for (a, b), r in results["coop"].items():
            lines.append(f"  {labels[a]}+{labels[b]} vs {labels[a]}+{labels[a]}: {r}")

    return "\n".join(lines)


def results_to_json(results: dict) -> dict:
    labels = results["labels"]

    def keyed(d: dict) -> dict:
        return {
            f"{labels[k[0]]}|{labels[k[1]]}": r.as_dict() for k, r in d.items()
        }

    return {
        "labels": labels,
        "n_games_per_seating": results["n_games_per_seating"],
        "versus": keyed(results["versus"]),
        "mixed": keyed(results["mixed"]),
        "coop": keyed(results["coop"]),
        "self": {labels[k]: r.as_dict() for k, r in results["self"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round-robin arena for multiple checkpoints"
    )
    parser.add_argument(
        "models", nargs="+",
        help="Checkpoints as NAME=PATH (or bare paths; label derived from path)",
    )
    parser.add_argument("--games", type=int, default=200,
                        help="Games per seating (each matchup pools 2 seatings)")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", type=str, default=None,
                        help="Write machine-readable results to this file")
    parser.add_argument("--no-coop", action="store_true",
                        help="Skip the cooperation (mixed pair vs pure pair) matchups")
    parser.add_argument("--self-check", action="store_true",
                        help="Also run each model against itself (~0 sanity)")
    args = parser.parse_args()

    if len(args.models) < 2 and not args.self_check:
        parser.error("Need at least two models (or --self-check with one)")

    device = torch.device(args.device)
    print(f"\nLoading {len(args.models)} models:", flush=True)
    models = [load_arena_model(spec, device) for spec in args.models]

    labels = [m.label for m in models]
    if len(set(labels)) != len(labels):
        parser.error(f"Duplicate model labels: {labels} — use NAME=PATH to disambiguate")

    print(f"\nRunning arena ({args.games} games per seating):", flush=True)
    results = run_arena(
        models,
        n_games=args.games,
        batch=args.batch,
        device=device,
        include_coop=not args.no_coop,
        include_self=args.self_check,
        seed=args.seed,
    )

    print(format_report(results), flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results_to_json(results), fh, indent=2)
        print(f"\nWrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
