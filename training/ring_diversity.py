"""How differently the ring agents play, over time.

Every checkpoint scores the same fixed bank of decision states; two policies
are compared by the Jensen-Shannon divergence of their move distributions
(base 2, 0 = identical, 1 = disjoint) and by how often their most likely
move agrees. The agents are reconstructed as they stood after every phase
from the ring's phase records, so one run gives the whole history.

Moves that differ only in suit count as the same move (their probabilities
are pooled): in Tichu suits matter only for straight-flush bombs, so states
whose hand holds one keep every move separate, and bombs are never pooled
with non-bombs. Every divergence and agreement comes with a 95% interval
from a bootstrap over games (decisions from one game are correlated). Use
``--states 8000`` for plateau checks.

Style statistics on the same bank: probability of passing (where passing is
legal), of calling Tichu (where callable), of bombing (where a bomb is
legal), and the expected number of cards in a non-pass play.

Usage:
    python -m training.ring_diversity --state ring_runs/ring1/ring_state.json \
        --agents-dir ring_runs/ring1/agents --out local_evaluations/ring_diversity.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from tichu_env.combinations import classify_cards
from tichu_env.encoding import encode_history, encode_pass_source_ids, encode_pass_target_ids
from tichu_env.game_state import GameState, _enum_straight_bombs, _hand_analysis
from training.checkpoints import read_best_record
from training.model_embed import ACTION_DIM
from training.observations import encode_policy_static
from training.play import load_play_bundle
from training.rollout_embed import _action_vec

DEVICE = torch.device("cpu")


def _logits(policy, items: list[tuple]) -> list[np.ndarray]:
    """Move probabilities for (state, player, legal vecs) items, batched."""
    out = []
    for start in range(0, len(items), 128):
        batch = items[start:start + 128]
        n = np.array([len(v) for _, _, v in batch])
        pad = np.zeros((len(batch), n.max(), ACTION_DIM), dtype=np.float32)
        hists, lens = [], []
        for j, (gs, p, vecs) in enumerate(batch):
            pad[j, :len(vecs)] = vecs
            h, hl = encode_history(gs, p)
            hists.append(h)
            lens.append(hl)
        max_h = max(max(lens), 1)
        with torch.no_grad():
            logits, _ = policy._forward_scores(
                torch.as_tensor(np.stack([encode_policy_static(policy, gs, p) for gs, p, _ in batch]), dtype=torch.float32),
                torch.as_tensor(pad),
                torch.as_tensor(np.stack([h[:max_h] for h in hists]), dtype=torch.float32),
                torch.as_tensor(np.array(lens, dtype=np.int32)),
                torch.as_tensor(np.stack([encode_pass_source_ids(gs, p) for gs, p, _ in batch]), dtype=torch.int64),
                torch.as_tensor(np.stack([encode_pass_target_ids(gs, p) for gs, p, _ in batch]), dtype=torch.int64),
            )
        for j, k in enumerate(n):
            prob = torch.softmax(logits[j, :k].double(), dim=0).numpy()
            out.append(prob / prob.sum())
    return out


def build_state_bank(n_states: int, seed: int, record_prob: float = 0.15,
                     generator: str | None = None) -> list[dict]:
    """Decision states (>= 2 legal moves) from fixed-seed games of ``generator``
    (default: the best model)."""
    policy, pass_policy, _, _, _ = load_play_bundle(generator or read_best_record()["path"], None, DEVICE)
    policy.eval()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    bank: list[dict] = []
    game = 0
    while len(bank) < n_states:
        gs = GameState(track_history=True)
        gs.reset(seed=seed * 100_003 + game, with_calling=True, with_passing=True)
        game += 1
        for p in range(4):
            gs.declare_grand_tichu(p, False)
        for p in range(4):
            gs.pass_cards(p, *rng.sample(gs.get_passing_hand(p), 3))
        while not gs.round_over and len(bank) < n_states:
            if gs.dragon_give_pending is not None:
                g = gs.dragon_give_pending
                gs.give_dragon_trick(g, rng.choice(gs.get_dragon_give_options(g)))
                continue
            p = gs.current_player
            legal = gs.get_policy_legal_actions(p)
            vecs = np.array([_action_vec(a) for a in legal], dtype=np.float32)
            if len(legal) >= 2 and rng.random() < record_prob:
                bank.append({"gs": copy.deepcopy(gs), "player": p, "legal": legal, "vecs": vecs,
                             "game": game})
            probs = _logits(policy, [(gs, p, vecs)])[0]
            action = legal[int(np.random.default_rng(rng.randrange(2**32)).choice(len(legal), p=probs))]
            gs.play(p, action)
            if any(c == 52 for c in action[0]) and not gs.round_over and gs.mahjong_wish is None:
                gs.set_mahjong_wish(None)
    for entry in bank:
        legal = entry["legal"]
        entry["is_pass"] = np.array([not a[0] for a in legal])
        entry["is_tichu"] = np.array([bool(a[1]) for a in legal])
        entry["is_bomb"] = np.array([bool(a[0]) and bool(getattr(classify_cards(a[0]), "is_bomb", False)) for a in legal])
        entry["n_cards"] = np.array([len(a[0]) for a in legal], dtype=np.float64)
        entry["groups"] = move_groups(legal, entry["gs"].hands[entry["player"]], entry["is_bomb"])
    return bank


def move_groups(legal: list, hand, is_bomb: np.ndarray) -> np.ndarray:
    """Group id per legal move; moves equal up to suit share a group.

    Nothing is pooled when the hand holds a straight-flush bomb, because then
    which suit is played can matter.
    """
    if _enum_straight_bombs(_hand_analysis(sorted(hand))):
        return np.arange(len(legal))
    keys: dict[tuple, int] = {}
    groups = []
    for action, bomb in zip(legal, is_bomb):
        cards, tichu = action[0], action[1]
        high = action[2] if len(action) > 2 else False
        ranks = tuple(sorted((c % 13 + 2) if c < 52 else 100 + c for c in cards))
        groups.append(keys.setdefault((ranks, bool(tichu), bool(high), bool(bomb)), len(keys)))
    return np.array(groups)


def pooled(prob: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.bincount(groups, weights=prob, minlength=int(groups.max()) + 1)


def game_ci(values: np.ndarray, games: np.ndarray, samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of a per-state mean, resampling whole games."""
    ids, inverse = np.unique(games, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(samples, len(ids)))
    w = np.stack([np.bincount(d, minlength=len(ids)) for d in draws])
    means = (w @ sums) / (w @ counts)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def policy_probs(path: str, bank: list[dict]) -> list[np.ndarray]:
    policy, _, _, _, _ = load_play_bundle(path, None, DEVICE)
    policy.eval()
    return _logits(policy, [(e["gs"], e["player"], e["vecs"]) for e in bank])


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def per_state(a: list[np.ndarray], b: list[np.ndarray], bank: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-state JS divergence and top-move agreement on suit-pooled moves."""
    js, agree = [], []
    for p, q, e in zip(a, b, bank):
        pp, qq = pooled(p, e["groups"]), pooled(q, e["groups"])
        js.append(js_divergence(pp, qq))
        agree.append(float(np.argmax(pp) == np.argmax(qq)))
    return np.array(js), np.array(agree)


def compare(a: list[np.ndarray], b: list[np.ndarray], bank: list[dict]) -> dict:
    js, agree = per_state(a, b, bank)
    games = np.array([e["game"] for e in bank])
    return {"js": float(js.mean()), "js_ci": game_ci(js, games),
            "agreement": float(agree.mean()), "agreement_ci": game_ci(agree, games),
            "_js": js, "_agree": agree}


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def style(probs: list[np.ndarray], bank: list[dict]) -> dict[str, float]:
    def mass(key):
        vals = [float(p[e[key]].sum()) for p, e in zip(probs, bank) if e[key].any()]
        return float(np.mean(vals)) if vals else float("nan")

    cards = []
    for p, e in zip(probs, bank):
        play = ~e["is_pass"]
        if p[play].sum() > 0:
            cards.append(float((p[play] * e["n_cards"][play]).sum() / p[play].sum()))
    return {"pass": mass("is_pass"), "tichu": mass("is_tichu"), "bomb": mass("is_bomb"),
            "cards_per_play": float(np.mean(cards))}


def time_slices(state: dict, agents_dir: Path) -> list[dict]:
    """Each agent's checkpoint at the start and after every finished phase, plus now."""
    current = [s for s in state["snapshots"][:4]]
    slices = [{"label": "start", "ring_games": 0, "checkpoints": list(current)}]
    for phase in state["phases"]:
        for agent, path in zip(phase["trainees"], phase["checkpoints"]):
            current[agent] = str(agents_dir / f"agent{agent}" / Path(path).name)
        slices.append({"label": f"after phase {phase['phase']}", "ring_games": phase["ended_at_ring_games"],
                       "checkpoints": list(current)})
    slices.append({"label": "now", "ring_games": state["ring_games"],
                   "checkpoints": [str(agents_dir / f"agent{a}" / "latest.pt") for a in range(4)]})
    return slices


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", required=True)
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--slice-every", type=int, default=1,
                    help="score the start, every Nth phase end, and now")
    args = ap.parse_args()
    torch.set_num_threads(4)

    state = json.loads(Path(args.state).read_text())
    slices = time_slices(state, Path(args.agents_dir))
    if args.slice_every > 1:
        slices = [sl for k, sl in enumerate(slices)
                  if k == 0 or k == len(slices) - 1 or k % args.slice_every == 0]
    best = read_best_record()["path"]
    print(f"building {args.states}-state bank...", flush=True)
    bank = build_state_bank(args.states, args.seed)

    cache: dict[str, list[np.ndarray]] = {}

    def probs(path: str) -> list[np.ndarray]:
        if path not in cache:
            print(f"  scoring {path}", flush=True)
            cache[path] = policy_probs(path, bank)
        return cache[path]

    best_probs = probs(best)
    results = []
    for sl in slices:
        ps = [probs(path) for path in sl["checkpoints"]]
        pairs = {f"{i}-{j}": compare(ps[i], ps[j], bank) for i in range(4) for j in range(i + 1, 4)}
        games = np.array([e["game"] for e in bank])
        mean_js_state = np.mean([v["_js"] for v in pairs.values()], axis=0)
        mean_agree_state = np.mean([v["_agree"] for v in pairs.values()], axis=0)
        results.append({
            **sl,
            "pairs": {k: strip(v) for k, v in pairs.items()},
            "mean_js": float(np.mean([v["js"] for v in pairs.values()])),
            "mean_js_ci": game_ci(mean_js_state, games),
            "mean_agreement": float(np.mean([v["agreement"] for v in pairs.values()])),
            "mean_agreement_ci": game_ci(mean_agree_state, games),
            "vs_best": [strip(compare(p, best_probs, bank)) for p in ps],
            "style": [style(p, bank) for p in ps],
        })
        print(f"{sl['label']:>14}: mean pairwise JS {results[-1]['mean_js']:.4f}, "
              f"agreement {results[-1]['mean_agreement']:.1%}", flush=True)
    out = {"states": args.states, "seed": args.seed, "best": best, "best_style": style(best_probs, bank),
           "suit_pooled": True, "ci": "95% bootstrap over games",
           "slices": results}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
