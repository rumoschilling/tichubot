"""Saturation probe: network internals and policy determinism per checkpoint.

Every checkpoint is scored on the same fixed bank of decision states, pass
hands and Grand Tichu hands, so records are comparable across runs. One JSON
record per checkpoint (cached by file hash, metrics version and bank id):

  units     per nonlinearity (pre-activation over the bank): dead = positive
            on < 1% of inputs, always = positive on > 99%, flat = share of
            entries below -4 (SiLU/GELU gradient ~0); Tanh layers report the
            share of |tanh| > 0.99 as `saturated`
  erank     effective rank exp(H(normalized singular values)) and d99 (dims
            for 99% of variance) of the actor, critic and trunk features
  weights   total Frobenius norm, per top-level module, stable rank of the
            trunk linears
  history   scale of the history branch: valid-token norm after the encoder,
            pooled actor/critic vector norm (before any pool norm), static
            feature norm, history share of the fusion input's variance, and
            mean-ablation of each fusion branch: share of argmax moves that
            change, mean KL, mean |dV| in points vs the std of V
  policy    entropy (nats), share of decisions with max p > 0.99, H/log(n),
            logit std over legal moves; 95% intervals resample whole games
  pass, gt  trunk saturation, first-pick entropy, share of decided GT hands

Usage:
  python -m training.saturation bank  [--states 4000] [--seed 20261022] [--generator CKPT]
  python -m training.saturation probe LABEL=CKPT ... [--run-dir DIR --every 1000000 --prefix NAME]
  python -m training.saturation report [--records DIR] [--filter SUBSTR]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tichu_env.cards import NUM_CARDS
from tichu_env.encoding import (
    encode_history,
    encode_pass_obs,
    encode_pass_source_ids,
    encode_pass_target_ids,
)
from tichu_env.game_state import GameState

METRICS_VERSION = 1
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "local_evaluations" / "saturation"
CPU = torch.device("cpu")
VALUE_POINTS = 400.0  # value targets are score differences / 400
ACTIVATIONS = (nn.SiLU, nn.GELU, nn.ReLU, nn.Tanh)


# ── bank ─────────────────────────────────────────────────────────────────────


def bank_path(states: int, seed: int, root: Path = DEFAULT_DIR) -> Path:
    return root / f"bank_v1_n{states}_s{seed}.pkl"


def build_bank(states: int, seed: int, generator: str | None = None) -> dict:
    """Decision states from fixed-seed games, plus pass hands and GT hands.

    Decision states come from games played by ``generator`` (default: the
    current BEST_CHECKPOINT); the file, not the generator, defines the bank,
    so build it once and keep it.
    """
    from training.checkpoints import read_best_record
    from training.ring_diversity import build_state_bank

    generator = generator or read_best_record()["path"]
    items = []
    for entry in build_state_bank(states, seed, generator=generator):
        gs, player = entry["gs"], entry["player"]
        hist, hist_len = encode_history(gs, player)
        items.append({
            "gs": gs, "player": player, "vecs": entry["vecs"], "game": entry["game"],
            "hist": hist[: max(hist_len, 1)].astype(np.float32), "hist_len": hist_len,
            "src": encode_pass_source_ids(gs, player), "tgt": encode_pass_target_ids(gs, player),
        })
    pass_obs, pass_cards = [], []
    game = 0
    while len(pass_obs) < states:
        gs = GameState(track_history=False)
        gs.reset(seed=seed * 7919 + game, with_calling=True, with_passing=True)
        game += 1
        for p in range(4):
            gs.declare_grand_tichu(p, False)
        for p in range(4):
            pass_obs.append(encode_pass_obs(gs, p).astype(np.float32))
            pass_cards.append(np.array(sorted(gs.get_passing_hand(p)), dtype=np.int64))
    rng = np.random.default_rng(seed)
    gt = np.zeros((states, NUM_CARDS), dtype=np.float32)
    for row in gt:
        row[rng.choice(NUM_CARDS, 8, replace=False)] = 1.0
    return {
        "meta": {"version": 1, "states": states, "seed": seed, "generator": generator,
                 "games": len({it["game"] for it in items})},
        "items": items,
        "pass_obs": np.stack(pass_obs[:states]),
        "pass_cards": np.stack(pass_cards[:states]),
        "gt": gt,
    }


def load_bank(path: Path) -> dict:
    with open(path, "rb") as fh:
        bank = pickle.load(fh)
    bank["meta"]["id"] = f"{path.name}:{_sha256(path)[:12]}"
    return bank


# ── helpers ──────────────────────────────────────────────────────────────────


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def erank(features: np.ndarray) -> tuple[float, int]:
    """Effective rank and number of dims holding 99% of the variance."""
    x = features - features.mean(0)
    s = np.linalg.svd(x, compute_uv=False)
    if s.sum() == 0:
        return 0.0, 0
    p = s / s.sum()
    p = p[p > 0]
    var = np.cumsum(s**2) / (s**2).sum()
    return float(np.exp(-(p * np.log(p)).sum())), int(np.searchsorted(var, 0.99) + 1)


def unit_stats(pre: np.ndarray, tanh: bool = False) -> dict:
    if tanh:
        return {"units": int(pre.shape[1]), "saturated": float((np.abs(np.tanh(pre)) > 0.99).mean())}
    pos = (pre > 0).mean(0)
    return {"units": int(pre.shape[1]), "dead": float((pos < 0.01).mean()),
            "always": float((pos > 0.99).mean()), "flat": float((pre < -4).mean())}


def _mean_ci(values: np.ndarray, games: np.ndarray) -> dict:
    from training.ring_diversity import game_ci

    lo, hi = game_ci(values.astype(np.float64), games)
    return {"mean": float(values.mean()), "lo": lo, "hi": hi}


def _batches(items: list[dict], obs: np.ndarray, size: int = 128):
    from training.model_embed import ACTION_DIM

    for start in range(0, len(items), size):
        b = items[start : start + size]
        n = np.array([len(it["vecs"]) for it in b])
        acts = np.zeros((len(b), n.max(), ACTION_DIM), dtype=np.float32)
        for j, it in enumerate(b):
            acts[j, : n[j]] = it["vecs"]
        lens = np.array([it["hist_len"] for it in b], dtype=np.int64)
        max_h = max(int(lens.max()), 1)
        hist = np.zeros((len(b), max_h, b[0]["hist"].shape[1]), dtype=np.float32)
        for j, it in enumerate(b):
            hist[j, : len(it["hist"])] = it["hist"][:max_h]
        yield n, lens, max_h, (
            torch.as_tensor(obs[start : start + size]), torch.as_tensor(acts),
            torch.as_tensor(hist), torch.as_tensor(lens),
            torch.as_tensor(np.stack([it["src"] for it in b])),
            torch.as_tensor(np.stack([it["tgt"] for it in b])),
        )


def _forward_bank(policy, bank: dict, obs: np.ndarray, *, capture: bool = False,
                  replace: dict | None = None) -> dict:
    """One pass over the bank. Returns per-decision probs, values, and (when
    ``capture``) pre-activations, features and history-branch tensors.

    ``replace`` maps a fusion module name to a constant output (mean ablation).
    """
    items = bank["items"]
    rec: dict[str, list] = {}
    masks: dict[str, torch.Tensor] = {}
    hooks = []

    def keep(name, x):
        rec.setdefault(name, []).append(x.detach())

    if capture:
        for name, module in policy.named_modules():
            if isinstance(module, ACTIVATIONS):
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, name=name: keep(f"pre:{name}", i[0])))
        for li, layer in enumerate(policy.history_encoder.encoder.layers):
            hooks.append(layer.linear1.register_forward_hook(
                lambda m, i, o, li=li: keep(f"pre:history_encoder.encoder.layers.{li}.ffn", o)))
        hooks.append(policy.trunk.register_forward_hook(lambda m, i, o: keep("trunk_out", o)))
        hooks.append(policy.history_encoder.encoder.register_forward_hook(
            lambda m, i, o: keep("tokens", o)))
        hooks.append(policy.history_encoder.pool_attn.register_forward_hook(
            lambda m, i, o: keep("actor_pool", o[0].squeeze(1))))
        def fuse_hook(prefix):
            def fn(_m, inputs, output):  # must return None (a value replaces the output)
                keep(f"{prefix}_fuse_in", inputs[0])
                keep(f"{prefix}_delta", output)
            return fn

        hooks.append(policy.history_fuse.register_forward_hook(fuse_hook("actor")))
        if policy.split_actor_critic_history:
            hooks.append(policy.critic_pool_attn.register_forward_hook(
                lambda m, i, o: keep("critic_pool", o[0].squeeze(1))))
            hooks.append(policy.critic_history_fuse.register_forward_hook(fuse_hook("critic")))
        original = policy.actor_critic_features

        def features(*args, **kwargs):
            actor, critic = original(*args, **kwargs)
            keep("actor_feat", actor)
            keep("critic_feat", critic)
            return actor, critic

        policy.actor_critic_features = features
    for name, value in (replace or {}).items():
        module = policy.get_submodule(name)
        hooks.append(module.register_forward_hook(lambda m, i, o, v=value: v.expand_as(o)))

    probs, values, logit_std = [], [], []
    try:
        with torch.no_grad():
            for n, lens, max_h, args in _batches(items, obs):
                tok = torch.arange(max_h)[None, :] < args[3][:, None]  # valid tokens only
                act = torch.arange(int(n.max()))[None, :] < torch.as_tensor(n)[:, None]
                masks.setdefault("tok", []).append(tok)
                masks.setdefault("act", []).append(act)
                logits, value = policy._forward_scores(*args)
                values.append(value.numpy())
                for j, k in enumerate(n):
                    lg = logits[j, :k].double()
                    probs.append(torch.softmax(lg, 0).numpy())
                    logit_std.append(float(lg.std()))
    finally:
        for h in hooks:
            h.remove()
        if capture:
            del policy.actor_critic_features
    out = {"probs": probs, "values": np.concatenate(values), "logit_std": np.array(logit_std)}
    if capture:
        def rows(name, kind=None):
            parts = rec[name]
            if kind is None:
                return torch.cat([p.reshape(-1, p.shape[-1]) for p in parts]).numpy()
            return torch.cat([p[m] for p, m in zip(parts, masks[kind])]).numpy()

        pre = {}
        for name in rec:
            if not name.startswith("pre:"):
                continue
            module_name = name[4:]
            kind = "tok" if module_name.startswith("history_encoder") else (
                "act" if module_name.startswith("action_encoder") else None)
            if kind and rec[name][0].dim() != 3:
                kind = None
            pre[module_name] = rows(name, kind)
        out["pre"] = pre
        out["tanh"] = {n for n, m in policy.named_modules() if isinstance(m, nn.Tanh)}
        for name in ("trunk_out", "actor_feat", "critic_feat", "actor_pool", "critic_pool",
                     "actor_fuse_in", "critic_fuse_in", "actor_delta", "critic_delta"):
            if name in rec:
                out[name] = rows(name)
        out["tokens"] = rows("tokens", "tok")
    return out


def _history_share(fuse_in: np.ndarray, static_dim: int) -> float:
    centered = fuse_in - fuse_in.mean(1, keepdims=True)
    total = (centered**2).sum(1)
    return float(np.mean((centered[:, static_dim:] ** 2).sum(1) / np.maximum(total, 1e-12)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    return float((p * (np.log(np.clip(p, 1e-300, None)) - np.log(np.clip(q, 1e-300, None)))).sum())


# ── probe ────────────────────────────────────────────────────────────────────


def probe(path: str, bank: dict, label: str | None = None) -> dict:
    from training.checkpoints import history_norm_flags, is_legacy_history, tag_legacy_label
    from training.observations import encode_policy_static
    from training.play import load_play_bundle

    raw = torch.load(path, map_location="cpu", weights_only=False)
    policy, pass_policy, gt_policy, kind, _ = load_play_bundle(path, None, CPU)
    if kind != "history":
        raise ValueError(f"{path}: saturation probe supports history checkpoints, got {kind}")
    # train(): dropout is 0, and it keeps the transformer off its fused fast
    # path so the FFN hooks fire; no batch norm anywhere.
    policy.train()
    items = bank["items"]
    games = np.array([it["game"] for it in items])
    obs = np.stack([encode_policy_static(policy, it["gs"], it["player"]) for it in items]).astype(np.float32)

    full = _forward_bank(policy, bank, obs, capture=True)
    rec: dict = {
        "version": METRICS_VERSION,
        "label": tag_legacy_label(label or Path(path).stem, raw),
        "path": str(Path(path).resolve().relative_to(ROOT)) if Path(path).resolve().is_relative_to(ROOT) else str(path),
        "sha256": _sha256(path),
        "games": int(raw.get("games_played") or 0),
        "legacy_history": is_legacy_history(raw),
        "arch": dict(zip(("final_norm", "pool_norm"), history_norm_flags(raw["model"]))),
        "bank": bank["meta"]["id"],
        "n_states": len(items),
    }
    rec["units"] = {name: unit_stats(pre, tanh=name in full["tanh"]) for name, pre in full["pre"].items()}
    rec["erank"] = {}
    for key, name in (("actor", "actor_feat"), ("critic", "critic_feat"), ("trunk", "trunk_out")):
        er, d99 = erank(full[name])
        rec["erank"][key] = {"erank": er, "d99": d99}

    params = dict(policy.named_parameters())
    groups: dict[str, float] = {}
    for name, p in params.items():
        groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0.0) + float((p.detach() ** 2).sum())
    stable = {}
    for name, module in policy.trunk.named_modules():
        if isinstance(module, nn.Linear):
            s = np.linalg.svd(module.weight.detach().numpy(), compute_uv=False)
            stable[f"trunk.{name}"] = float((s**2).sum() / s[0] ** 2)
    rec["weights"] = {"total": math.sqrt(sum(groups.values())),
                      "groups": {k: math.sqrt(v) for k, v in sorted(groups.items())},
                      "stable_rank": stable}

    static_dim = full["trunk_out"].shape[1]
    hist = {
        "token_norm": float(np.median(np.linalg.norm(full["tokens"], axis=1))),
        "static_norm": float(np.median(np.linalg.norm(full["actor_fuse_in"][:, :static_dim], axis=1))),
        "actor_pool_norm": float(np.median(np.linalg.norm(full["actor_pool"], axis=1))),
        "actor_history_share": _history_share(full["actor_fuse_in"], static_dim),
    }
    ablate = {"history_fuse": torch.as_tensor(full["actor_delta"].mean(0))}
    p_abl = _forward_bank(policy, bank, obs, replace=ablate)["probs"]
    hist["actor_argmax_flip"] = float(np.mean([p.argmax() != q.argmax() for p, q in zip(full["probs"], p_abl)]))
    hist["actor_kl"] = float(np.mean([_kl(p, q) for p, q in zip(full["probs"], p_abl)]))
    if policy.split_actor_critic_history:
        hist["critic_pool_norm"] = float(np.median(np.linalg.norm(full["critic_pool"], axis=1)))
        hist["critic_history_share"] = _history_share(full["critic_fuse_in"], static_dim)
        ablate = {"critic_history_fuse": torch.as_tensor(full["critic_delta"].mean(0))}
        v_abl = _forward_bank(policy, bank, obs, replace=ablate)["values"]
        hist["critic_abs_dv_pts"] = float(VALUE_POINTS * np.abs(full["values"] - v_abl).mean())
        hist["critic_v_std_pts"] = float(VALUE_POINTS * full["values"].std())
    rec["history"] = hist

    ent = np.array([-(p[p > 0] * np.log(p[p > 0])).sum() for p in full["probs"]])
    n_legal = np.array([len(p) for p in full["probs"]])
    pmax = np.array([p.max() for p in full["probs"]])
    rec["policy"] = {
        "entropy": _mean_ci(ent, games),
        "det99": _mean_ci((pmax > 0.99).astype(np.float64), games),
        "norm_entropy": _mean_ci(ent / np.log(n_legal), games),
        "logit_std": _mean_ci(full["logit_std"], games),
    }

    po = torch.as_tensor(bank["pass_obs"])
    with torch.no_grad():
        tf = pass_policy.trunk(po)
        mask = torch.ones(len(po), bank["pass_cards"].shape[1], dtype=torch.bool)
        lg = pass_policy._pointer_logits(tf, 0, torch.zeros(len(po), NUM_CARDS),
                                         torch.as_tensor(bank["pass_cards"]), mask)
        pp = torch.softmax(lg.double(), -1)
        pent = -(pp * torch.log(pp.clamp_min(1e-300))).sum(-1)
    rec["pass"] = {"tanh_saturated": float((tf.abs() > 0.99).float().mean()),
                   "pick_entropy": float(pent.mean()),
                   "pick_det99": float((pp.max(-1).values > 0.99).double().mean()),
                   "erank": erank(tf.numpy())[0]}
    gh = torch.as_tensor(bank["gt"])
    with torch.no_grad():
        gtf = gt_policy.trunk(gh)
        gp = torch.sigmoid(gt_policy.call_logits(gh).double().reshape(-1))
    rec["gt"] = {"tanh_saturated": float((gtf.abs() > 0.99).float().mean()),
                 "decided": float(((gp < 0.01) | (gp > 0.99)).double().mean()),
                 "mean_p": float(gp.mean()), "erank": erank(gtf.numpy())[0]}
    return rec


def record_path(records: Path, label: str) -> Path:
    return records / f"{re.sub(r'[^A-Za-z0-9_.+-]', '_', label)}.json"


def probe_cached(path: str, label: str, bank: dict, records: Path) -> tuple[dict, bool]:
    """Probe unless a record for this file, metrics version and bank exists."""
    from training.checkpoints import tag_legacy_label

    target = record_path(records, tag_legacy_label(label, torch.load(path, map_location="cpu", weights_only=False)))
    if target.exists():
        old = json.loads(target.read_text())
        if (old.get("version"), old.get("bank"), old.get("sha256")) == (
            METRICS_VERSION, bank["meta"]["id"], _sha256(path)):
            return old, False
    rec = probe(path, bank, label)
    records.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    tmp.replace(target)
    return rec, True


def run_dir_checkpoints(run_dir: str, every: int) -> list[tuple[int, str]]:
    """(games, path) of ckpt_*.pt files, the first at or after each multiple of ``every``."""
    found = []
    for f in glob.glob(os.path.join(run_dir, "ckpt_*.pt")):
        m = re.search(r"ckpt_(\d+)\.pt$", f)
        if m:
            found.append((int(m.group(1)), f))
    found.sort()
    picked, next_mark = [], None
    for games, f in found:
        mark = games // every
        if next_mark is None or mark >= next_mark:
            picked.append((games, f))
            next_mark = mark + 1
    return picked


# ── report ───────────────────────────────────────────────────────────────────


def _pct(x: float | None) -> str:
    return "   -" if x is None else f"{100 * x:4.0f}"


def report(records: list[dict]) -> str:
    records = sorted(records, key=lambda r: (re.sub(r"_?\d+(-noLN)?$", "", r["label"]), r["games"]))
    lines = []
    layer_names = sorted({n for r in records for n in r["units"]})
    groups = {
        "trunk": [n for n in layer_names if n.startswith("trunk")],
        "hist ffn": [n for n in layer_names if n.startswith("history_encoder")],
        "actor fuse": [n for n in layer_names if n.startswith("history_fuse")],
        "critic fuse": [n for n in layer_names if n.startswith("critic_history_fuse")],
        "action enc": [n for n in layer_names if n.startswith("action_encoder")],
    }

    def dead(r, names):
        vals = [r["units"][n]["dead"] for n in names if n in r["units"] and "dead" in r["units"][n]]
        return max(vals) if vals else None

    lines.append("INTERNALS  dead% = units positive on <1% of the bank (worst layer of each group)")
    head = f"{'label':28s} {'games':>10s} " + " ".join(f"{g:>11s}" for g in groups)
    lines.append(head + "  eR act/crt/trk   |W|")
    for r in records:
        er = r["erank"]
        lines.append(f"{r['label']:28s} {r['games']:10,d} " + " ".join(f"{_pct(dead(r, n)):>11s}" for n in groups.values())
                     + f"  {er['actor']['erank']:5.1f}/{er['critic']['erank']:5.1f}/{er['trunk']['erank']:5.1f}"
                     f" {r['weights']['total']:6.0f}")
    lines.append("")
    lines.append("HISTORY BRANCH  norms are medians; share = history's share of the fusion input variance;"
                 " flip = moves changed by mean-ablating the actor branch")
    lines.append(f"{'label':28s} {'games':>10s} {'token':>7s} {'static':>7s} {'a_pool':>8s} {'c_pool':>9s}"
                 f" {'a_share':>7s} {'c_share':>7s} {'flip%':>6s} {'KL':>7s} {'|dV| pts':>8s} {'sd V':>6s}")
    for r in records:
        h = r["history"]
        lines.append(
            f"{r['label']:28s} {r['games']:10,d} {h['token_norm']:7.1f} {h['static_norm']:7.1f}"
            f" {h['actor_pool_norm']:8.1f} {h.get('critic_pool_norm', float('nan')):9.1f}"
            f" {h['actor_history_share']:7.3f} {h.get('critic_history_share', float('nan')):7.3f}"
            f" {100 * h['actor_argmax_flip']:6.1f} {h['actor_kl']:7.4f}"
            f" {h.get('critic_abs_dv_pts', float('nan')):8.1f} {h.get('critic_v_std_pts', float('nan')):6.1f}")
    lines.append("")
    lines.append("POLICY  [95% CI over games]")
    lines.append(f"{'label':28s} {'games':>10s} {'entropy':>22s} {'max p>.99 %':>20s} {'H/log n':>7s}"
                 f" | {'pass H':>6s} {'pass tanh%':>10s} | {'GT decided%':>11s}")
    for r in records:
        p = r["policy"]
        e, d = p["entropy"], p["det99"]
        lines.append(
            f"{r['label']:28s} {r['games']:10,d} {e['mean']:.3f} [{e['lo']:.3f},{e['hi']:.3f}]"
            f" {100 * d['mean']:5.1f} [{100 * d['lo']:4.1f},{100 * d['hi']:4.1f}]"
            f" {p['norm_entropy']['mean']:7.3f} | {r['pass']['pick_entropy']:6.3f}"
            f" {100 * r['pass']['tanh_saturated']:10.2f} | {100 * r['gt']['decided']:11.1f}")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank", help="build the fixed state bank (once)")
    b.add_argument("--states", type=int, default=4000)
    b.add_argument("--seed", type=int, default=20261022)
    b.add_argument("--generator", default=None, help="checkpoint that plays the bank games (default BEST)")
    b.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    p = sub.add_parser("probe", help="probe checkpoints (cached per file/version/bank)")
    p.add_argument("checkpoints", nargs="*", help="LABEL=PATH or PATH")
    p.add_argument("--run-dir", default=None, help="probe ckpt_*.pt of a run, one per --every games")
    p.add_argument("--every", type=int, default=1_000_000)
    p.add_argument("--prefix", default=None, help="label prefix for --run-dir (default: dir name)")
    p.add_argument("--bank", type=Path, default=None, help=f"default {bank_path(4000, 20261022)}")
    p.add_argument("--records", type=Path, default=DEFAULT_DIR / "records")
    p.add_argument("--threads", type=int, default=4)
    r = sub.add_parser("report", help="tabulate records")
    r.add_argument("--records", type=Path, default=DEFAULT_DIR / "records")
    r.add_argument("--filter", default=None, help="only labels containing this")
    args = ap.parse_args()

    if args.cmd == "bank":
        target = bank_path(args.states, args.seed, args.dir)
        if target.exists():
            raise SystemExit(f"{target} exists; the bank defines comparability, delete it deliberately")
        bank = build_bank(args.states, args.seed, args.generator)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            pickle.dump(bank, fh)
        print(f"wrote {target}: {len(bank['items'])} states from {bank['meta']['games']} games "
              f"(generator {bank['meta']['generator']})")
    elif args.cmd == "probe":
        torch.set_num_threads(args.threads)
        bank = load_bank(args.bank or bank_path(4000, 20261022))
        specs = []
        for spec in args.checkpoints:
            label, sep, path = spec.partition("=")
            specs.append((label, path) if sep else (Path(spec).stem, spec))
        if args.run_dir:
            prefix = args.prefix or Path(args.run_dir.rstrip("/")).parent.name
            specs += [(f"{prefix}_{games:08d}", f) for games, f in run_dir_checkpoints(args.run_dir, args.every)]
        for label, path in specs:
            rec, fresh = probe_cached(path, label, bank, args.records)
            h, pol = rec["history"], rec["policy"]
            print(f"{'probed' if fresh else 'cached'} {rec['label']}: flip {100 * h['actor_argmax_flip']:.1f}%"
                  f"  token {h['token_norm']:.1f}  H {pol['entropy']['mean']:.3f}", flush=True)
    else:
        recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(args.records / "*.json")))]
        if args.filter:
            recs = [x for x in recs if args.filter in x["label"]]
        print(report(recs) if recs else "no records")


if __name__ == "__main__":
    main()
