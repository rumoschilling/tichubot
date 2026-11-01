"""Create an evaluation-only checkpoint with history inputs disabled.

This preserves the learned static-state columns and bias in every history
fusion layer. Only columns fed by the history encoder are zeroed. That
distinction matters because zeroing the complete fusion layer also removes a
learned static residual and is not a clean history ablation.

Usage:
    python -m training.ablate_history play /tmp/standard_no_history.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from tichu_env.encoding import PASS_SOURCE_NONE
from training.checkpoints import (
    detect_kind_from_checkpoint,
    load_policy_checkpoint,
    load_raw_checkpoint,
)


def _fusion_weight_keys(
    state: dict[str, torch.Tensor], hidden_dim: int
) -> list[str]:
    prefixes = ("history_fuse.", "critic_history_fuse.")
    keys = [
        key
        for key, value in state.items()
        if key.startswith(prefixes)
        and key.endswith(".weight")
        and value.ndim == 2
        and value.shape[0] == hidden_dim
        and value.shape[1] > hidden_dim
    ]
    if not any(key.startswith("history_fuse.") for key in keys):
        raise ValueError("Could not find the actor/shared history fusion layer")
    return sorted(keys)


def _verify_history_invariance(path: Path) -> None:
    loaded = load_policy_checkpoint(path)
    policy = loaded.policy
    if loaded.kind != "history":
        raise ValueError(f"Expected history policy, got {loaded.kind}")

    batch_size = 2
    history_steps = 8
    n_actions = 4
    obs_size = policy.trunk[0].in_features
    action_dim = policy.action_embed.in_features
    token_dim = policy.history_encoder.input_proj.in_features

    generator = torch.Generator().manual_seed(0)
    obs = torch.randn(batch_size, obs_size, generator=generator)
    action_vecs = torch.randn(
        batch_size, n_actions, action_dim, generator=generator
    )
    history_a = torch.randn(
        batch_size, history_steps, token_dim, generator=generator
    )
    history_b = torch.randn(
        batch_size, history_steps, token_dim, generator=generator
    )
    history_len = torch.tensor([history_steps, history_steps - 2])
    pass_ids = torch.full(
        (batch_size, 3), PASS_SOURCE_NONE, dtype=torch.int64
    )

    with torch.no_grad():
        logits_a, values_a = policy._forward_scores(
            obs,
            action_vecs,
            history_a,
            history_len,
            pass_ids,
            pass_ids,
        )
        logits_b, values_b = policy._forward_scores(
            obs,
            action_vecs,
            history_b,
            history_len,
            pass_ids,
            pass_ids,
        )
    if not torch.equal(logits_a, logits_b):
        change = float((logits_a - logits_b).abs().max())
        raise RuntimeError(f"History still changes action logits (max {change})")
    if not torch.equal(values_a, values_b):
        change = float((values_a - values_b).abs().max())
        raise RuntimeError(f"History still changes critic values (max {change})")


def create_history_ablation(
    source: str | Path,
    output: str | Path,
    *,
    verify_behavior: bool = True,
) -> dict[str, Any]:
    """Write a checkpoint whose fusion layers ignore history encoder output."""
    checkpoint, source_path = load_raw_checkpoint(source)
    if detect_kind_from_checkpoint(checkpoint) != "history":
        raise ValueError(f"Not a history checkpoint: {source_path}")

    state = checkpoint["model"]
    hidden_dim = int(state["value_head.weight"].shape[1])
    changed: list[dict[str, Any]] = []
    for key in _fusion_weight_keys(state, hidden_dim):
        weight = state[key]
        bias_key = f"{key[:-len('weight')]}bias"
        static_before = weight[:, :hidden_dim].clone()
        bias_before = state[bias_key].clone()
        history_norm = float(weight[:, hidden_dim:].norm())

        weight[:, hidden_dim:].zero_()

        if not torch.equal(weight[:, :hidden_dim], static_before):
            raise RuntimeError(f"Static fusion columns changed in {key}")
        if not torch.equal(state[bias_key], bias_before):
            raise RuntimeError(f"Fusion bias changed in {bias_key}")
        if torch.count_nonzero(weight[:, hidden_dim:]).item() != 0:
            raise RuntimeError(f"History fusion columns remain nonzero in {key}")
        changed.append(
            {
                "weight": key,
                "static_columns": [0, hidden_dim],
                "history_columns": [hidden_dim, weight.shape[1]],
                "history_norm_before": history_norm,
            }
        )

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["ablation"] = {
        "type": "history_input_zero",
        "evaluation_only": True,
        "source": str(source_path),
        "method": "zero history fusion input columns; preserve static columns and bias",
        "layers": changed,
    }
    torch.save(checkpoint, output_path)
    if verify_behavior:
        _verify_history_invariance(output_path)
    return checkpoint["ablation"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a clean no-history evaluation checkpoint"
    )
    parser.add_argument("source", help="History checkpoint path or alias")
    parser.add_argument("output", help="Destination .pt path")
    parser.add_argument(
        "--skip-behavior-check",
        action="store_true",
        help="Skip the post-save history-invariance forward-pass check",
    )
    args = parser.parse_args()

    metadata = create_history_ablation(
        args.source,
        args.output,
        verify_behavior=not args.skip_behavior_check,
    )
    print(f"Wrote clean history ablation: {Path(args.output).resolve()}")
    for layer in metadata["layers"]:
        print(
            f"  {layer['weight']}: zeroed columns "
            f"{layer['history_columns'][0]}:{layer['history_columns'][1]} "
            f"(prior norm {layer['history_norm_before']:.4f})"
        )
    if not args.skip_behavior_check:
        print("  verified: replacing history changes logits=0 and values=0")


if __name__ == "__main__":
    main()
