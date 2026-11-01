"""Passing network for Tichu — sequential pointer over 14-card hand.

Architecture:
    pass obs (60-dim: 56 hand binary + 4 relative grand-tichu bits,
    see tichu_env.encoding.encode_pass_obs) → MLP trunk → hidden state
    3 autoregressive pointer steps:
      step 1: score 14 cards → pick card for left     (14 options)
      step 2: mask picked   → pick card for partner   (13 options)
      step 3: mask picked   → pick card for right     (12 options)

    Each step:
      context = trunk_feat + step_embed + picked_proj(picked_cards_binary)
      card_key_i = Embedding(actual_card_id_i)
      logit_i = legacy_position_logit_i + dot(card_query(context), card_key_i)

The candidate embedding branch fixes the original scorer's main
representational weakness: an output index referred to an ordinal position
in the sorted hand rather than a stable card identity. The positional branch
is retained as a residual so legacy checkpoints migrate with exactly the same
policy; the new query is zero-initialized and starts learning card-aware
corrections on the next update.

Training signal (since 2026-07-25): realized team round returns of the
passer, harvested from the main self-play rollouts (rollout_history.py),
with advantage = return − baseline(pass_obs). The original own-view
V(post-pass) − V(pre-pass) proxy was exactly invariant to the
left/partner/right assignment — direction trained on pure noise. The
bootstrap pipeline (rollout_passing.py, main-loop retired) has since been
upgraded to a team-signed all-hands exchange score, so it is
direction-aware too.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

NUM_CARDS = 56
GT_DECL_DIM = 4  # relative grand-tichu declaration bits ([0] = self)
PASS_OBS_DIM = NUM_CARDS + GT_DECL_DIM  # = tichu_env.encoding.PASS_OBS_SIZE
HAND_SIZE = 14
NUM_STEPS = 3  # left, partner, right
CARD_EMBED_DIM = 32


class TichuPassingPolicy(nn.Module):
    """Sequential pointer network for card passing.

    Parameters
    ----------
    hidden_dim : trunk width (default 128)
    n_layers   : trunk depth (default 2)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 2,
        card_embed_dim: int = CARD_EMBED_DIM,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.card_embed_dim = card_embed_dim

        # Trunk: hand binary + grand-tichu declaration bits → hidden
        layers: list[nn.Module] = []
        in_dim = PASS_OBS_DIM
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Step embedding: which recipient (0=left, 1=partner, 2=right)
        self.step_embed = nn.Embedding(NUM_STEPS, hidden_dim)

        # Project already-picked cards (56-dim binary) into context space
        self.picked_proj = nn.Linear(NUM_CARDS, hidden_dim, bias=False)

        # Score each of the 14 hand positions from context
        self.pointer_head = nn.Linear(hidden_dim, HAND_SIZE)

        # Decision-blind REINFORCE baseline b(pass_obs) for the
        # realized-return signal: absorbs hand quality (and declaration
        # context), leaving the pass decision's causal contribution.
        self.baseline_head = nn.Linear(hidden_dim, 1)

        # Candidate-aware residual scorer. These modules deliberately follow
        # baseline_head in registration order so old Adam parameter indices
        # remain valid when the new parameter slots are appended on resume.
        self.card_embed = nn.Embedding(NUM_CARDS, card_embed_dim)
        self.card_query = nn.Linear(hidden_dim, card_embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Declaration bits start with zero influence: a fresh net behaves as
        # if they didn't exist until training assigns them meaning.
        with torch.no_grad():
            self.trunk[0].weight[:, NUM_CARDS:].zero_()
        nn.init.orthogonal_(self.pointer_head.weight, gain=0.01)
        nn.init.zeros_(self.pointer_head.bias)
        nn.init.zeros_(self.picked_proj.weight)  # start neutral
        nn.init.normal_(self.step_embed.weight, std=0.01)
        nn.init.zeros_(self.baseline_head.weight)
        nn.init.zeros_(self.baseline_head.bias)
        nn.init.orthogonal_(self.card_embed.weight, gain=1.0)
        # A zero query makes the embedding branch an exact no-op initially,
        # preserving both fresh-policy scale and legacy checkpoint behavior.
        nn.init.zeros_(self.card_query.weight)
        nn.init.zeros_(self.card_query.bias)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Migrate legacy positional passing checkpoints without behavior drift.

        Declaration columns and the candidate query both start at zero
        influence, so migrated models behave exactly like their source until
        subsequent training uses the new inputs and card embeddings.
        """
        migrated_card_scorer = "card_embed.weight" not in state_dict
        w = state_dict.get("trunk.0.weight")
        if (
            w is not None
            and (
                w.shape[1] == NUM_CARDS
                or "baseline_head.weight" not in state_dict
                or migrated_card_scorer
            )
        ):
            state_dict = dict(state_dict)
        if w is not None and w.shape[1] == NUM_CARDS:
            state_dict["trunk.0.weight"] = torch.cat(
                [w, w.new_zeros(w.shape[0], GT_DECL_DIM)], dim=1)
            print(
                f"  [compat] Zero-padded passing-net input {NUM_CARDS} -> "
                f"{PASS_OBS_DIM} (grand tichu declaration bits)",
                flush=True,
            )
        if "baseline_head.weight" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["baseline_head.weight"] = self.baseline_head.weight.detach().clone()
            state_dict["baseline_head.bias"] = self.baseline_head.bias.detach().clone()
            print("  [compat] Passing net without baseline head: baseline starts fresh",
                  flush=True)
        if migrated_card_scorer:
            state_dict["card_embed.weight"] = (
                self.card_embed.weight.detach().clone()
            )
            state_dict["card_query.weight"] = (
                self.card_query.weight.detach().clone()
            )
            state_dict["card_query.bias"] = (
                self.card_query.bias.detach().clone()
            )
            print(
                "  [compat] Added zero-influence candidate card embeddings "
                "to passing net",
                flush=True,
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def baseline(self, hand_binary: torch.Tensor) -> torch.Tensor:
        """Decision-blind expected return b(pass_obs), shape (B,)."""
        return self.baseline_head(self.trunk(hand_binary)).squeeze(-1)

    def _pointer_logits(
        self,
        trunk_feat: torch.Tensor,  # (B, H)
        step: int,
        picked_binary: torch.Tensor,  # (B, 56) binary of already-picked cards
        hand_cards: torch.Tensor,  # (B, 14) actual sorted card ids
        mask: torch.Tensor,  # (B, 14) bool — True = available
    ) -> torch.Tensor:
        ctx = trunk_feat + self.step_embed.weight[step] + self.picked_proj(picked_binary)
        position_logits = self.pointer_head(ctx)  # (B, 14)
        query = self.card_query(ctx)  # (B, E)
        card_keys = self.card_embed(hand_cards)  # (B, 14, E)
        card_logits = torch.bmm(
            card_keys, query.unsqueeze(-1)
        ).squeeze(-1)  # (B, 14)
        logits = position_logits + card_logits
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    @torch.no_grad()
    def act(
        self,
        hand_binary: torch.Tensor,  # (B, 60) pass obs — see encode_pass_obs
        hand_cards: torch.Tensor,    # (B, 14) int — sorted card indices
        noise: torch.Tensor | None = None,  # (B, 3, 14) uniforms; None = fresh
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample 3 cards to pass. Returns (choices, log_probs).

        ``noise`` supplies the Gumbel-max uniforms per step (common random
        numbers); each step is still an exact draw from the policy.

        choices:   (B, 3) int — indices into the 14-card hand
        log_probs: (B, 3) float — log probability of each choice
        """
        B = hand_binary.shape[0]
        trunk_feat = self.trunk(hand_binary)  # (B, H)

        mask = torch.ones(B, HAND_SIZE, dtype=torch.bool, device=hand_binary.device)
        picked_binary = torch.zeros(B, NUM_CARDS, device=hand_binary.device)

        choices = torch.zeros(B, NUM_STEPS, dtype=torch.long, device=hand_binary.device)
        log_probs = torch.zeros(B, NUM_STEPS, device=hand_binary.device)

        for step in range(NUM_STEPS):
            logits = self._pointer_logits(
                trunk_feat, step, picked_binary, hand_cards, mask
            )

            # Gumbel-max sampling
            uniforms = torch.rand_like(logits) if noise is None else noise[:, step].to(logits)
            gumbel = -torch.log(-torch.log(uniforms.clamp(1e-20, 1.0 - 1e-7)))
            choice = (logits + gumbel).argmax(dim=-1)  # (B,)

            lp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            log_probs[:, step] = lp.gather(1, choice.unsqueeze(1)).squeeze(1)
            choices[:, step] = choice

            # Update mask and picked card binary
            mask.scatter_(1, choice.unsqueeze(1), False)
            # Convert hand position to card index and set in picked_binary
            card_idx = hand_cards[torch.arange(B, device=hand_binary.device), choice]
            picked_binary.scatter_(1, card_idx.unsqueeze(1), 1.0)

        return choices, log_probs

    def evaluate(
        self,
        hand_binary: torch.Tensor,  # (B, 60) pass obs — see encode_pass_obs
        hand_cards: torch.Tensor,   # (B, 14) int
        choices: torch.Tensor,      # (B, 3) int — indices into hand
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate stored choices during PPO update.

        Returns (log_probs, entropy) each (B, 3).
        """
        B = hand_binary.shape[0]
        trunk_feat = self.trunk(hand_binary)  # (B, H)

        mask = torch.ones(B, HAND_SIZE, dtype=torch.bool, device=hand_binary.device)
        picked_binary = torch.zeros(B, NUM_CARDS, device=hand_binary.device)

        log_probs = torch.zeros(B, NUM_STEPS, device=hand_binary.device)
        entropies = torch.zeros(B, NUM_STEPS, device=hand_binary.device)

        for step in range(NUM_STEPS):
            logits = self._pointer_logits(
                trunk_feat, step, picked_binary, hand_cards, mask
            )
            dist = Categorical(logits=logits)

            choice = choices[:, step]
            log_probs[:, step] = dist.log_prob(choice)
            entropies[:, step] = dist.entropy()

            mask = mask.scatter(1, choice.unsqueeze(1), False)
            card_idx = hand_cards[torch.arange(B, device=hand_binary.device), choice]
            one_hot = torch.zeros_like(picked_binary)
            one_hot.scatter_(1, card_idx.unsqueeze(1), 1.0)
            picked_binary = picked_binary + one_hot

        return log_probs, entropies
