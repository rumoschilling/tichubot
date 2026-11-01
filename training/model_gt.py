"""Grand Tichu decision network.

Input:  8 cards (56-dim binary)
Output: Decision to call Grand Tichu (binary), derived from a regressed
        declare advantage.

Since 2026-07-25 the calling policy is a proxy derived from regression:
adv_head fits d(h8) ≈ E[declared − quiet | h8] on the paired probe data
(gt_probe_update), and the call probability is p = σ(d / GT_DERIVE_TEMP).
That is the same fixed point p = σ(E[δ]/entropy_coef) the old
policy-gradient probe objective converged to — but reached by regression,
whose gradient does not vanish when the policy saturates. The old PG form
δ·p had gradient scaled by p(1−p): once the head honestly said "never"
(p≈0), it could barely learn which hands were exceptions.
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

NUM_CARDS = 56

# Softness of the derived policy: p = σ(d / GT_DERIVE_TEMP). 0.01 in
# normalized return units ≈ 4 pts/round — matches the entropy_coef the old
# probe objective used, so the derived policy is its exact fixed point.
GT_DERIVE_TEMP = 0.01

# Fresh adv_head bias: ≈ −100 pts/round, the measured mean cost of a forced
# sole declare under current play strength. A fresh or migrated model starts
# at "never call" (honest) and the regression raises d where hands earn it.
GT_ADV_PRIOR = -0.25


class TichuGrandTichuPolicy(nn.Module):
    """Grand Tichu declaration policy derived from a declare-advantage head.

    Parameters
    ----------
    hidden_dim : trunk width (default 64)
    n_layers   : trunk depth (default 2)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = NUM_CARDS
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        # d(h8) ≈ E[return(declared) − return(quiet) | h8], normalized units.
        self.adv_head = nn.Linear(hidden_dim, 1)
        # Baseline b(h8): expected quiet-branch return for hands like this.
        # Kept decision-blind; used by calibration probes and as REINFORCE
        # baseline in the (legacy, non-probe) realized-return path.
        self.baseline_head = nn.Linear(hidden_dim, 1)

        # Set when load_state_dict consumed a legacy classifier head; the
        # trainer uses it to skip stale GT optimizer state (the old head's
        # Adam moments would silently land on the fresh adv_head — the two
        # heads have identical shapes).
        self.migrated_from_classifier = False

        self._init_weights()

    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.adv_head.weight)
        nn.init.constant_(self.adv_head.bias, GT_ADV_PRIOR)
        nn.init.zeros_(self.baseline_head.weight)
        nn.init.zeros_(self.baseline_head.bias)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Accept legacy checkpoints across both head generations:
        pre-baseline ones get a fresh (zero) baseline, and classifier-head
        ones ("head.*") get a fresh adv_head — the classifier's logits lived
        in a different parametrization and are dropped; the derived policy
        starts at the conservative "never call" prior."""
        if "baseline_head.weight" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["baseline_head.weight"] = self.baseline_head.weight.detach().clone()
            state_dict["baseline_head.bias"] = self.baseline_head.bias.detach().clone()
            print("  [compat] GT model without baseline head: baseline starts fresh",
                  flush=True)
        if "head.weight" in state_dict:
            state_dict = dict(state_dict)
            del state_dict["head.weight"]
            del state_dict["head.bias"]
            print("  [compat] GT classifier head dropped: calling policy now "
                  "derived from the declare-advantage regression (starts at "
                  "the never-call prior)", flush=True)
            self.migrated_from_classifier = True
        if "adv_head.weight" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["adv_head.weight"] = self.adv_head.weight.detach().clone()
            state_dict["adv_head.bias"] = self.adv_head.bias.detach().clone()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def baseline(self, hand_8_binary: torch.Tensor) -> torch.Tensor:
        """b(h8) — expected quiet return of the hand, decision-blind. (B,)"""
        return self.baseline_head(self.trunk(hand_8_binary)).squeeze(-1)

    def declare_advantage(self, hand_8_binary: torch.Tensor) -> torch.Tensor:
        """d(h8) — regressed declare advantage in normalized return units. (B,)"""
        return self.adv_head(self.trunk(hand_8_binary)).squeeze(-1)

    def call_logits(self, hand_8_binary: torch.Tensor) -> torch.Tensor:
        """Derived policy logits: d(h8) / GT_DERIVE_TEMP. (B,)"""
        return self.declare_advantage(hand_8_binary) / GT_DERIVE_TEMP

    @torch.no_grad()
    def act(
        self,
        hand_8_binary: torch.Tensor,  # (B, 56) binary of the first 8 cards
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample GT decision. Returns (declare, log_probs).

        declare:   (B,) bool/long — 1 if declare GT, 0 otherwise
        log_probs: (B,) float — log probability of the choice
        """
        logits = self.call_logits(hand_8_binary)  # (B,)
        probs  = torch.sigmoid(logits)
        dist   = Bernoulli(probs=probs)
        choice = dist.sample()  # (B,)
        return choice.long(), dist.log_prob(choice)

    def evaluate(
        self,
        hand_8_binary: torch.Tensor,  # (B, 56)
        declare: torch.Tensor,        # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate stored choices during PPO update.

        Returns (log_probs, entropy) each (B,).
        """
        logits  = self.call_logits(hand_8_binary)  # (B,)
        dist    = Bernoulli(logits=logits)
        return dist.log_prob(declare.float()), dist.entropy()
