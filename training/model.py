"""Policy + value network for Tichu PPO.

All four players share one network. Observations are encoded in the
acting player's relative frame (self=0, next=1, partner=2, prev=3),
so parameter sharing across players is valid.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from tichu_env.encoding import STATIC_OBS_SIZE
from tichu_env.constants import MAX_ACTIONS


class TichuPolicy(nn.Module):
    """Two-headed MLP: shared trunk → policy logits + scalar value.

    Architecture:
        obs (196) → [Linear→Tanh] × n_layers → trunk (hidden_dim)
                                                    ├─ policy head → 2000 logits (masked)
                                                    └─ value head  → 1 scalar
    """

    def __init__(
        self,
        obs_size: int = STATIC_OBS_SIZE,
        n_actions: int = MAX_ACTIONS,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_size
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Small init on policy head so early logits are near-uniform
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        obs:  (B, 196)         float32
        mask: (B, 2000)        bool — True where action is legal
        Returns: logits (B, 2000), value (B,)
        """
        feat = self.trunk(obs)
        logits = self.policy_head(feat).masked_fill(~mask, float("-inf"))
        value = self.value_head(feat).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one action during rollout collection.
        Returns (action, log_prob, value) — shape (B,) each.

        Uses the Gumbel-max trick instead of torch.multinomial:
        adding Gumbel(0,1) noise to logits and taking argmax is
        equivalent to categorical sampling but ~3x faster on CPU.
        """
        logits, value = self(obs, mask)
        # Gumbel noise: -log(-log(U)), U ~ Uniform(0,1)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-20)))
        action = (logits + gumbel).argmax(dim=-1)
        # log_prob = log_softmax at chosen action
        log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        log_prob = log_prob.gather(1, action.unsqueeze(1)).squeeze(1)
        return action, log_prob, value

    def evaluate(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions during PPO update.
        Returns (log_prob, value, entropy) — shape (B,) each.
        """
        logits, value = self(obs, mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()
