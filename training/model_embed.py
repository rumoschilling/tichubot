"""Action-embedding policy for Tichu PPO.

Architecture:
    obs (196) → [Linear→Tanh] × n_layers → feat (hidden_dim)
                                               ├─ value_head  → 1 scalar
                                               └─ query_head  → query (embed_dim)

    action (58-dim: 56 card bits + tichu bit + Phoenix-high bit)
        ├─ Linear(58, embed_dim) ───────────────────────────────┐
        └─ nonlinear residual MLP ──────────────────────────────┴→ key

    logit_i = dot(query, key_i)

Advantages over the fixed-table model (model.py):
  - Policy head is O(embed_dim) instead of O(MAX_ACTIONS=2000).
  - Computation scales with actual n_legal (~2-33 typical), not 2000.
  - Similar card combos get similar embeddings → generalises across positions.
  - Tiny card-provenance embeddings inject the received and sent pass cards
    without bloating the static observation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from tichu_env.encoding import STATIC_OBS_SIZE

ACTION_DIM = 58  # 56 card bits + tichu-call bit + Phoenix-high bit
LEGACY_ACTION_DIM = 57
PASS_SOURCE_VOCAB = 57  # 56 cards + 1 "unknown/not-set" sentinel
PASS_SOURCE_COUNT = 3
PASS_TARGET_COUNT = 3
_PASS_PROVENANCE_INIT_SEED = 0x51A7
_ACTION_ENCODER_PREFIX = "action_encoder."


class NonlinearActionEncoder(nn.Module):
    """Add combination-specific nonlinear features to the legacy action key.

    The output projection is zero-initialized. Consequently a fresh encoder
    initially returns the exact legacy linear key, which lets old checkpoints
    acquire these parameters without perturbing their policy. The hidden
    layers are initialized normally so the output projection receives useful,
    action-dependent gradients on the first optimizer step.
    """

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        action_hidden_dim: int = 128,
    ):
        super().__init__()
        if action_hidden_dim <= 0:
            raise ValueError("action_hidden_dim must be positive")

        self.residual = nn.Sequential(
            nn.Linear(action_dim, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, embed_dim),
        )
        self._init_weights()

    @staticmethod
    def _init_residual_mlp(module: nn.Sequential) -> None:
        linears = [layer for layer in module if isinstance(layer, nn.Linear)]
        for layer in linears[:-1]:
            nn.init.orthogonal_(layer.weight, gain=2**0.5)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(linears[-1].weight)
        nn.init.zeros_(linears[-1].bias)

    def _init_weights(self) -> None:
        self._init_residual_mlp(self.residual)

    def forward(
        self,
        action_vecs: torch.Tensor,
        linear_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Return legacy keys plus nonlinear, combination-specific features."""
        return linear_keys + self.residual(action_vecs)


def with_legacy_action_encoder_defaults(
    policy: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Fill encoder keys absent from pre-upgrade checkpoints.

    The returned tensors come from the newly constructed, behavior-neutral
    residual head.  Other missing or unexpected keys are deliberately left
    alone so ``strict=True`` retains its normal validation semantics.
    """
    current = policy.state_dict()
    upgraded = state_dict

    # The Phoenix-choice upgrade appended one input bit. Zero-padding keeps
    # every pre-upgrade action key and logit exactly unchanged.
    for key in ("action_embed.weight", "action_encoder.residual.0.weight"):
        if key not in upgraded or key not in current:
            continue
        old = upgraded[key]
        target = current[key]
        if (
            old.ndim == 2
            and old.shape[1] == LEGACY_ACTION_DIM
            and target.shape[1] == ACTION_DIM
            and old.shape[0] == target.shape[0]
        ):
            if upgraded is state_dict:
                upgraded = state_dict.copy()
                metadata = getattr(state_dict, "_metadata", None)
                if metadata is not None:
                    upgraded._metadata = metadata
            upgraded[key] = torch.cat(
                (old, old.new_zeros(old.shape[0], 1)), dim=1
            )

    missing = [
        key
        for key in current
        if key.startswith(_ACTION_ENCODER_PREFIX) and key not in upgraded
    ]
    if not missing:
        return upgraded

    if upgraded is state_dict:
        upgraded = state_dict.copy()
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            upgraded._metadata = metadata
    for key in missing:
        upgraded[key] = current[key].detach().clone()
    return upgraded


def initialize_pass_provenance_embedding_(embedding: nn.Embedding) -> None:
    """Initialize card identities without changing the global Torch RNG.

    The following projection remains zero-initialized, so these nonzero
    embeddings have no effect on fresh policy outputs. They do, however, let
    the projection receive card-specific gradients on the first update.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_PASS_PROVENANCE_INIT_SEED)
    values = torch.empty(
        tuple(embedding.weight.shape), dtype=torch.float32, device="cpu"
    )
    nn.init.orthogonal_(values, generator=generator)
    with torch.no_grad():
        embedding.weight.copy_(
            values.to(
                device=embedding.weight.device,
                dtype=embedding.weight.dtype,
            )
        )


class TichuPolicyEmbed(nn.Module):
    """Static Actor-Critic policy using action-embedding.
    
    This model processes the current game state through a shared MLP trunk 
    and produces four distinct outputs:
      1. Main Play (query_head): Generates a latent query used to rank legal
         card combinations by dotting the query with nonlinear action keys.
      2. Value (value_head): Estimating the expected return from the current 
         state (Critic).
      3. Dragon Give (dragon_head): Decides which opponent receives the trick
         when the Dragon is played.
      4. Mahjong Wish (wish_head): Chooses a rank to wish for (or no wish).
    """

    def __init__(
        self,
        obs_size: int = STATIC_OBS_SIZE,
        hidden_dim: int = 256,
        n_layers: int = 2,
        embed_dim: int = 64,
        action_dim: int = ACTION_DIM,
        pass_embed_dim: int = 8,
        action_hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_size = obs_size
        layers: list[nn.Module] = []
        in_dim = obs_size
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.pass_source_embed = nn.Embedding(PASS_SOURCE_VOCAB, pass_embed_dim)
        self.pass_source_proj = nn.Linear(
            PASS_SOURCE_COUNT * pass_embed_dim, hidden_dim
        )
        self.pass_target_embed = nn.Embedding(PASS_SOURCE_VOCAB, pass_embed_dim)
        self.pass_target_proj = nn.Linear(
            PASS_TARGET_COUNT * pass_embed_dim, hidden_dim
        )
        self.query_head = nn.Linear(hidden_dim, embed_dim)
        self.action_embed = nn.Linear(action_dim, embed_dim, bias=False)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Auxiliary heads for rare decisions
        self.dragon_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        self.wish_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 14)  # 0: no wish, 1-13: ranks 2-14
        )

        # Registered after every legacy module so old Adam parameter ordering
        # remains a prefix and can be upgraded without moving existing moments.
        self.action_encoder = NonlinearActionEncoder(
            action_dim=action_dim,
            embed_dim=embed_dim,
            action_hidden_dim=action_hidden_dim,
        )

        self._init_weights()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = with_legacy_action_encoder_defaults(self, state_dict)
        return super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )

    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        initialize_pass_provenance_embedding_(self.pass_source_embed)
        nn.init.zeros_(self.pass_source_proj.weight)
        nn.init.zeros_(self.pass_source_proj.bias)
        initialize_pass_provenance_embedding_(self.pass_target_embed)
        nn.init.zeros_(self.pass_target_proj.weight)
        nn.init.zeros_(self.pass_target_proj.bias)
        nn.init.orthogonal_(self.query_head.weight, gain=0.01)
        nn.init.zeros_(self.query_head.bias)
        nn.init.orthogonal_(self.action_embed.weight, gain=1.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _forward_scores(
        self,
        obs: torch.Tensor,  # (B, obs_size)
        action_vecs: torch.Tensor,  # (B, A, 58)
        pass_source_ids: torch.Tensor | None = None,  # (B, 3) int64
        pass_target_ids: torch.Tensor | None = None,  # (B, 3) int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw (unmasked) logits and value estimate.

        logits: (B, A)   value: (B,)
        """
        feat = self.trunk(obs)  # (B, H)
        if pass_source_ids is not None:
            pass_feat = self.pass_source_embed(pass_source_ids).reshape(
                obs.shape[0], -1
            )
            feat = feat + self.pass_source_proj(pass_feat)
        if pass_target_ids is not None:
            pass_feat = self.pass_target_embed(pass_target_ids).reshape(
                obs.shape[0], -1
            )
            feat = feat + self.pass_target_proj(pass_feat)
        query = self.query_head(feat)  # (B, E)
        linear_keys = self.action_embed(action_vecs)  # (B, A, E)
        keys = self.action_encoder(action_vecs, linear_keys)  # (B, A, E)
        logits = torch.bmm(
            query.unsqueeze(1), keys.transpose(1, 2)
        ).squeeze(1)  # (B, A)
        value = self.value_head(feat).squeeze(-1)  # (B,)
        return logits, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,  # (B, obs_size)
        action_vecs: torch.Tensor,  # (B, A, 58)  padded to max n_legal in batch
        n_legal: torch.Tensor,  # (B,) int32
        pass_source_ids: torch.Tensor | None = None,  # (B, 3) int64
        pass_target_ids: torch.Tensor | None = None,  # (B, 3) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one action per game. Returns (action_idx, log_prob, value), shape (B,).

        Uses Gumbel-max trick: argmax(logits + Gumbel) ~ Categorical(logits).
        """
        A = action_vecs.shape[1]
        valid = torch.arange(A, device=obs.device).unsqueeze(0) < n_legal.unsqueeze(1)
        logits, value = self._forward_scores(
            obs, action_vecs, pass_source_ids, pass_target_ids
        )
        logits = logits.masked_fill(~valid, float("-inf"))

        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-20)))
        action = (logits + gumbel).argmax(dim=-1)
        log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        log_prob = log_prob.gather(1, action.unsqueeze(1)).squeeze(1)
        return action, log_prob, value

    def get_dragon_gift_logits(self, fused_feat: torch.Tensor) -> torch.Tensor:
        """Predict which opponent to give the Dragon trick to.
        Returns (B, 2) logits for relative players 1 (next) and 3 (prev).
        """
        return self.dragon_head(fused_feat)

    def get_mahjong_wish_logits(self, fused_feat: torch.Tensor) -> torch.Tensor:
        """Predict which rank to wish for with Mahjong.
        Returns (B, 14) logits for no-wish (0) and ranks 2-14 (index i -> rank i+1 if i>0).
        """
        return self.wish_head(fused_feat)

    def evaluate(
        self,
        obs: torch.Tensor,  # (B, obs_size)
        action_vecs: torch.Tensor,  # (B, A, 58)
        n_legal: torch.Tensor,  # (B,) int32
        actions: torch.Tensor,  # (B,) int64 — index within legal list
        pass_source_ids: torch.Tensor | None = None,  # (B, 3) int64
        pass_target_ids: torch.Tensor | None = None,  # (B, 3) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions during PPO update.

        Returns (log_prob, value, entropy), shape (B,) each.
        """
        A = action_vecs.shape[1]
        valid = torch.arange(A, device=obs.device).unsqueeze(0) < n_legal.unsqueeze(1)
        logits, value = self._forward_scores(
            obs, action_vecs, pass_source_ids, pass_target_ids
        )
        logits = logits.masked_fill(~valid, float("-inf"))
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()
