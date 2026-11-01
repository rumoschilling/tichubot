"""Policy with static MLP + pluggable history encoder for Tichu PPO.

Architecture:
    static (196) → MLP trunk → static_feat (hidden_dim)
    history (T, 63) → HistoryEncoder → hist_feat (encoder.output_dim)
    [static_feat ; hist_feat] → history_fuse (Linear + Tanh) → delta (hidden_dim)
    feat = static_feat + delta   (residual skip)
                          ├─ value_head → 1
                          └─ query_head → query (embed_dim)
    action (58) → linear key + nonlinear residual key
    logit_i = dot(query, key_i)

The history encoder is pluggable — swap TransformerHistoryEncoder for any
nn.Module with an `output_dim` int attribute. The default transformer encoder
uses learned attention pooling over the sequence rather than mean pooling, so
it can emphasize the most decision-relevant past actions. With static attention
enabled, the learned pooling query is augmented by a projection of the current
static state, allowing the same history to be read differently for each hand
and decision context. Static and history features are then fused by
concatenation plus a learned projection. The history-side init gain is
configurable; larger values make the history branch matter earlier, at the
cost of perturbing the inherited static policy more strongly.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from tichu_env.encoding import STATIC_OBS_SIZE, PLAY_TOKEN_SIZE, MAX_HISTORY_LEN
from training.model_embed import (
    ACTION_DIM,
    PASS_SOURCE_COUNT,
    PASS_SOURCE_VOCAB,
    PASS_TARGET_COUNT,
    NonlinearActionEncoder,
    initialize_pass_provenance_embedding_,
    with_legacy_action_encoder_defaults,
)


# ── History encoders ─────────────────────────────────────────────────────────


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerHistoryEncoder(nn.Module):
    """Encode play history using a transformer plus attention pooling.

    Parameters
    ----------
    token_dim : input token width (default 63)
    d_model   : transformer hidden dimension
    nhead     : attention heads
    n_layers  : transformer layers
    ffn_dim   : feed-forward inner dimension
    dropout   : dropout (0 for self-play)
    """

    def __init__(
        self,
        token_dim: int = PLAY_TOKEN_SIZE,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.0,
        norm_first: bool = False,
        activation: str = "relu",
        use_checkpointing: bool = False,
        final_norm: bool = True,
    ):
        super().__init__()
        self.output_dim = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_enc = SinusoidalPE(d_model, max_len=MAX_HISTORY_LEN)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation=activation,
        )
        # Nested-tensor fastpath is noisy and not useful for this workload.
        # final_norm: a pre-norm (norm_first) stack never normalizes its
        # residual stream, so without a closing LayerNorm the token encodings
        # grow without bound over training (19 -> ~400 by 5.8M games).
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model) if final_norm else None,
            enable_nested_tensor=False,
        )
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.use_checkpointing = use_checkpointing
        nn.init.normal_(self.pool_query, std=0.02)

    def encode_tokens(
        self,
        history: torch.Tensor,  # (B, T, token_dim)
        history_len: torch.Tensor,  # (B,) int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = history.shape

        x = self.pos_enc(self.input_proj(history))  # (B, T, d_model)

        # Padding mask: True = ignore
        pad_mask = (
            torch.arange(T, device=x.device).unsqueeze(0)
            >= history_len.unsqueeze(1)
        )

        # For fully empty histories, unmask position 0 to avoid all-masked NaN
        empty = history_len == 0
        if empty.any():
            pad_mask = pad_mask.clone()
            pad_mask[empty, 0] = False

        # Use gradient checkpointing to save VRAM during backward pass
        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self.encoder, x, None, pad_mask, use_reentrant=False)
        else:
            x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, T, d_model)

        return x, pad_mask, empty

    def pool_encoded(
        self,
        encoded: torch.Tensor,
        pad_mask: torch.Tensor,
        empty: torch.Tensor,
        *,
        query: torch.Tensor | None = None,
        attention: nn.MultiheadAttention | None = None,
    ) -> torch.Tensor:
        """Pool an encoded sequence with the selected learned query."""
        B = encoded.shape[0]
        query_param = self.pool_query if query is None else query
        pool_attention = self.pool_attn if attention is None else attention

        # Learned attention pooling over valid tokens
        pool_mask = pad_mask
        if empty.any():
            pool_mask = pool_mask.clone()
            pool_mask[empty, 0] = False
        expanded_query = query_param.expand(B, -1, -1)
        pooled, _ = pool_attention(
            query=expanded_query,
            key=encoded,
            value=encoded,
            key_padding_mask=pool_mask,
            need_weights=False,
        )
        pooled = pooled.squeeze(1)  # (B, d_model)

        # Zero out empty sequences
        if empty.any():
            pooled = pooled.masked_fill(empty.unsqueeze(-1), 0.0)

        return pooled

    def forward(
        self,
        history: torch.Tensor,
        history_len: torch.Tensor,
    ) -> torch.Tensor:
        encoded, pad_mask, empty = self.encode_tokens(history, history_len)
        return self.pool_encoded(encoded, pad_mask, empty)


# ── Policy ───────────────────────────────────────────────────────────────────


class TichuPolicyHistory(nn.Module):
    """Hybrid Actor-Critic policy for Tichu with long-term memory.
    
    This model fuses two paths:
    1. Static Path: Process current game metadata (hands, scores, declarations)
       through a deep MLP.
    2. History Path: Process chronological play history (who played what and when)
       through a Transformer encoder with learned attention pooling.
       
    The two paths are fused using a residual connection, allowing the model
    to rely primarily on the static state while using history to refine its
    decisions.
    """
    def __init__(
        self,
        history_encoder: nn.Module,
        obs_size: int = STATIC_OBS_SIZE,
        hidden_dim: int = 256,
        n_layers: int = 2,
        embed_dim: int = 64,
        action_dim: int = ACTION_DIM,
        history_proj_gain: float = 0.1,
        pass_embed_dim: int = 8,
        trunk_dims: tuple[int, ...] | None = None,
        mlp_activation: str = "tanh",
        mlp_layer_norm: bool = False,
        split_actor_critic_history: bool = False,
        static_attention: bool = False,
        action_hidden_dim: int = 128,
        history_pool_norm: bool = True,
    ):
        super().__init__()

        # Static MLP trunk (same as TichuPolicyEmbed)
        self.split_actor_critic_history = split_actor_critic_history
        self.static_attention = static_attention
        dims = tuple(trunk_dims) if trunk_dims is not None else (hidden_dim,) * n_layers
        if not dims:
            raise ValueError("trunk_dims must contain at least one layer")
        if dims[-1] != hidden_dim:
            raise ValueError(
                f"trunk_dims must end at hidden_dim={hidden_dim}, got {dims}"
            )
        activation_factory = {
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }.get(mlp_activation)
        if activation_factory is None:
            raise ValueError(f"Unsupported MLP activation: {mlp_activation}")

        layers: list[nn.Module] = []
        in_dim = obs_size
        for out_dim in dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if mlp_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(activation_factory())
            in_dim = out_dim
        self.trunk = nn.Sequential(*layers)
        self.pass_source_embed = nn.Embedding(PASS_SOURCE_VOCAB, pass_embed_dim)
        self.pass_source_proj = nn.Linear(PASS_SOURCE_COUNT * pass_embed_dim, hidden_dim)
        self.pass_target_embed = nn.Embedding(PASS_SOURCE_VOCAB, pass_embed_dim)
        self.pass_target_proj = nn.Linear(PASS_TARGET_COUNT * pass_embed_dim, hidden_dim)

        # History — residual fusion: feat = static + fuse([static; hist])
        self.history_encoder = history_encoder
        fuse_in_dim = hidden_dim + history_encoder.output_dim
        if mlp_layer_norm:
            self.history_fuse = nn.Sequential(
                nn.LayerNorm(fuse_in_dim),
                nn.Linear(fuse_in_dim, hidden_dim),
                activation_factory(),
            )
        else:
            self.history_fuse = nn.Sequential(
                nn.Linear(fuse_in_dim, hidden_dim),
                activation_factory(),
            )
        self.history_proj_gain = history_proj_gain
        if static_attention:
            self.history_static_query = nn.Linear(
                hidden_dim, history_encoder.output_dim, bias=False
            )

        # history_pool_norm: normalize each pooled history vector on its own
        # before fusion. Otherwise the fuse LayerNorm sees [static ; history]
        # jointly, and a history vector hundreds of times larger than the
        # static features takes ~98% of its variance and squashes the rest.
        self.history_pool_norm = (
            nn.LayerNorm(history_encoder.output_dim) if history_pool_norm else None
        )
        if split_actor_critic_history:
            self.critic_history_pool_norm = (
                nn.LayerNorm(history_encoder.output_dim) if history_pool_norm else None
            )
            self.critic_pool_query = nn.Parameter(
                torch.zeros_like(history_encoder.pool_query)
            )
            self.critic_pool_attn = copy.deepcopy(history_encoder.pool_attn)
            self.critic_history_fuse = copy.deepcopy(self.history_fuse)
            nn.init.normal_(self.critic_pool_query, std=0.02)
            if static_attention:
                self.critic_history_static_query = nn.Linear(
                    hidden_dim, history_encoder.output_dim, bias=False
                )

        # Heads (same as TichuPolicyEmbed)
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

        # Keep all new encoder parameters after the legacy modules. This makes
        # the old optimizer parameter ordering a stable prefix on resume.
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
        # Fusion outputs the *delta* added via residual skip.
        # Zero the static portion so delta=0 when history is empty (preserves pretrained).
        # Orthogonal on the history portion so gradients flow from the start.
        fuse_linear = next(
            module for module in self.history_fuse if isinstance(module, nn.Linear)
        )
        hidden_dim = self.value_head.in_features
        hist_dim = self.history_encoder.output_dim
        with torch.no_grad():
            nn.init.zeros_(fuse_linear.weight)
            nn.init.zeros_(fuse_linear.bias)
            hist_block = torch.empty(hidden_dim, hist_dim)
            nn.init.orthogonal_(hist_block, gain=self.history_proj_gain)
            fuse_linear.weight[:, hidden_dim:] = hist_block
        if self.split_actor_critic_history:
            critic_fuse_linear = next(
                module for module in self.critic_history_fuse
                if isinstance(module, nn.Linear)
            )
            with torch.no_grad():
                nn.init.zeros_(critic_fuse_linear.weight)
                nn.init.zeros_(critic_fuse_linear.bias)
                hist_block = torch.empty(hidden_dim, hist_dim)
                nn.init.orthogonal_(hist_block, gain=self.history_proj_gain)
                critic_fuse_linear.weight[:, hidden_dim:] = hist_block
        if self.static_attention:
            nn.init.orthogonal_(self.history_static_query.weight, gain=1.0)
            if self.split_actor_critic_history:
                nn.init.orthogonal_(
                    self.critic_history_static_query.weight, gain=1.0
                )
        nn.init.orthogonal_(self.query_head.weight, gain=0.01)
        nn.init.zeros_(self.query_head.bias)
        nn.init.orthogonal_(self.action_embed.weight, gain=1.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _static_features(
        self,
        obs: torch.Tensor,
        pass_source_ids: torch.Tensor | None,
        pass_target_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        static_feat = self.trunk(obs)  # (B, H)
        if pass_source_ids is not None:
            pass_feat = self.pass_source_embed(pass_source_ids).reshape(obs.shape[0], -1)
            static_feat = static_feat + self.pass_source_proj(pass_feat)
        if pass_target_ids is not None:
            pass_feat = self.pass_target_embed(pass_target_ids).reshape(obs.shape[0], -1)
            static_feat = static_feat + self.pass_target_proj(pass_feat)
        return static_feat

    def _history_pool_query(
        self,
        static_feat: torch.Tensor,
        *,
        critic: bool = False,
    ) -> torch.Tensor | None:
        """Build a decision-conditioned query for history attention pooling."""
        if not self.static_attention:
            return self.critic_pool_query if critic else None

        if critic:
            base_query = self.critic_pool_query
            static_query = self.critic_history_static_query(static_feat)
        else:
            base_query = self.history_encoder.pool_query
            static_query = self.history_static_query(static_feat)
        return base_query + static_query.unsqueeze(1)

    def actor_critic_features(
        self,
        obs: torch.Tensor,
        history: torch.Tensor,
        history_len: torch.Tensor,
        pass_source_ids: torch.Tensor | None = None,
        pass_target_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return task-specific actor and critic state representations."""
        static_feat = self._static_features(
            obs, pass_source_ids, pass_target_ids
        )
        if not self.split_actor_critic_history:
            if self.static_attention:
                encoded, pad_mask, empty = self.history_encoder.encode_tokens(
                    history, history_len
                )
                hist = self.history_encoder.pool_encoded(
                    encoded,
                    pad_mask,
                    empty,
                    query=self._history_pool_query(static_feat),
                )
            else:
                hist = self.history_encoder(history, history_len)
            if self.history_pool_norm is not None:
                hist = self.history_pool_norm(hist)
            delta = self.history_fuse(torch.cat([static_feat, hist], dim=-1))
            fused = static_feat + delta
            return fused, fused

        encoded, pad_mask, empty = self.history_encoder.encode_tokens(
            history, history_len
        )
        actor_hist = self.history_encoder.pool_encoded(
            encoded,
            pad_mask,
            empty,
            query=self._history_pool_query(static_feat),
        )
        critic_hist = self.history_encoder.pool_encoded(
            encoded,
            pad_mask,
            empty,
            query=self._history_pool_query(static_feat, critic=True),
            attention=self.critic_pool_attn,
        )
        if self.history_pool_norm is not None:
            actor_hist = self.history_pool_norm(actor_hist)
            critic_hist = self.critic_history_pool_norm(critic_hist)
        actor_delta = self.history_fuse(
            torch.cat([static_feat, actor_hist], dim=-1)
        )
        critic_delta = self.critic_history_fuse(
            torch.cat([static_feat, critic_hist], dim=-1)
        )
        return static_feat + actor_delta, static_feat + critic_delta

    def fused_features(
        self,
        obs: torch.Tensor,
        history: torch.Tensor,
        history_len: torch.Tensor,
        pass_source_ids: torch.Tensor | None = None,
        pass_target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actor_feat, _ = self.actor_critic_features(
            obs, history, history_len, pass_source_ids, pass_target_ids
        )
        return actor_feat

    def value_features(
        self,
        obs: torch.Tensor,
        history: torch.Tensor,
        history_len: torch.Tensor,
        pass_source_ids: torch.Tensor | None = None,
        pass_target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, critic_feat = self.actor_critic_features(
            obs, history, history_len, pass_source_ids, pass_target_ids
        )
        return critic_feat

    def _forward_scores(
        self,
        obs: torch.Tensor,  # (B, obs_size)
        action_vecs: torch.Tensor,  # (B, A, action_dim)
        history: torch.Tensor,  # (B, T, token_dim)
        history_len: torch.Tensor,  # (B,) int
        pass_source_ids: torch.Tensor | None = None,  # (B, 3) int64
        pass_target_ids: torch.Tensor | None = None,  # (B, 3) int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_feat, critic_feat = self.actor_critic_features(
            obs, history, history_len, pass_source_ids, pass_target_ids
        )
        value = self.value_head(critic_feat).squeeze(-1)  # (B,)

        if action_vecs is None:
            return torch.empty((obs.shape[0], 0), device=obs.device), value

        query = self.query_head(actor_feat)  # (B, E)
        linear_keys = self.action_embed(action_vecs)  # (B, A, E)
        keys = self.action_encoder(action_vecs, linear_keys)  # (B, A, E)
        logits = torch.bmm(
            query.unsqueeze(1), keys.transpose(1, 2)
        ).squeeze(1)  # (B, A)
        return logits, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,  # (B, obs_size)
        action_vecs: torch.Tensor,  # (B, A, action_dim)
        n_legal: torch.Tensor,  # (B,) int
        history: torch.Tensor,  # (B, T, token_dim)
        history_len: torch.Tensor,  # (B,) int
        pass_source_ids: torch.Tensor | None = None,
        pass_target_ids: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,  # (B, A) uniforms; None = fresh randomness
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action. Returns (action_idx, log_prob, value), shape (B,).

        ``noise`` supplies the Gumbel-max uniforms, e.g. common random numbers
        shared between evaluation seatings; the action is still an exact draw
        from the policy.
        """
        A = action_vecs.shape[1]
        valid = (
            torch.arange(A, device=obs.device).unsqueeze(0) < n_legal.unsqueeze(1)
        )
        logits, value = self._forward_scores(
            obs, action_vecs, history, history_len, pass_source_ids, pass_target_ids
        )
        logits = logits.masked_fill(~valid, float("-inf"))

        uniforms = torch.rand_like(logits) if noise is None else noise.to(logits)
        gumbel = -torch.log(-torch.log(uniforms.clamp(1e-20, 1.0 - 1e-7)))
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
        obs: torch.Tensor,
        action_vecs: torch.Tensor,
        n_legal: torch.Tensor,
        actions: torch.Tensor,
        history: torch.Tensor,
        history_len: torch.Tensor,
        pass_source_ids: torch.Tensor | None = None,
        pass_target_ids: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Re-evaluate stored actions. Returns (log_prob, value, entropy), plus
        the masked logits and the legal mask when ``return_logits``."""
        A = action_vecs.shape[1]
        valid = (
            torch.arange(A, device=obs.device).unsqueeze(0) < n_legal.unsqueeze(1)
        )
        logits, value = self._forward_scores(
            obs, action_vecs, history, history_len, pass_source_ids, pass_target_ids
        )
        logits = logits.masked_fill(~valid, float("-inf"))
        dist = Categorical(logits=logits)
        if return_logits:
            return dist.log_prob(actions), value, dist.entropy(), logits, valid
        return dist.log_prob(actions), value, dist.entropy()
