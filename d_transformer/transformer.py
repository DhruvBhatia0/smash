from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from core.training.codec.codec import HeadLayerNorm, _apply_rope, _rope


def _spatial_rope(height: int, width: int, dim: int, device: torch.device):
    """MIRA's axial 2D RoPE, spanning periods from 2 to 100 tokens."""

    if dim % 4:
        raise ValueError("Attention head width must be divisible by four")
    frequencies = torch.logspace(
        math.log10(2 * math.pi / 100), math.log10(math.pi), dim // 4, device=device
    )
    rows = torch.arange(height, device=device)[:, None] * frequencies
    columns = torch.arange(width, device=device)[:, None] * frequencies
    row_cos, row_sin = (
        rows.cos().repeat_interleave(2, -1),
        rows.sin().repeat_interleave(2, -1),
    )
    column_cos = columns.cos().repeat_interleave(2, -1)
    column_sin = columns.sin().repeat_interleave(2, -1)
    return (
        torch.cat(
            (
                row_cos[:, None].expand(height, width, -1),
                column_cos[None].expand(height, width, -1),
            ),
            -1,
        ).reshape(height * width, dim),
        torch.cat(
            (
                row_sin[:, None].expand(height, width, -1),
                column_sin[None].expand(height, width, -1),
            ),
            -1,
        ).reshape(height * width, dim),
    )


class FlowTimeEmbedding(nn.Module):
    def __init__(self, width: int, frequency_dim: int = 256):
        super().__init__()
        half = frequency_dim // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half).float() / half)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, width), nn.SiLU(), nn.Linear(width, width)
        )

    def forward(self, flow_time: Tensor) -> Tensor:
        angles = 1_000 * flow_time.float()[..., None] * self.frequencies
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, 2 * width)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


def _initialize(module: nn.Module) -> None:
    if isinstance(module, AdaptiveLayerNorm):
        nn.init.zeros_(module.modulation.weight)
        nn.init.zeros_(module.modulation.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ActionEncoder(nn.Module):
    """Pool lossless, ordered Melee controller states to the codec rate."""

    def __init__(
        self,
        feature_dim: int,
        width: int,
        *,
        players: int = 2,
        microsteps: int = 3,
        temporal_downsampling: int = 2,
    ) -> None:
        super().__init__()
        self.players = players
        self.microsteps = microsteps
        self.temporal_downsampling = temporal_downsampling
        self.input = nn.Linear(feature_dim, width)
        self.microstep_embedding = nn.Parameter(torch.randn(microsteps, width) * 0.02)
        self.pool = nn.Linear(temporal_downsampling * microsteps * width, width)
        self.initial = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.player_embedding = nn.Parameter(torch.randn(players, width) * 0.02)
        self.player_projection = nn.Sequential(nn.SiLU(), nn.Linear(width, width))

    def forward(self, actions: Tensor) -> Tensor:
        if actions.ndim != 5 or actions.shape[2:4] != (self.microsteps, self.players):
            raise ValueError(
                "Actions must have shape (batch, video transitions, 3 microsteps, 2 players, features)"
            )
        embedded = self.input(actions)
        embedded = embedded + self.microstep_embedding[None, None, :, None]
        offset = self.temporal_downsampling - 1
        embedded = embedded[:, offset:]
        if embedded.shape[1] % self.temporal_downsampling:
            raise ValueError(
                "Action transitions do not align with the codec temporal stride"
            )
        embedded = rearrange(
            embedded,
            "b (t d) m p c -> b p t (d m c)",
            d=self.temporal_downsampling,
        )
        per_player = self.pool(embedded)
        initial = self.initial.expand(actions.shape[0], self.players, -1, -1)
        per_player = torch.cat((initial, per_player), dim=2)
        return self.player_projection(
            per_player + self.player_embedding[None, :, None]
        ).mean(1)


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, gating: bool = True):
        super().__init__()
        if width % heads or heads % kv_heads:
            raise ValueError(
                "Attention width/heads and query/KV heads must divide evenly"
            )
        self.heads, self.kv_heads = heads, kv_heads
        self.head_width = width // heads
        kv_width = kv_heads * self.head_width
        self.gating = gating
        self.qkv = nn.Linear(
            width, width + 2 * kv_width + (width if gating else 0), bias=False
        )
        self.q_norm = HeadLayerNorm(heads, self.head_width)
        self.k_norm = HeadLayerNorm(kv_heads, self.head_width)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor, rope, causal: bool = False) -> Tensor:
        width = self.heads * self.head_width
        kv_width = self.kv_heads * self.head_width
        parts = self.qkv(x).split(
            (width, kv_width, kv_width, width)
            if self.gating
            else (width, kv_width, kv_width),
            -1,
        )
        q, k, v = parts[:3]
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.kv_heads)
        q, k = _apply_rope(self.q_norm(q), rope), _apply_rope(self.k_norm(k), rope)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, enable_gqa=self.heads != self.kv_heads
        )
        y = rearrange(y, "b h n d -> b n (h d)")
        if self.gating:
            y = y * torch.sigmoid(parts[3])
        return self.output(y)


class SpaceTimeBlock(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, time_attention: bool):
        super().__init__()
        hidden_width = 256 * math.ceil((8 * width / 3) / 256)
        self.space_norm = AdaptiveLayerNorm(width)
        self.space_attention = Attention(width, heads, kv_heads)
        self.time_norm = AdaptiveLayerNorm(width) if time_attention else None
        self.time_attention = (
            Attention(width, heads, kv_heads) if time_attention else None
        )
        self.mlp_norm = AdaptiveLayerNorm(width)
        self.mlp_in = nn.Linear(width, 2 * hidden_width, bias=False)
        self.mlp_out = nn.Linear(hidden_width, width, bias=False)

    def forward(
        self, x: Tensor, condition: Tensor, spatial_rope, temporal_rope
    ) -> Tensor:
        batch, time, height, width, _ = x.shape
        space = rearrange(self.space_norm(x, condition), "b t h w c -> (b t) (h w) c")
        x = x + rearrange(
            self.space_attention(space, spatial_rope),
            "(b t) (h w) c -> b t h w c",
            b=batch,
            t=time,
            h=height,
            w=width,
        )
        if self.time_attention is not None and self.time_norm is not None:
            temporal = rearrange(
                self.time_norm(x, condition), "b t h w c -> (b h w) t c"
            )
            x = x + rearrange(
                self.time_attention(temporal, temporal_rope, causal=True),
                "(b h w) t c -> b t h w c",
                b=batch,
                h=height,
                w=width,
            )
        gate, value = self.mlp_in(self.mlp_norm(x, condition)).chunk(2, dim=-1)
        return x + self.mlp_out(F.silu(gate) * value)


class DiffusionTransformer(nn.Module):
    """MIRA 1B DiT adapted to two players' lossless Melee controller streams."""

    def __init__(
        self,
        latent_dim: int = 32,
        action_features: int = 56,
        latent_height: int = 7,
        latent_width: int = 8,
        width: int = 2048,
        depth: int = 16,
        heads: int = 16,
        kv_heads: int = 4,
        time_attention_every: int = 4,
        use_clean_past: bool = True,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_height, self.latent_width = latent_height, latent_width
        self.head_width = width // heads
        self.use_clean_past = use_clean_past
        self.activation_checkpointing = activation_checkpointing
        self.from_latent = nn.Linear(latent_dim, width)
        self.from_past = nn.Linear(latent_dim, width)
        self.bos = nn.Parameter(
            torch.randn(latent_dim, latent_height, latent_width) * 0.02
        )
        self.actions = ActionEncoder(action_features, width)
        self.flow_time_embedding = FlowTimeEmbedding(width)
        self.blocks = nn.ModuleList(
            SpaceTimeBlock(
                width,
                heads,
                kv_heads,
                time_attention=index % time_attention_every == 0 or index == depth - 1,
            )
            for index in range(depth)
        )
        self.to_velocity = nn.Linear(width, latent_dim)
        self.apply(_initialize)

    def shifted_past(self, clean_latents: Tensor) -> Tensor:
        batch = clean_latents.shape[0]
        return torch.cat(
            (
                self.bos[None, None].expand(batch, -1, -1, -1, -1),
                clean_latents[:, :-1],
            ),
            1,
        )

    def forward(
        self,
        z: Tensor,
        actions: Tensor,
        flow_time: Tensor,
        clean_past: Tensor | None = None,
    ) -> Tensor:
        if z.ndim != 5 or z.shape[2:] != (
            self.latent_dim,
            self.latent_height,
            self.latent_width,
        ):
            raise ValueError("Unexpected codec latent shape")
        if flow_time.shape != z.shape[:2]:
            raise ValueError("Flow time must have shape (batch, latent time)")
        if actions.shape[1] != z.shape[1] * 2 - 1:
            raise ValueError(
                "Expected one three-input action group per video transition"
            )

        batch, time, _, height, width = z.shape
        if clean_past is None and self.use_clean_past:
            clean_past = self.shifted_past(z)
        x = self.from_latent(rearrange(z, "b t c h w -> b t h w c"))
        if clean_past is not None:
            x = x + self.from_past(rearrange(clean_past, "b t c h w -> b t h w c"))
        condition = self.actions(actions) + self.flow_time_embedding(flow_time)
        condition = rearrange(condition, "b t c -> b t 1 1 c")
        spatial_rope = _spatial_rope(height, width, self.head_width, z.device)
        temporal_rope = _rope(
            torch.arange(time, device=z.device).float() / 10, self.head_width, 64.0
        )
        for block in self.blocks:
            if self.training and self.activation_checkpointing:
                x = checkpoint(
                    block,
                    x,
                    condition,
                    spatial_rope,
                    temporal_rope,
                    use_reentrant=False,
                )
            else:
                x = block(x, condition, spatial_rope, temporal_rope)
        return rearrange(self.to_velocity(x), "b t h w c -> b t c h w")


def flow_matching_loss(
    model: nn.Module,
    clean_latents: Tensor,
    actions: Tensor,
    *,
    noise: Tensor | None = None,
    flow_time: Tensor | None = None,
) -> Tensor:
    """MIRA's diagonal per-latent-frame flow-matching objective."""

    predicted_velocity, target_velocity, _ = flow_matching_prediction(
        model,
        clean_latents,
        actions,
        noise=noise,
        flow_time=flow_time,
    )
    return F.mse_loss(predicted_velocity.float(), target_velocity.float())


def flow_matching_prediction(
    model: nn.Module,
    clean_latents: Tensor,
    actions: Tensor,
    *,
    noise: Tensor | None = None,
    flow_time: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return velocity prediction, target, and sampled flow time for diagnostics."""

    batch, time = clean_latents.shape[:2]
    noise = torch.randn_like(clean_latents) if noise is None else noise
    flow_time = (
        torch.rand(batch, time, device=clean_latents.device)
        if flow_time is None
        else flow_time
    )
    interpolation = rearrange(flow_time, "b t -> b t 1 1 1").to(clean_latents.dtype)
    noised_latents = interpolation * clean_latents + (1 - interpolation) * noise
    target_velocity = clean_latents - noise
    raw_model = getattr(model, "module", model)
    clean_past = (
        raw_model.shifted_past(clean_latents) if raw_model.use_clean_past else None
    )
    predicted_velocity = model(noised_latents, actions, flow_time, clean_past)
    return predicted_velocity, target_velocity, flow_time
