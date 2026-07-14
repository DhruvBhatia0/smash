from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from core.training.codec.codec import _apply_rope, _rope, _spatial_rope


class FlowTimeEmbedding(nn.Module):
    def __init__(self, width: int, frequency_dim: int = 256):
        super().__init__()
        if frequency_dim % 2:
            raise ValueError("Flow-time frequency dimension must be even")

        half = frequency_dim // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half).float() / half)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, flow_time: Tensor) -> Tensor:
        # b t -> b t 1
        flow_time = rearrange(flow_time.float(), "b t -> b t 1")
        angles = 1_000 * flow_time * self.frequencies
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int):
        super().__init__()
        if width % heads:
            raise ValueError(
                "Transformer width must be divisible by its attention heads"
            )
        if heads % kv_heads:
            raise ValueError("Query heads must be divisible by key/value heads")

        self.heads = heads
        self.kv_heads = kv_heads
        self.head_width = width // heads
        kv_width = kv_heads * self.head_width
        self.query_key_value_gate = nn.Linear(
            width, 2 * width + 2 * kv_width, bias=False
        )
        self.q_norm = nn.LayerNorm(self.head_width)
        self.k_norm = nn.LayerNorm(self.head_width)
        self.output = nn.Linear(width, width, bias=False)

    def forward(
        self, x: Tensor, rope: tuple[Tensor, Tensor], causal: bool = False
    ) -> Tensor:
        width = self.heads * self.head_width
        kv_width = self.kv_heads * self.head_width
        q, k, v, gate = self.query_key_value_gate(x).split(
            (width, kv_width, kv_width, width), dim=-1
        )

        # b n c -> b h n d
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.kv_heads)
        q = _apply_rope(self.q_norm(q), rope)
        k = _apply_rope(self.k_norm(k), rope)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            enable_gqa=self.heads != self.kv_heads,
        )
        # b h n d -> b n c
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.output(x * torch.sigmoid(gate))


class SpaceTimeBlock(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, time_attention: bool):
        super().__init__()
        hidden_width = 8 * width // 3
        self.space_norm = AdaptiveLayerNorm(width)
        self.space_attention = Attention(width, heads, kv_heads)
        self.time_norm = AdaptiveLayerNorm(width) if time_attention else None
        self.time_attention = (
            Attention(width, heads, kv_heads) if time_attention else None
        )
        self.mlp_norm = AdaptiveLayerNorm(width)
        self.mlp_in = nn.Linear(width, 2 * hidden_width, bias=False)
        self.mlp_out = nn.Linear(hidden_width, width, bias=False)
        self.scales = nn.Parameter(torch.full((3, width), 1e-4))

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        spatial_rope: tuple[Tensor, Tensor],
        temporal_rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        batch, time, height, width, _ = x.shape

        # b t h w c -> (b t) (h w) c
        space = rearrange(self.space_norm(x, condition), "b t h w c -> (b t) (h w) c")
        space = self.space_attention(space, spatial_rope)
        # (b t) (h w) c -> b t h w c
        space = rearrange(
            space, "(b t) (h w) c -> b t h w c", b=batch, t=time, h=height, w=width
        )
        x = x + self.scales[0] * space

        if self.time_attention is not None and self.time_norm is not None:
            # b t h w c -> (b h w) t c
            temporal = rearrange(
                self.time_norm(x, condition), "b t h w c -> (b h w) t c"
            )
            temporal = self.time_attention(temporal, temporal_rope, causal=True)
            # (b h w) t c -> b t h w c
            temporal = rearrange(
                temporal,
                "(b h w) t c -> b t h w c",
                b=batch,
                h=height,
                w=width,
            )
            x = x + self.scales[1] * temporal

        gate, value = self.mlp_in(self.mlp_norm(x, condition)).chunk(2, dim=-1)
        return x + self.scales[2] * self.mlp_out(F.silu(gate) * value)


class DiffusionTransformer(nn.Module):
    """Predicts flow velocity on codec latents shaped ``(batch, time, channels, height, width)``."""

    def __init__(
        self,
        latent_dim: int = 32,
        width: int = 2048,
        depth: int = 16,
        heads: int = 16,
        kv_heads: int = 4,
        time_attention_every: int = 4,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("Transformer depth must be positive")
        if time_attention_every < 1:
            raise ValueError("Temporal attention frequency must be positive")
        if heads < 1 or kv_heads < 1:
            raise ValueError("Attention head counts must be positive")
        if width % heads:
            raise ValueError(
                "Transformer width must be divisible by its attention heads"
            )
        if width // heads % 4:
            raise ValueError(
                "Attention head width must be divisible by four for spatial RoPE"
            )

        self.latent_dim = latent_dim
        self.head_width = width // heads
        self.activation_checkpointing = activation_checkpointing
        self.from_latent = nn.Linear(latent_dim, width)
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
        self.output_norm = nn.LayerNorm(width)
        self.to_velocity = nn.Linear(width, latent_dim)

    def forward(self, z: Tensor, flow_time: Tensor) -> Tensor:
        if z.ndim != 5:
            raise ValueError(
                "Latents must have shape (batch, time, channels, height, width)"
            )
        if z.shape[2] != self.latent_dim:
            raise ValueError(
                f"Expected {self.latent_dim} latent channels, got {z.shape[2]}"
            )
        if flow_time.shape != z.shape[:2]:
            raise ValueError("Flow time must have shape (batch, time)")

        _, time, _, height, width = z.shape
        # b t c h w -> b t h w c
        x = self.from_latent(rearrange(z, "b t c h w -> b t h w c"))
        # b t c -> b t 1 1 c
        condition = rearrange(self.flow_time_embedding(flow_time), "b t c -> b t 1 1 c")
        spatial_rope = _spatial_rope(height, width, self.head_width, z.device)
        temporal_rope = _rope(
            torch.arange(time, device=z.device), self.head_width, 64.0
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

        x = self.to_velocity(self.output_norm(x))
        # b t h w c -> b t c h w
        return rearrange(x, "b t h w c -> b t c h w")


def flow_matching_loss(model: nn.Module, clean_latents: Tensor) -> Tensor:
    """Diffusion-forcing loss with an independent flow time for every latent frame."""
    batch, time = clean_latents.shape[:2]
    noise = torch.randn_like(clean_latents)
    target_velocity = clean_latents - noise
    flow_time = torch.rand(batch, time, device=clean_latents.device)
    # b t -> b t 1 1 1
    interpolation = rearrange(flow_time, "b t -> b t 1 1 1").to(clean_latents.dtype)
    noised_latents = interpolation * clean_latents + (1 - interpolation) * noise
    predicted_velocity = model(noised_latents, flow_time)
    return F.mse_loss(predicted_velocity.float(), target_velocity.float())
