from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .dinov3_wrapper import DinoV3, DinoV3_versions

DEFAULT_DINO_VERSION = DinoV3_versions.VIT_SP


def _init_weights(module: nn.Module) -> None:
    """MIRA initialization for trainable codec projections and transformer layers."""
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _rope(positions: Tensor, dim: int, theta: float) -> tuple[Tensor, Tensor]:
    frequencies = theta ** (-torch.arange(0, dim, 2, device=positions.device) / dim)
    angles = positions.float()[:, None] * frequencies[None]
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def _spatial_rope(
    height: int, width: int, dim: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    if dim % 4:
        raise ValueError("Attention head width must be divisible by four")
    half = dim // 2
    rows = torch.arange(height, device=device).repeat_interleave(width)
    columns = torch.arange(width, device=device).repeat(height)
    row_cos, row_sin = _rope(rows, half, 100.0)
    col_cos, col_sin = _rope(columns, half, 100.0)
    return torch.cat((row_cos, col_cos), -1), torch.cat((row_sin, col_sin), -1)


def _apply_rope(x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = (value[None, None] for value in rope)
    rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


class HeadLayerNorm(nn.Module):
    """Per-head LayerNorm used for MIRA's Q/K normalization."""

    def __init__(self, heads: int, width: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(heads, width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).square().mean(dim=-1, keepdim=True)
        scale = self.scale.float()[None, :, None, :]
        return ((x - mean) * torch.rsqrt(variance + self.eps) * scale).to(dtype)


class Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("Decoder width must be divisible by its attention heads")
        self.heads = heads
        head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_norm = HeadLayerNorm(heads, head_width)
        self.k_norm = HeadLayerNorm(heads, head_width)
        self.output = nn.Linear(width, width, bias=False)

    def forward(
        self, x: Tensor, rope: tuple[Tensor, Tensor], causal: bool = False
    ) -> Tensor:
        q, k, v = (
            rearrange(value, "b n (h d) -> b h n d", h=self.heads)
            for value in self.qkv(x).chunk(3, dim=-1)
        )
        q, k = _apply_rope(self.q_norm(q), rope), _apply_rope(self.k_norm(k), rope)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.output(rearrange(x, "b h n d -> b n (h d)"))


class SpaceTimeBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        hidden_width = 8 * width // 3
        self.norms = nn.ModuleList(nn.LayerNorm(width, eps=1e-6) for _ in range(3))
        self.space_attention = Attention(width, heads)
        self.time_attention = Attention(width, heads)
        self.mlp_in = nn.Linear(width, 2 * hidden_width, bias=False)
        self.mlp_out = nn.Linear(hidden_width, width, bias=False)
        self.scales = nn.Parameter(torch.full((3, width), 1e-4))

    def forward(
        self,
        x: Tensor,
        spatial_rope: tuple[Tensor, Tensor],
        temporal_rope: tuple[Tensor, Tensor],
    ) -> Tensor:
        batch, time, height, width, _ = x.shape
        x = rearrange(x, "b t h w c -> (b t) (h w) c")
        x = x + self.scales[0] * self.space_attention(self.norms[0](x), spatial_rope)
        x = rearrange(
            x, "(b t) (h w) c -> (b h w) t c", b=batch, t=time, h=height, w=width
        )
        x = x + self.scales[1] * self.time_attention(
            self.norms[1](x), temporal_rope, causal=True
        )
        x = rearrange(x, "(b h w) t c -> b t h w c", b=batch, h=height, w=width)
        gate, value = self.mlp_in(self.norms[2](x)).chunk(2, dim=-1)
        return x + self.scales[2] * self.mlp_out(F.silu(gate) * value)


class CodecDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        width: int = 1152,
        depth: int = 28,
        heads: int = 16,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.head_width = width // heads
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.activation_checkpointing = activation_checkpointing
        self.from_latent = nn.ConvTranspose2d(
            latent_dim, width, kernel_size=2, stride=2
        )
        self.blocks = nn.ModuleList(SpaceTimeBlock(width, heads) for _ in range(depth))
        self.output_norm = nn.LayerNorm(width, eps=1e-6)
        self.to_pixels = nn.Linear(width, 3 * temporal_patch_size * patch_size**2)
        self.apply(_init_weights)

    @property
    def last_layer_weight(self) -> Tensor:
        return self.to_pixels.weight

    def forward(self, z: Tensor) -> Tensor:
        batch, time = z.shape[:2]
        x = self.from_latent(rearrange(z, "b t c h w -> (b t) c h w"))
        x = rearrange(x, "(b t) c h w -> b t h w c", b=batch, t=time)
        spatial_rope = _spatial_rope(x.shape[2], x.shape[3], self.head_width, x.device)
        temporal_rope = _rope(
            torch.arange(time, device=x.device), self.head_width, 64.0
        )
        for block in self.blocks:
            if self.training and self.activation_checkpointing:
                x = checkpoint(
                    block, x, spatial_rope, temporal_rope, use_reentrant=False
                )
            else:
                x = block(x, spatial_rope, temporal_rope)
        x = self.to_pixels(self.output_norm(x))
        x = rearrange(
            x,
            "b t h w (c pt ph pw) -> b (t pt) c (h ph) (w pw)",
            c=3,
            pt=self.temporal_patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return torch.tanh(x)


class CodecEncoder(torch.nn.Module):
    def __init__(
        self,
        desired_hidden_states: list[int],
        dino_version: DinoV3_versions = DEFAULT_DINO_VERSION,
        pretrained_dino: bool = True,
    ):
        super().__init__()
        self.dinov3 = DinoV3(
            desired_hidden_states=desired_hidden_states,
            dino_version=dino_version,
            pretrained=pretrained_dino,
        )
        self.compressor = torch.nn.Sequential(
            Rearrange(
                "batch time features patch_h patch_w -> "
                "batch features time patch_h patch_w"
            ),
            torch.nn.Conv3d(
                in_channels=self.dinov3.model.num_features,
                out_channels=32,
                kernel_size=2,
                stride=2,
            ),
            Rearrange(
                "batch features time patch_h patch_w -> "
                "batch time features patch_h patch_w"
            ),
        )
        self.compressor.apply(_init_weights)

    def forward(
        self, input_video: Tensor, *, return_dino_features: bool = False
    ) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        dino_features = self.dinov3.intermediate_layers(input_video)
        aggregated = torch.stack(dino_features).mean(dim=0) + dino_features[-1]
        latent = self.compressor(aggregated)
        if return_dino_features:
            return latent, dino_features
        return latent


class Codec(torch.nn.Module):
    def __init__(
        self,
        desired_hidden_states: list[int],
        dino_version: DinoV3_versions = DEFAULT_DINO_VERSION,
        decoder: CodecDecoder | None = None,
    ):
        super().__init__()
        self.encoder = CodecEncoder(
            desired_hidden_states=desired_hidden_states,
            dino_version=dino_version,
        )
        self.decoder = decoder if decoder is not None else CodecDecoder()

    def encode(self, input_video: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_video)

    def decode(self, encoded_video: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded_video)

    def forward(
        self, input_video: Tensor, *, return_dino_features: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        if not return_dino_features:
            return self.decode(self.encode(input_video))
        encoded, dino_features = self.encoder(input_video, return_dino_features=True)
        return self.decode(encoded), encoded, dino_features
