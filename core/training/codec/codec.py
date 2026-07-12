import torch
from einops.layers.torch import Rearrange

from .dinov3_wrapper import DinoV3, DinoV3_versions

DEFAULT_DINO_VERSION = DinoV3_versions.VIT_SP


class Codec(torch.nn.Module):

    def __init__(
        self,
        desired_hidden_states: list[int],
        dino_version: DinoV3_versions = DEFAULT_DINO_VERSION,
    ):
        super().__init__()
        self.dinov3 = DinoV3(
            desired_hidden_states=desired_hidden_states,
            dino_version=dino_version,
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

    def forward(self, input_video: torch.Tensor) -> torch.Tensor:
        features = self.dinov3(input_video)
        return self.compressor(features)
