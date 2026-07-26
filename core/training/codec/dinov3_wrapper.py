import os
from enum import Enum
from pathlib import Path

import torch
from torch import Tensor


class DinoV3_versions(Enum):
    VIT_S = "dinov3_vits16"
    VIT_SP = "dinov3_vits16plus"
    VIT_B = "dinov3_vitb16"
    VIT_L = "dinov3_vitl16"
    VIT_HP = "dinov3_vith16plus"
    VIT_7 = "dinov3_vit7b16"


class DinoV3(torch.nn.Module):
    """Load DINOv3 and expose its requested intermediate feature maps."""

    def __init__(
        self,
        desired_hidden_states: list,
        dino_version: DinoV3_versions = DinoV3_versions.VIT_SP,
        pretrained: bool = True,
    ):
        super().__init__()
        self.desired_hidden_states = desired_hidden_states
        self.dino_version = dino_version
        load_options = {"pretrained": pretrained}
        if pretrained:
            load_options["weights"] = str(self._weights_path(dino_version))
        self.model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            model=dino_version.value,
            source="github",
            **load_options,
        )

        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _weights_path(dino_version: DinoV3_versions) -> Path:
        direct = os.environ.get("SMASH_DINO_WEIGHTS")
        if direct:
            path = Path(direct).expanduser()
            if path.is_file():
                return path
            raise FileNotFoundError(f"SMASH_DINO_WEIGHTS does not exist: {path}")

        roots = []
        if directory := os.environ.get("SMASH_DINO_WEIGHTS_DIR"):
            roots.append(Path(directory).expanduser())
        roots.append(Path(torch.hub.get_dir()) / "checkpoints")
        matches = [
            match
            for root in roots
            for match in root.glob(f"{dino_version.value}_pretrain_*.pth")
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                "Multiple matching DINO weights found; set SMASH_DINO_WEIGHTS to the one to use"
            )
        raise FileNotFoundError(
            f"No pretrained weights found for {dino_version.value}. Set SMASH_DINO_WEIGHTS "
            "to the .pth file or SMASH_DINO_WEIGHTS_DIR to its directory."
        )

    def intermediate_layers(self, input_video: Tensor) -> tuple[Tensor, ...]:
        batch, time, channels, height, width = input_video.shape
        flattened_video = input_video.reshape(batch * time, channels, height, width)
        hidden_states = tuple(
            self.model.get_intermediate_layers(
                flattened_video,
                n=self.desired_hidden_states,
                norm=True,
                reshape=True,
            )
        )
        return tuple(
            features.reshape(batch, time, *features.shape[1:])
            for features in hidden_states
        )

    def forward(self, input_video: Tensor) -> Tensor:
        hidden_states = self.intermediate_layers(input_video)
        # MIRA aggregates selected layers as mean(features) + features[-1].
        return torch.stack(hidden_states).mean(dim=0) + hidden_states[-1]
