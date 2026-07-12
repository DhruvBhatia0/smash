import torch
from enum import Enum


class DinoV3_versions(Enum):
    VIT_S = "dinov3_vits16"
    VIT_SP = "dinov3_vits16plus"
    VIT_B = "dinov3_vitb16"
    VIT_L = "dinov3_vitl16"
    VIT_HP = "dinov3_vith16plus"
    VIT_7 = "dinov3_vit7b16"

class DinoV3(torch.nn.Module):
    '''loads a dinov3 model, provides a forward method that also returns intermediate layers'''
    def __init__(self, desired_hidden_states: list, dino_version: DinoV3_versions = DinoV3_versions.VIT_SP):
        super().__init__()
        self.desired_hidden_states = desired_hidden_states
        self.dino_version = dino_version
        self.model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            model=dino_version.value,
            source="github",
            pretrained=True,
            weights="/Users/dhruv/code/smash/.checkpoints/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        )

        self.model.eval()
        self.model.requires_grad_(False)
    
    def forward(self, input_video: torch.tensor ):
        B, T, C, H, W = input_video.shape
        # flatten dim 0, 1
        flattened_video = input_video.reshape(B*T, C, H, W)

        hidden_states = self.model.get_intermediate_layers(
            flattened_video,
            n=self.desired_hidden_states,
            reshape=True,
        )

        averaged_hidden_states = torch.stack(hidden_states).mean(dim=0)
        # Output: (B, T, num_features, patch_h, patch_w); num_features is 384 for ViT-S+.
        return averaged_hidden_states.reshape(B, T, *averaged_hidden_states.shape[1:])
