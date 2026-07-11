#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image


REPO = Path("dinov3")
WEIGHTS = Path("checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
IMAGE = Path("smash-sample.jpg")
OUT = Path("outputs/smash-sample.dinov3_vits16plus.pt")


def load_image(path: Path, size: int = 256) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0)


sys.path.insert(0, str(REPO.resolve()))
from dinov3.hub.backbones import dinov3_vits16plus  # noqa: E402

model = dinov3_vits16plus(pretrained=False)
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
model.eval()

with torch.inference_mode():
    x = load_image(IMAGE)
    cls = model(x)
    features = model.forward_features(x)

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "cls_embedding": cls,
        "patch_tokens": features["x_norm_patchtokens"],
        "storage_tokens": features["x_storage_tokens"],
    },
    OUT,
)

print("model: dinov3_vits16plus")
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"cls_embedding: {tuple(cls.shape)}")
print(f"patch_tokens: {tuple(features['x_norm_patchtokens'].shape)}")
print(f"saved: {OUT}")
