"""Cache frozen codec latents for fast, matched transformer experiments."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from core.training.codec.codec import CodecEncoder
from core.training.codec.data import committed_archives

from .data import SlpVideoDataset


DINO_LAYERS = (1, 4, 7, 9, 10, 11)
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def _autocast(device: torch.device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _load_encoder(
    checkpoint_path: Path, device: torch.device
) -> tuple[CodecEncoder, Tensor, Tensor]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    encoder = CodecEncoder(
        desired_hidden_states=list(DINO_LAYERS), pretrained_dino=False
    )
    encoder.load_state_dict(
        {
            name.removeprefix("encoder."): value
            for name, value in checkpoint["model"].items()
            if name.startswith("encoder.")
        }
    )
    latent_mean, latent_std = (
        torch.as_tensor(value, device=device) for value in checkpoint["latent_mean_std"]
    )
    return encoder.requires_grad_(False).eval().to(device), latent_mean, latent_std


@torch.inference_mode()
def prepare_cache(
    data_dir: Path,
    codec_checkpoint: Path,
    output: Path,
    *,
    samples: int,
    frames: int,
    height: int,
    width: int,
    first_clip_frame: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SlpVideoDataset(
        committed_archives(data_dir),
        frames=frames,
        stride_frames=frames,
        size=(height, width),
        limit=samples,
        clips_per_sample=1,
        first_clip_frame=first_clip_frame,
    )
    loader = DataLoader(
        dataset, batch_size=1, num_workers=0, pin_memory=device.type == "cuda"
    )
    encoder, latent_mean, latent_std = _load_encoder(codec_checkpoint, device)
    mean = torch.tensor(DINO_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(DINO_STD, device=device).view(1, 1, 3, 1, 1)
    latents, actions, sample_names = [], [], []

    for index, (video, sample_actions, names, _) in enumerate(loader):
        video = video.to(device, non_blocking=True).float().div_(255)
        video = F.pad(
            (video - mean) / std,
            (0, -video.shape[-1] % 32, 0, -video.shape[-2] % 32, 0, 0),
            mode="replicate",
        )
        with _autocast(device):
            latent = (encoder(video) - latent_mean) / latent_std
        latents.append(latent.float().cpu())
        actions.append(sample_actions.float())
        sample_names.extend(names)
        print(json.dumps({"cached": index + 1, "sample": names[0]}), flush=True)

    if len(latents) != samples:
        raise RuntimeError(f"Requested {samples} samples, cached {len(latents)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.cat(latents),
            "actions": torch.cat(actions),
            "samples": sample_names,
            "video_frames": frames,
            "video_size": (height, width),
            "first_clip_frame": first_clip_frame,
            "codec_checkpoint": codec_checkpoint.name,
        },
        output,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "samples": len(sample_names),
                "latent_shape": list(torch.cat(latents).shape),
                "action_shape": list(torch.cat(actions).shape),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("codec_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--height", type=int, default=208)
    parser.add_argument("--width", type=int, default=252)
    parser.add_argument("--first-clip-frame", type=int, default=40)
    arguments = parser.parse_args()
    prepare_cache(**vars(arguments))


if __name__ == "__main__":
    main()
