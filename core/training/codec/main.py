import av
import torch

from dinov3_wrapper import DinoV3, DinoV3_versions


def load_video(video_path: str) -> torch.Tensor:
    with av.open(video_path) as container:
        container.streams.video[0].thread_type = "AUTO"
        frames = [
            torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
            for frame in container.decode(video=0)
        ]
    return torch.stack(frames).unsqueeze(0)


if __name__ == "__main__":
    video = load_video("/Users/dhruv/code/smash/core/training/codec/earth.mp4")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    dino = DinoV3(
        desired_hidden_states=[1, 4, 7, 9, 10, 11],
        dino_version=DinoV3_versions.VIT_SP,
    ).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    feature_chunks = []
    with torch.inference_mode():
        for chunk in video.split(16, dim=1):
            chunk = chunk.to(device=device, dtype=torch.float32).div_(255)
            feature_chunks.append(dino((chunk - mean) / std).cpu())

    features = torch.cat(feature_chunks, dim=1)
    print(features.shape)
