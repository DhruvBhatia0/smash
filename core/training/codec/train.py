"""Train the video codec with ``torchrun --nproc-per-node=8 -m core.training.codec.train``."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import random
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import av
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from .codec import Codec, CodecDecoder


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    checkpoint_dir: Path = Path("checkpoints/codec")
    epochs: int = 3
    steps: int | None = None
    save_every: int = 5_000
    log_every: int = 10
    batch_size: int = 4
    workers: int = 6
    learning_rate: float = 1e-4
    train_fraction: float = 0.9
    split_seed: int = 28
    clip_seconds: float = 2.0
    eval_batches: int = 32
    frames: int = 40
    height: int = 208
    width: int = 252
    fps: int = 20
    loss: str = "l1"
    compile: bool = True
    wandb_project: str = "smash-codec"
    wandb_name: str | None = None
    wandb_mode: str = "online"
    hf_repo: str | None = None

    @classmethod
    def from_cli(cls) -> TrainConfig:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("data_dir", type=Path)
        parser.add_argument("--checkpoint-dir", type=Path, default=cls.checkpoint_dir)
        parser.add_argument("--epochs", type=int, default=cls.epochs)
        parser.add_argument("--steps", type=int)
        parser.add_argument("--save-every", type=int, default=cls.save_every)
        parser.add_argument("--log-every", type=int, default=cls.log_every)
        parser.add_argument("--batch-size", type=int, default=cls.batch_size)
        parser.add_argument("--workers", type=int, default=cls.workers)
        parser.add_argument("--learning-rate", type=float, default=cls.learning_rate)
        parser.add_argument("--train-fraction", type=float, default=cls.train_fraction)
        parser.add_argument("--split-seed", type=int, default=cls.split_seed)
        parser.add_argument("--clip-seconds", type=float, default=cls.clip_seconds)
        parser.add_argument("--eval-batches", type=int, default=cls.eval_batches)
        parser.add_argument("--loss", choices=("l1", "mira"), default=cls.loss)
        parser.add_argument(
            "--compile", action=argparse.BooleanOptionalAction, default=cls.compile
        )
        parser.add_argument("--wandb-project", default=cls.wandb_project)
        parser.add_argument("--wandb-name")
        parser.add_argument("--hf-repo")
        parser.add_argument(
            "--wandb-mode",
            choices=("online", "offline", "disabled"),
            default=cls.wandb_mode,
        )
        return cls(**vars(parser.parse_args()))


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frames: int
    fps: float


class VideoClipDataset(Dataset[Tensor]):
    """Seek-decodes fixed clips without loading complete replays."""

    INDEX_NAME = ".codec-video-index.json"
    VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".nut", ".webm"}

    def __init__(
        self,
        videos: list[VideoInfo],
        clip_seconds: float,
        target_fps: int,
        size: tuple[int, int],
    ):
        self.videos = videos
        self.clip_seconds = clip_seconds
        self.target_fps = target_fps
        self.output_frames = round(clip_seconds * target_fps)
        self.height, self.width = size
        self.offsets = [0]
        for video in videos:
            if video.fps < target_fps:
                raise ValueError(
                    f"{video.path} is {video.fps:g} FPS; expected at least {target_fps}"
                )
            source_frames = round(video.fps * clip_seconds)
            self.offsets.append(self.offsets[-1] + video.frames // source_frames)
        self._containers: OrderedDict[Path, Any] = OrderedDict()
        if not self.offsets[-1]:
            raise ValueError("No complete video clips found")

    @classmethod
    def index(cls, folder: Path) -> list[VideoInfo]:
        index_path = folder / cls.INDEX_NAME
        if index_path.exists():
            rows = json.loads(index_path.read_text())
            return [
                VideoInfo(folder / row["path"], row["frames"], row["fps"])
                for row in rows
            ]

        paths = sorted(
            path
            for path in folder.rglob("*")
            if path.suffix.lower() in cls.VIDEO_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"No videos found in {folder}")
        with ThreadPoolExecutor(max_workers=min(32, len(paths))) as pool:
            videos = list(pool.map(cls._probe, paths))
        rows = [
            {
                "path": str(video.path.relative_to(folder)),
                "frames": video.frames,
                "fps": video.fps,
            }
            for video in videos
        ]
        temporary = index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, indent=2) + "\n")
        temporary.replace(index_path)
        return videos

    @staticmethod
    def _probe(path: Path) -> VideoInfo:
        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            if stream.frames:
                frames = stream.frames
            elif stream.duration is not None:
                frames = round(float(stream.duration * stream.time_base) * fps)
            elif container.duration is not None:
                frames = round(container.duration / av.time_base * fps)
            else:
                raise ValueError(f"Could not determine the duration of {path}")
        return VideoInfo(path, frames, fps)

    @staticmethod
    def split(
        videos: list[VideoInfo], train_fraction: float, seed: int
    ) -> tuple[list[VideoInfo], list[VideoInfo]]:
        if len(videos) < 2 or not 0 < train_fraction < 1:
            raise ValueError(
                "The dataset needs at least two videos and a split strictly between 0 and 1"
            )
        videos = videos.copy()
        random.Random(seed).shuffle(videos)
        split = min(len(videos) - 1, max(1, round(len(videos) * train_fraction)))
        return videos[:split], videos[split:]

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> Tensor:
        video_index = bisect.bisect_right(self.offsets, index) - 1
        clip_index = index - self.offsets[video_index]
        info = self.videos[video_index]
        source_start = clip_index * round(info.fps * self.clip_seconds)
        wanted = [
            source_start + round(frame * info.fps / self.target_fps)
            for frame in range(self.output_frames)
        ]

        container = self._open(info.path)
        stream = container.streams.video[0]
        stream_start = stream.start_time or 0
        start_seconds = source_start / info.fps
        container.seek(
            stream_start + round(start_seconds / float(stream.time_base)),
            stream=stream,
            backward=True,
        )
        output = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            source_frame = round(
                float((frame.pts - stream_start) * stream.time_base) * info.fps
            )
            if source_frame < wanted[len(output)]:
                continue
            resized = frame.reformat(
                width=self.width, height=self.height, format="rgb24"
            )
            output.append(torch.from_numpy(resized.to_ndarray()).permute(2, 0, 1))
            if len(output) == self.output_frames:
                return torch.stack(output)
        raise RuntimeError(f"Could not decode clip {clip_index} from {info.path}")

    def _open(self, path: Path):
        if path in self._containers:
            container = self._containers.pop(path)
        else:
            container = av.open(path)
            container.streams.video[0].thread_count = 1
        self._containers[path] = container
        if len(self._containers) > 2:
            self._containers.popitem(last=False)[1].close()
        return container

    @property
    def replay_ranges(self) -> list[tuple[int, int]]:
        return list(zip(self.offsets, self.offsets[1:]))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_containers"] = OrderedDict()
        return state


class ReplaySampler(Sampler[int]):
    """Shuffles replays and clips while keeping file access locally grouped."""

    def __init__(self, dataset: VideoClipDataset, shuffle: bool, seed: int):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.samples = (len(dataset) + self.replicas - 1) // self.replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        replay_order = (
            torch.randperm(len(self.dataset.videos), generator=generator).tolist()
            if self.shuffle
            else range(len(self.dataset.videos))
        )
        indices = []
        for replay in replay_order:
            start, stop = self.dataset.replay_ranges[replay]
            if self.shuffle:
                order = (
                    torch.randperm(stop - start, generator=generator)
                    .add(start)
                    .tolist()
                )
                indices.extend(order)
            else:
                indices.extend(range(start, stop))
        total = self.samples * self.replicas
        indices.extend(
            (indices * ((total - len(indices)) // len(indices) + 1))[
                : total - len(indices)
            ]
        )
        return iter(indices[self.rank : total : self.replicas])

    def __len__(self) -> int:
        return self.samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class ReconstructionLoss(nn.Module):
    """Pixel L1 alone or MIRA's L1, LPIPS, and DINO objective."""

    def __init__(self, mode: str, dino: nn.Module, last_layer: Tensor):
        super().__init__()
        self.mode = mode
        self.dino = dino
        self.last_layer = last_layer

        self.lpips: nn.Module | None = None
        if mode == "mira":
            import lpips

            self.lpips = (
                lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False)
            )

    def forward(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        prediction, target = prediction.float(), target.float()
        losses = {"pixel": F.l1_loss(prediction, target)}
        if self.mode == "l1":
            return {"loss": losses["pixel"], **losses}

        losses |= self._mira_losses(prediction, target)
        total = losses["pixel"]
        for name in ("lpips", "dino"):
            weight = self._adaptive_weight(losses["pixel"], losses[name])
            losses[f"{name}_weight"] = weight
            total = total + weight * losses[name]

        return {"loss": total, **losses}

    def _mira_losses(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        assert self.lpips is not None
        time = prediction.shape[1]
        count = max(1, round(0.25 * time))

        lpips_frames = (
            torch.randperm(time, device=prediction.device)[:count].sort().values
        )
        predicted = prediction[:, lpips_frames].flatten(0, 1)
        expected = target[:, lpips_frames].flatten(0, 1)
        perceptual = self.lpips(predicted, expected).mean()

        dino_frames = (
            torch.randperm(time, device=prediction.device)[:count].sort().values
        )
        predicted = self._normalize_dino((prediction[:, dino_frames] + 1) / 2)
        expected = self._normalize_dino((target[:, dino_frames] + 1) / 2)
        predicted_features = self._dino_features(predicted)
        with torch.no_grad():
            expected_features = self._dino_features(expected)
        dino = torch.stack(
            [
                F.mse_loss(
                    F.normalize(actual, dim=1, eps=1e-6),
                    F.normalize(wanted, dim=1, eps=1e-6),
                )
                for actual, wanted in zip(predicted_features, expected_features)
            ]
        ).mean()
        return {"lpips": perceptual, "dino": dino}

    def _normalize_dino(self, video: Tensor) -> Tensor:
        mean = video.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
        std = video.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
        return (video - mean) / std

    def _dino_features(self, video: Tensor) -> tuple[Tensor, ...]:
        frames = video.flatten(0, 1)
        return tuple(
            self.dino.model.get_intermediate_layers(
                frames,
                n=self.dino.desired_hidden_states,
                norm=True,
                reshape=True,
            )
        )

    def _adaptive_weight(self, anchor: Tensor, other: Tensor) -> Tensor:
        if not torch.is_grad_enabled() or not anchor.requires_grad:
            return anchor.new_ones(())
        anchor_gradient = torch.autograd.grad(
            anchor, self.last_layer, retain_graph=True
        )[0]
        other_gradient = torch.autograd.grad(other, self.last_layer, retain_graph=True)[
            0
        ]
        return (
            (anchor_gradient.norm() / (other_gradient.norm() + 1e-6))
            .clamp(0, 1e4)
            .detach()
        )


class CodecTrainer:
    DINO_MEAN = (0.485, 0.456, 0.406)
    DINO_STD = (0.229, 0.224, 0.225)
    DINO_LAYERS = (1, 4, 7, 9, 10, 11)

    def __init__(self, config: TrainConfig):
        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.distributed = self.world_size > 1

        if self.distributed:
            dist.init_process_group("nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        torch.manual_seed(28 + self.rank)
        torch.set_float32_matmul_precision("high")
        videos = self._video_index()
        train_videos, eval_videos = VideoClipDataset.split(
            videos, config.train_fraction, config.split_seed
        )
        self.train_loader, self.train_sampler = self._loader(
            train_videos, training=True
        )
        self.eval_loader, _ = self._loader(eval_videos, training=False)
        epoch_steps = len(self.train_loader)
        self.steps = min(
            config.steps or config.epochs * epoch_steps, config.epochs * epoch_steps
        )
        if not self.rank:
            print(
                f"{len(train_videos):,} train videos / {len(eval_videos):,} eval videos; "
                f"{len(self.train_loader.dataset):,} train clips / "
                f"{len(self.eval_loader.dataset):,} eval clips; {epoch_steps:,} steps/epoch; "
                f"training for {self.steps:,} steps"
            )

        self.raw_model = Codec(
            desired_hidden_states=list(self.DINO_LAYERS),
            decoder=CodecDecoder(activation_checkpointing=True),
        ).to(self.device)
        self.model: nn.Module = self.raw_model
        if self.distributed:
            self.model = DistributedDataParallel(
                self.raw_model, device_ids=[self.local_rank], broadcast_buffers=False
            )
        if config.compile:
            self.model.compile()

        parameters = (
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self.optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.steps, eta_min=1e-6
        )
        self.loss = ReconstructionLoss(
            config.loss,
            dino=self.raw_model.encoder.dinov3,
            last_layer=self.raw_model.decoder.last_layer_weight,
        ).to(self.device)
        self.mean = torch.tensor(self.DINO_MEAN, device=self.device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(self.DINO_STD, device=self.device).view(1, 1, 3, 1, 1)
        self.wandb: Any | None = None
        self.hf: Any | None = None
        self.hf_uploads: list[Any] = []
        self._start_wandb()
        self._start_huggingface()

    def _video_index(self) -> list[VideoInfo]:
        if not self.rank:
            VideoClipDataset.index(self.config.data_dir)
        if self.distributed:
            dist.barrier()
        return VideoClipDataset.index(self.config.data_dir)

    def _loader(
        self, videos: list[VideoInfo], training: bool
    ) -> tuple[DataLoader[Tensor], ReplaySampler]:
        dataset = VideoClipDataset(
            videos,
            clip_seconds=self.config.clip_seconds,
            target_fps=self.config.fps,
            size=(self.config.height, self.config.width),
        )
        if dataset.output_frames != self.config.frames:
            raise ValueError(
                f"{self.config.clip_seconds:g}s at {self.config.fps} FPS produces "
                f"{dataset.output_frames} frames, not {self.config.frames}"
            )
        sampler = ReplaySampler(dataset, shuffle=training, seed=self.config.split_seed)
        worker_options = (
            {"persistent_workers": True, "prefetch_factor": 2}
            if self.config.workers
            else {}
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=training,
            **worker_options,
        )
        if training and not len(loader):
            raise ValueError(
                "The training folder must contain at least one full global batch"
            )
        return loader, sampler

    def _start_wandb(self) -> None:
        if self.rank or self.config.wandb_mode == "disabled":
            return
        import wandb  # Imported only by the process that logs.

        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self.config).items()
        }
        config |= {
            "train_clips": len(self.train_loader.dataset),
            "eval_clips": len(self.eval_loader.dataset),
            "training_steps": self.steps,
        }
        self.wandb = wandb
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            mode=self.config.wandb_mode,
            config=config,
        )

    def _start_huggingface(self) -> None:
        if self.rank or not self.config.hf_repo:
            return
        from huggingface_hub import HfApi

        self.hf = HfApi()
        self.hf.create_repo(self.config.hf_repo, exist_ok=True)

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _reconstruct(self, video: Tensor) -> tuple[Tensor, Tensor]:
        target = video.mul(2).sub(1)
        model_input = (video - self.mean) / self.std
        alignment = 2 * self.raw_model.decoder.patch_size
        pad_height = -video.shape[-2] % alignment
        pad_width = -video.shape[-1] % alignment
        model_input = F.pad(
            model_input, (0, pad_width, 0, pad_height, 0, 0), mode="replicate"
        )
        prediction = self.model(model_input)[..., : video.shape[-2], : video.shape[-1]]
        return prediction, target

    def _train_step(self, video: Tensor) -> dict[str, Tensor]:
        video = video.to(self.device, non_blocking=True).float().div_(255)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            prediction, target = self._reconstruct(video)
            losses = self.loss(prediction, target)
        losses["loss"].backward()
        self.optimizer.step()
        self.scheduler.step()
        return {name: value.detach() for name, value in losses.items()}

    @torch.inference_mode()
    def _evaluate(self) -> tuple[dict[str, float], tuple[Tensor, Tensor]]:
        self.model.eval()
        totals: dict[str, Tensor] = {}
        samples = torch.zeros((), device=self.device)
        sample: tuple[Tensor, Tensor] | None = None
        for batch, video in enumerate(self.eval_loader):
            if batch == self.config.eval_batches:
                break
            video = video.to(self.device, non_blocking=True).float().div_(255)
            with self._autocast():
                prediction, target = self._reconstruct(video)
                losses = self.loss(prediction, target)
            for name, value in losses.items():
                totals[name] = totals.get(name, torch.zeros_like(value)) + value * len(
                    video
                )
            samples += len(video)
            sample = sample or (prediction[0].float().cpu(), target[0].cpu())
        if self.distributed:
            for value in (*totals.values(), samples):
                dist.all_reduce(value)
        self.model.train()
        self.raw_model.encoder.dinov3.eval()
        assert sample is not None
        return {
            name: (value / samples).item() for name, value in totals.items()
        }, sample

    def _save_and_log(self, step: int) -> None:
        eval_losses, (prediction, target) = self._evaluate()
        if not self.rank:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.config.checkpoint_dir / f"step-{step:07d}.pt"
            torch.save(
                {
                    "step": step,
                    "model": self.raw_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "config": asdict(self.config),
                },
                checkpoint_path,
            )
            comparison = torch.cat((target, prediction), dim=-1).add(1).div(2)
            comparison = comparison.clamp(0, 1).mul(255).byte()
            video_path = checkpoint_path.with_suffix(".mp4")
            self._write_video(comparison, video_path)
            if self.hf:
                self.hf_uploads.append(
                    self.hf.upload_folder(
                        repo_id=self.config.hf_repo,
                        folder_path=self.config.checkpoint_dir,
                        allow_patterns=[checkpoint_path.name, video_path.name],
                        commit_message=f"Codec checkpoint at step {step:,}",
                        run_as_future=True,
                    )
                )
            if self.wandb:
                self.wandb.log(
                    {
                        **{
                            f"eval/{name}": value for name, value in eval_losses.items()
                        },
                        "eval/reconstruction": self.wandb.Video(
                            str(video_path),
                            format="mp4",
                            caption="target | reconstruction",
                        ),
                    },
                    step=step,
                )
        if self.distributed:
            dist.barrier()

    def _write_video(self, video: Tensor, path: Path) -> None:
        frames = video.permute(0, 2, 3, 1).contiguous().numpy()
        with av.open(path, mode="w") as container:
            stream = container.add_stream("libx264", rate=self.config.fps)
            stream.width, stream.height = video.shape[-1], video.shape[-2]
            stream.pix_fmt = "yuv420p"
            for frame in frames:
                container.mux(
                    stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24"))
                )
            container.mux(stream.encode())

    def run(self) -> None:
        self.model.train()
        self.raw_model.encoder.dinov3.eval()
        step = 0
        try:
            for epoch in range(self.config.epochs):
                self.train_sampler.set_epoch(epoch)
                for video in self.train_loader:
                    step += 1
                    train_losses = self._train_step(video)
                    if step % self.config.log_every == 0:
                        if self.distributed:
                            for value in train_losses.values():
                                dist.all_reduce(value)
                                value /= self.world_size
                        if self.wandb:
                            self.wandb.log(
                                {
                                    **{
                                        f"train/{name}": value.item()
                                        for name, value in train_losses.items()
                                    },
                                    "train/learning_rate": self.scheduler.get_last_lr()[
                                        0
                                    ],
                                },
                                step=step,
                            )
                    if step % self.config.save_every == 0 or step == self.steps:
                        self._save_and_log(step)
                    if step == self.steps:
                        return
        finally:
            if self.wandb:
                self.wandb.finish()
            for upload in self.hf_uploads:
                upload.result()
            if self.distributed:
                dist.destroy_process_group()


if __name__ == "__main__":
    CodecTrainer(TrainConfig.from_cli()).run()
