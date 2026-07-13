"""Train the video codec with ``torchrun --nproc-per-node=8 -m core.training.codec.train``."""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from typing import Any

import av
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .codec import Codec, CodecDecoder


@dataclass(frozen=True)
class TrainConfig:
    train_dir: Path
    eval_dir: Path
    checkpoint_dir: Path = Path("checkpoints/codec")
    steps: int = 250_000
    save_every: int = 5_000
    log_every: int = 10
    batch_size: int = 4
    workers: int = 6
    learning_rate: float = 1e-4
    frames: int = 40
    height: int = 288
    width: int = 512
    fps: int = 20
    loss: str = "l1"
    compile: bool = True
    wandb_project: str = "smash-codec"
    wandb_name: str | None = None
    wandb_mode: str = "online"

    @classmethod
    def from_cli(cls) -> TrainConfig:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("train_dir", type=Path)
        parser.add_argument("eval_dir", type=Path)
        parser.add_argument("--checkpoint-dir", type=Path, default=cls.checkpoint_dir)
        parser.add_argument("--steps", type=int, default=cls.steps)
        parser.add_argument("--save-every", type=int, default=cls.save_every)
        parser.add_argument("--log-every", type=int, default=cls.log_every)
        parser.add_argument("--batch-size", type=int, default=cls.batch_size)
        parser.add_argument("--workers", type=int, default=cls.workers)
        parser.add_argument("--learning-rate", type=float, default=cls.learning_rate)
        parser.add_argument("--loss", choices=("l1", "mira"), default=cls.loss)
        parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=cls.compile)
        parser.add_argument("--wandb-project", default=cls.wandb_project)
        parser.add_argument("--wandb-name")
        parser.add_argument(
            "--wandb-mode",
            choices=("online", "offline", "disabled"),
            default=cls.wandb_mode,
        )
        return cls(**vars(parser.parse_args()))


class VideoFolderDataset(Dataset[Tensor]):
    """Loads one fixed-length, uniformly sampled video clip per file."""

    VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}

    def __init__(self, folder: Path, frames: int, size: tuple[int, int]):
        self.paths = sorted(
            path for path in folder.rglob("*") if path.suffix.lower() in self.VIDEO_EXTENSIONS
        )
        self.frames = frames
        self.height, self.width = size
        if not self.paths:
            raise ValueError(f"No video clips found in {folder}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[index]
        with av.open(path) as container:
            container.streams.video[0].thread_type = "AUTO"
            frames = list(container.decode(video=0))

        if len(frames) < self.frames:
            raise ValueError(
                f"{path} contains {len(frames)} frames; expected at least {self.frames}"
            )

        indices = torch.linspace(0, len(frames) - 1, self.frames).round().long()
        video = []
        for index in indices.tolist():
            frame = frames[index].reformat(width=self.width, height=self.height, format="rgb24")
            video.append(torch.from_numpy(frame.to_ndarray()).permute(2, 0, 1))
        return torch.stack(video)


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

        lpips_frames = torch.randperm(time, device=prediction.device)[:count].sort().values
        predicted = prediction[:, lpips_frames].flatten(0, 1)
        expected = target[:, lpips_frames].flatten(0, 1)
        perceptual = self.lpips(predicted, expected).mean()

        dino_frames = torch.randperm(time, device=prediction.device)[:count].sort().values
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
        self.train_loader, self.train_sampler = self._loader(config.train_dir, training=True)
        self.eval_loader, _ = self._loader(config.eval_dir, training=False)

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

        parameters = (parameter for parameter in self.model.parameters() if parameter.requires_grad)
        self.optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.steps, eta_min=1e-6
        )
        self.loss = ReconstructionLoss(
            config.loss,
            dino=self.raw_model.encoder.dinov3,
            last_layer=self.raw_model.decoder.last_layer_weight,
        ).to(self.device)
        self.mean = torch.tensor(self.DINO_MEAN, device=self.device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(self.DINO_STD, device=self.device).view(1, 1, 3, 1, 1)
        self.wandb: Any | None = None
        self._start_wandb()

    def _loader(
        self, folder: Path, training: bool
    ) -> tuple[DataLoader[Tensor], DistributedSampler[Tensor] | None]:
        dataset = VideoFolderDataset(
            folder,
            frames=self.config.frames,
            size=(self.config.height, self.config.width),
        )
        sampler = DistributedSampler(dataset, shuffle=training) if self.distributed else None
        worker_options = (
            {"persistent_workers": True, "prefetch_factor": 2} if self.config.workers else {}
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=training and sampler is None,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=training,
            **worker_options,
        )
        if training and not len(loader):
            raise ValueError("The training folder must contain at least one full global batch")
        return loader, sampler

    def _start_wandb(self) -> None:
        if self.rank or self.config.wandb_mode == "disabled":
            return
        import wandb  # Imported only by the process that logs.

        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self.config).items()
        }
        self.wandb = wandb
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            mode=self.config.wandb_mode,
            config=config,
        )

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _reconstruct(self, video: Tensor) -> tuple[Tensor, Tensor]:
        target = video.mul(2).sub(1)
        prediction = self.model((video - self.mean) / self.std)
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
        for video in self.eval_loader:
            video = video.to(self.device, non_blocking=True).float().div_(255)
            with self._autocast():
                prediction, target = self._reconstruct(video)
                losses = self.loss(prediction, target)
            for name, value in losses.items():
                totals[name] = totals.get(name, torch.zeros_like(value)) + value * len(video)
            samples += len(video)
            sample = sample or (prediction[0].float().cpu(), target[0].cpu())
        if self.distributed:
            for value in (*totals.values(), samples):
                dist.all_reduce(value)
        self.model.train()
        self.raw_model.encoder.dinov3.eval()
        assert sample is not None
        return {name: (value / samples).item() for name, value in totals.items()}, sample

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
            if self.wandb:
                self.wandb.log(
                    {
                        **{f"eval/{name}": value for name, value in eval_losses.items()},
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
                container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
            container.mux(stream.encode())

    def run(self) -> None:
        self.model.train()
        self.raw_model.encoder.dinov3.eval()
        step = 0
        try:
            for epoch in count():
                if self.train_sampler:
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
                                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                                },
                                step=step,
                            )
                    if step % self.config.save_every == 0 or step == self.config.steps:
                        self._save_and_log(step)
                    if step == self.config.steps:
                        return
        finally:
            if self.wandb:
                self.wandb.finish()
            if self.distributed:
                dist.destroy_process_group()


if __name__ == "__main__":
    CodecTrainer(TrainConfig.from_cli()).run()
