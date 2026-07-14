"""Train the video codec with ``torchrun --nproc-per-node=8 -m core.training.codec.train``."""

from __future__ import annotations

import argparse
import os
import time
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
from torch.utils.data import DataLoader

from .data import TarZstdClipDataset, committed_archives, split_archives


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    checkpoint_dir: Path = Path("checkpoints/codec")
    steps: int = 250_001
    save_every: int = 12_500
    eval_every: int = 5_000
    log_every: int = 2_500
    batch_size: int = 4
    workers: int = 6
    learning_rate: float = 1e-4
    warmup_steps: int = 1_000
    min_learning_rate: float = 1e-6
    model_ema_decay: float = 0.9999
    latent_ema_decay: float = 0.99
    train_fraction: float = 0.9
    split_seed: int = 28
    eval_samples: int = 1_024
    frames: int = 40
    height: int = 208
    width: int = 252
    fps: int = 20
    prefetch_factor: int = 1
    spool_mb: int = 64
    decoder_threads: int = 1
    loader_benchmark_batches: int = 0
    compile: bool = True
    wandb_project: str = "smash-codec"
    wandb_name: str | None = None
    wandb_mode: str = "online"

    @classmethod
    def from_cli(cls) -> TrainConfig:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("data_dir", type=Path)
        parser.add_argument("--checkpoint-dir", type=Path, default=cls.checkpoint_dir)
        parser.add_argument("--steps", type=int, default=cls.steps)
        parser.add_argument("--save-every", type=int, default=cls.save_every)
        parser.add_argument("--eval-every", type=int, default=cls.eval_every)
        parser.add_argument("--log-every", type=int, default=cls.log_every)
        parser.add_argument("--batch-size", type=int, default=cls.batch_size)
        parser.add_argument("--workers", type=int, default=cls.workers)
        parser.add_argument("--learning-rate", type=float, default=cls.learning_rate)
        parser.add_argument("--warmup-steps", type=int, default=cls.warmup_steps)
        parser.add_argument(
            "--min-learning-rate", type=float, default=cls.min_learning_rate
        )
        parser.add_argument(
            "--model-ema-decay", type=float, default=cls.model_ema_decay
        )
        parser.add_argument(
            "--latent-ema-decay", type=float, default=cls.latent_ema_decay
        )
        parser.add_argument("--train-fraction", type=float, default=cls.train_fraction)
        parser.add_argument("--split-seed", type=int, default=cls.split_seed)
        parser.add_argument("--eval-samples", type=int, default=cls.eval_samples)
        parser.add_argument("--prefetch-factor", type=int, default=cls.prefetch_factor)
        parser.add_argument("--spool-mb", type=int, default=cls.spool_mb)
        parser.add_argument("--decoder-threads", type=int, default=cls.decoder_threads)
        parser.add_argument(
            "--loader-benchmark-batches", type=int, default=cls.loader_benchmark_batches
        )
        parser.add_argument(
            "--compile", action=argparse.BooleanOptionalAction, default=cls.compile
        )
        parser.add_argument("--wandb-project", default=cls.wandb_project)
        parser.add_argument("--wandb-name")
        parser.add_argument(
            "--wandb-mode",
            choices=("online", "offline", "disabled"),
            default=cls.wandb_mode,
        )
        arguments = parser.parse_args()
        if (
            min(
                arguments.steps,
                arguments.save_every,
                arguments.eval_every,
                arguments.log_every,
                arguments.batch_size,
                arguments.eval_samples,
            )
            < 1
        ):
            parser.error(
                "steps, intervals, batch size, and evaluation samples must be positive"
            )
        if (
            arguments.workers < 0
            or arguments.loader_benchmark_batches < 0
            or arguments.warmup_steps < 0
        ):
            parser.error(
                "workers, benchmark batches, and warmup steps cannot be negative"
            )
        if (
            arguments.prefetch_factor < 1
            or arguments.spool_mb < 1
            or arguments.decoder_threads < 1
        ):
            parser.error("prefetch, spool size, and decoder threads must be positive")
        if (
            not 0 <= arguments.model_ema_decay < 1
            or not 0 <= arguments.latent_ema_decay < 1
        ):
            parser.error("EMA decay values must be in [0, 1)")
        return cls(**vars(arguments))


class CodecTrainer:
    DINO_MEAN = (0.485, 0.456, 0.406)
    DINO_STD = (0.229, 0.224, 0.225)
    DINO_LAYERS = (1, 4, 7, 9, 10, 11)

    def __init__(self, config: TrainConfig):
        from .codec import Codec, CodecDecoder
        from .loss import ReconstructionLoss
        from .optimization import ModelEMA, ScalarEMA, WarmupCosineLR

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
        self.train_loader, self.eval_loader = _loaders(
            config, rank=self.rank, world_size=self.world_size
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
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            decay_steps=max(0, config.steps - config.warmup_steps - 1),
            min_lr=config.min_learning_rate,
        )
        self.loss = ReconstructionLoss(
            dino=self.raw_model.encoder.dinov3,
            last_layer=self.raw_model.decoder.last_layer_weight,
        ).to(self.device)
        self.model_ema = ModelEMA(self.raw_model, decay=config.model_ema_decay)
        self.latent_mean = ScalarEMA(config.latent_ema_decay)
        self.latent_std = ScalarEMA(config.latent_ema_decay, initial_value=1.0)
        self.mean = torch.tensor(self.DINO_MEAN, device=self.device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(self.DINO_STD, device=self.device).view(1, 1, 3, 1, 1)
        self.wandb: Any | None = None
        self._start_wandb()

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

    def _reconstruct(
        self, video: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[Tensor, ...]]:
        target = video.mul(2).sub(1)
        model_input = (video - self.mean) / self.std
        alignment = 2 * self.raw_model.decoder.patch_size
        pad_height = -video.shape[-2] % alignment
        pad_width = -video.shape[-1] % alignment
        model_input = F.pad(
            model_input, (0, pad_width, 0, pad_height, 0, 0), mode="replicate"
        )
        output = self.model(model_input, return_dino_features=True)
        assert isinstance(output, tuple)
        padded_prediction, latent, dino_features = output
        prediction = padded_prediction[..., : video.shape[-2], : video.shape[-1]]
        return prediction, target, padded_prediction, latent, dino_features

    def _train_step(self, video: Tensor) -> dict[str, Tensor]:
        video = video.to(self.device, non_blocking=True).float().div_(255)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            prediction, target, dino_prediction, latent, dino_features = (
                self._reconstruct(video)
            )
            losses = self.loss(
                prediction,
                target,
                dino_prediction=dino_prediction,
                target_dino_features=dino_features,
            )
        losses["loss_total"].backward()
        self.optimizer.step()
        self.scheduler.step()
        self.model_ema.step()
        self.latent_mean.update(latent)
        self.latent_std.update(latent.float().std(keepdim=True))
        return {name: value.detach() for name, value in losses.items()}

    @torch.inference_mode()
    def _evaluate(self) -> tuple[dict[str, float], tuple[Tensor, Tensor]]:
        self.model.eval()
        totals: dict[str, Tensor] = {}
        samples = torch.zeros((), device=self.device)
        sample: tuple[Tensor, Tensor] | None = None
        eval_batches = max(
            1,
            self.config.eval_samples // (self.config.batch_size * self.world_size),
        )
        for batch, video in enumerate(self.eval_loader):
            if batch == eval_batches:
                break
            video = video.to(self.device, non_blocking=True).float().div_(255)
            with self._autocast():
                prediction, target, dino_prediction, _, dino_features = (
                    self._reconstruct(video)
                )
                losses = self.loss(
                    prediction,
                    target,
                    dino_prediction=dino_prediction,
                    target_dino_features=dino_features,
                )
            for name, value in losses.items():
                totals[name] = totals.get(name, torch.zeros_like(value)) + value * len(
                    video
                )
            samples += len(video)
            if sample is None:
                sample = (prediction[0].float().cpu(), target[0].cpu())
        if self.distributed:
            for value in (*totals.values(), samples):
                dist.all_reduce(value)
        self.model.train()
        self.raw_model.encoder.dinov3.eval()
        assert sample is not None
        return (
            {name: (value / samples).item() for name, value in totals.items()},
            sample,
        )

    def _evaluate_and_log(self, step: int) -> None:
        with self.model_ema.average_parameters():
            eval_losses, (prediction, target) = self._evaluate()
            if not self.rank:
                self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                comparison = torch.cat((target, prediction), dim=-1).add(1).div(2)
                comparison = comparison.clamp(0, 1).mul(255).byte()
                video_path = self.config.checkpoint_dir / f"eval-step-{step:07d}.mp4"
                self._write_video(comparison, video_path)
                if self.wandb:
                    self.wandb.log(
                        {
                            **{
                                f"eval/{name}": value
                                for name, value in eval_losses.items()
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

    def _save(self, step: int) -> None:
        self.latent_mean.synchronize(self.device)
        self.latent_std.synchronize(self.device)
        with self.model_ema.average_parameters():
            if not self.rank:
                self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = self.config.checkpoint_dir / f"step-{step:07d}.pt"
                torch.save(
                    {
                        "step": step,
                        # Downstream consumers get MIRA's EMA weights.
                        "model": self.raw_model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "config": asdict(self.config),
                        "latent_mean_std": [
                            self.latent_mean.value,
                            self.latent_std.value,
                        ],
                    },
                    checkpoint_path,
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
            self._evaluate_and_log(step)
            for video in self.train_loader:
                step += 1
                train_losses = self._train_step(video)
                if (
                    step <= 10
                    or step % self.config.log_every == 0
                    or step == self.config.steps
                ):
                    if self.distributed:
                        for value in train_losses.values():
                            dist.all_reduce(value)
                            value /= self.world_size
                    self.latent_mean.synchronize(self.device)
                    self.latent_std.synchronize(self.device)
                    if self.wandb:
                        self.wandb.log(
                            {
                                **{
                                    f"train/{name}": value.item()
                                    for name, value in train_losses.items()
                                },
                                "train/learning_rate": self.scheduler.get_last_lr()[0],
                                "train/latent_mean": self.latent_mean.value,
                                "train/latent_std": self.latent_std.value,
                            },
                            step=step,
                        )
                if step % self.config.eval_every == 0 or step == self.config.steps:
                    self._evaluate_and_log(step)
                if step % self.config.save_every == 0 or step == self.config.steps:
                    self._save(step)
                if step == self.config.steps:
                    return
        finally:
            if self.wandb:
                self.wandb.finish()
            if self.distributed:
                dist.destroy_process_group()


def _loaders(
    config: TrainConfig, *, rank: int, world_size: int
) -> tuple[_BatchLoader, _BatchLoader]:
    if dist.is_initialized():
        payload = [committed_archives(config.data_dir) if not rank else None]
        dist.broadcast_object_list(payload, src=0)
        archives = payload[0]
        assert archives is not None
    else:
        archives = committed_archives(config.data_dir)
    train_archives, eval_archives = split_archives(
        archives, config.train_fraction, config.split_seed, minimum=world_size
    )
    common = {
        "frames": config.frames,
        "fps": config.fps,
        "size": (config.height, config.width),
        "rank": rank,
        "world_size": world_size,
        "seed": config.split_seed,
        "spool_bytes": config.spool_mb * 1024**2,
        "decoder_threads": config.decoder_threads,
    }
    train_dataset = TarZstdClipDataset(train_archives, training=True, **common)
    eval_dataset = TarZstdClipDataset(eval_archives, training=False, **common)
    if not rank:
        print(
            f"{len(train_archives)} train archives / "
            f"{sum(item.samples for item in train_archives):,} manifest rows; "
            f"{len(eval_archives)} eval archives / "
            f"{sum(item.samples for item in eval_archives):,} manifest rows"
        )
    return (
        _data_loader(config, train_dataset, training=True),
        _data_loader(config, eval_dataset, training=False),
    )


def _data_loader(
    config: TrainConfig, dataset: TarZstdClipDataset, *, training: bool
) -> _BatchLoader:
    worker_options = (
        {
            "persistent_workers": True,
            "prefetch_factor": config.prefetch_factor,
        }
        if config.workers
        else {}
    )
    samples = DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.workers,
        **worker_options,
    )
    return _BatchLoader(
        samples,
        batch_size=config.batch_size,
        drop_last=training,
        pin_memory=torch.cuda.is_available(),
    )


class _BatchLoader:
    """Collate in the parent process to avoid large worker shared-memory batches."""

    def __init__(
        self,
        samples: DataLoader[Tensor],
        *,
        batch_size: int,
        drop_last: bool,
        pin_memory: bool,
    ) -> None:
        self.samples = samples
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory

    def __iter__(self):
        batch: list[Tensor] = []
        for sample in self.samples:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self._stack(batch)
                batch.clear()
        if batch and not self.drop_last:
            yield self._stack(batch)

    def _stack(self, samples: list[Tensor]) -> Tensor:
        if not self.pin_memory:
            return torch.stack(samples)
        output = torch.empty(
            (len(samples), *samples[0].shape),
            dtype=samples[0].dtype,
            pin_memory=True,
        )
        return torch.stack(samples, out=output)


def _benchmark_loader(config: TrainConfig) -> None:
    archives = committed_archives(config.data_dir)
    dataset = TarZstdClipDataset(
        archives,
        frames=config.frames,
        fps=config.fps,
        size=(config.height, config.width),
        training=True,
        seed=config.split_seed,
        spool_bytes=config.spool_mb * 1024**2,
        decoder_threads=config.decoder_threads,
    )
    train_loader = _data_loader(config, dataset, training=True)
    print(
        f"benchmarking {len(archives)} archives / "
        f"{sum(item.samples for item in archives):,} manifest rows"
    )
    started = time.monotonic()
    clips = 0
    payload_bytes = 0
    for batch, video in enumerate(train_loader, start=1):
        clips += len(video)
        payload_bytes += video.numel() * video.element_size()
        if batch == config.loader_benchmark_batches:
            break
    elapsed = time.monotonic() - started
    print(
        f"loader benchmark: {config.loader_benchmark_batches} batches / {clips} clips in "
        f"{elapsed:.3f}s = {clips / elapsed:.2f} clips/s, "
        f"{payload_bytes / elapsed / 1024**2:.2f} MiB/s"
    )


if __name__ == "__main__":
    configuration = TrainConfig.from_cli()
    if configuration.loader_benchmark_batches:
        _benchmark_loader(configuration)
    else:
        CodecTrainer(configuration).run()
