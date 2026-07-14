"""Train the video codec with ``torchrun --nproc-per-node=8 -m core.training.codec.train``."""

from __future__ import annotations

import argparse
import math
import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

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
    epochs: int = 7
    steps: int | None = None
    log_every: int = 100
    batch_size: int = 18
    workers: int = 6
    learning_rate: float = 1e-4
    warmup_steps: int = 1_000
    min_learning_rate: float = 1e-6
    model_ema_decay: float = 0.9999
    latent_ema_decay: float = 0.99
    train_fraction: float = 0.95
    split_seed: int = 28
    eval_samples: int = 1_024
    frames: int = 40
    stride_frames: int = 10
    height: int = 208
    width: int = 252
    fps: int = 20
    prefetch_factor: int = 4
    video_prefetch: int = 4
    batch_prefetch: int = 4
    spool_mb: int = 64
    decoder_threads: int = 1
    loader_benchmark_batches: int = 0
    compile: bool = True
    wandb_project: str = "smash-codec"
    wandb_name: str | None = None
    wandb_mode: str = "online"
    hf_repo: str | None = None
    hf_private: bool = True

    @classmethod
    def from_cli(cls) -> TrainConfig:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("data_dir", type=Path)
        parser.add_argument("--checkpoint-dir", type=Path, default=cls.checkpoint_dir)
        parser.add_argument("--epochs", type=int, default=cls.epochs)
        parser.add_argument(
            "--steps",
            type=int,
            help="Override the epoch-derived step count (intended for smoke tests)",
        )
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
        parser.add_argument("--stride-frames", type=int, default=cls.stride_frames)
        parser.add_argument("--prefetch-factor", type=int, default=cls.prefetch_factor)
        parser.add_argument("--video-prefetch", type=int, default=cls.video_prefetch)
        parser.add_argument("--batch-prefetch", type=int, default=cls.batch_prefetch)
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
        parser.add_argument("--hf-repo")
        parser.add_argument(
            "--hf-private",
            action=argparse.BooleanOptionalAction,
            default=cls.hf_private,
        )
        arguments = parser.parse_args()
        positive = (
            arguments.epochs,
            arguments.log_every,
            arguments.batch_size,
            arguments.eval_samples,
            arguments.stride_frames,
            arguments.prefetch_factor,
            arguments.video_prefetch,
            arguments.batch_prefetch,
            arguments.spool_mb,
            arguments.decoder_threads,
        )
        if min(positive) < 1 or (arguments.steps is not None and arguments.steps < 1):
            parser.error("epochs, steps, sizes, and prefetch values must be positive")
        if (
            arguments.workers < 0
            or arguments.loader_benchmark_batches < 0
            or arguments.warmup_steps < 0
        ):
            parser.error(
                "workers, benchmark batches, and warmup steps cannot be negative"
            )
        if not 0 < arguments.train_fraction < 1:
            parser.error("train fraction must be strictly between zero and one")
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
        self.train_loader, self.eval_loader, self.steps_per_epoch = _loaders(
            config, rank=self.rank, world_size=self.world_size
        )
        self.total_steps = config.steps or config.epochs * self.steps_per_epoch

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
            decay_steps=max(0, self.total_steps - config.warmup_steps - 1),
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
        self.hf_api: Any | None = None
        self.hf_uploads: list[Any] = []
        self._start_wandb()
        self._start_huggingface()

    def _start_wandb(self) -> None:
        if self.rank or self.config.wandb_mode == "disabled":
            return
        import wandb  # Imported only by the process that logs.

        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self.config).items()
        }
        self.wandb = wandb
        run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            mode=self.config.wandb_mode,
            config=config,
        )
        if run.url:
            print(f"Weights & Biases: {run.url}", flush=True)

    def _start_huggingface(self) -> None:
        if self.rank or not self.config.hf_repo:
            return
        from huggingface_hub import HfApi

        self.hf_api = HfApi()
        repo = self.hf_api.create_repo(
            repo_id=self.config.hf_repo,
            repo_type="model",
            private=self.config.hf_private,
            exist_ok=True,
        )
        print(f"Hugging Face checkpoints: {repo}", flush=True)

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _reconstruct(
        self, video: Tensor, *, model: nn.Module | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[Tensor, ...]]:
        target = video.mul(2).sub(1)
        model_input = (video - self.mean) / self.std
        alignment = 2 * self.raw_model.decoder.patch_size
        pad_height = -video.shape[-2] % alignment
        pad_width = -video.shape[-1] % alignment
        model_input = F.pad(
            model_input, (0, pad_width, 0, pad_height, 0, 0), mode="replicate"
        )
        output = (model or self.model)(model_input, return_dino_features=True)
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
        if self.rank or self.eval_loader is None:
            raise RuntimeError("Evaluation is owned by rank zero")
        self.raw_model.eval()
        totals: dict[str, Tensor] = {}
        samples = 0
        sample: tuple[Tensor, Tensor] | None = None
        for video in self.eval_loader:
            video = video.to(self.device, non_blocking=True).float().div_(255)
            with self._autocast():
                prediction, target, dino_prediction, _, dino_features = (
                    self._reconstruct(video, model=self.raw_model)
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
            if samples >= self.config.eval_samples:
                break
        self.raw_model.train()
        self.raw_model.encoder.dinov3.eval()
        assert sample is not None and samples
        return (
            {name: (value / samples).item() for name, value in totals.items()},
            sample,
        )

    def _evaluate_and_log(self, step: int) -> None:
        if self.distributed:
            dist.barrier()
        if not self.rank:
            with self.model_ema.average_parameters():
                eval_losses, (prediction, target) = self._evaluate()
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
        if not self.rank:
            with self.model_ema.average_parameters():
                self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = self.config.checkpoint_dir / f"step-{step:07d}.pt"
                torch.save(
                    {
                        "step": step,
                        "epoch": math.ceil(step / self.steps_per_epoch),
                        "steps_per_epoch": self.steps_per_epoch,
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
            self._upload_checkpoint(checkpoint_path)
        if self.distributed:
            dist.barrier()

    def _upload_checkpoint(self, checkpoint_path: Path) -> None:
        if self.hf_api is None or self.config.hf_repo is None:
            return
        future = self.hf_api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=f"checkpoints/{checkpoint_path.name}",
            repo_id=self.config.hf_repo,
            repo_type="model",
            commit_message=f"Upload codec {checkpoint_path.stem}",
            run_as_future=True,
        )
        self.hf_uploads.append(future)
        print(f"Queued Hugging Face upload: {checkpoint_path.name}", flush=True)

    def _finish_uploads(self) -> None:
        for future in self.hf_uploads:
            print(f"Hugging Face upload complete: {future.result()}", flush=True)

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

    def _log_train(
        self,
        step: int,
        losses: dict[str, Tensor],
        *,
        data_wait_seconds: float,
        maximum_data_wait_seconds: float,
        measured_steps: int,
        elapsed_seconds: float,
    ) -> None:
        if self.distributed:
            for value in losses.values():
                dist.all_reduce(value)
                value /= self.world_size
        wait_total = torch.tensor(
            data_wait_seconds, dtype=torch.float64, device=self.device
        )
        wait_max = torch.tensor(
            maximum_data_wait_seconds, dtype=torch.float64, device=self.device
        )
        measured = torch.tensor(measured_steps, dtype=torch.float64, device=self.device)
        elapsed = torch.tensor(elapsed_seconds, dtype=torch.float64, device=self.device)
        if self.distributed:
            dist.all_reduce(wait_total)
            dist.all_reduce(measured)
            dist.all_reduce(wait_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        self.latent_mean.synchronize(self.device)
        self.latent_std.synchronize(self.device)
        mean_wait_ms = (wait_total / measured).item() * 1_000
        maximum_wait_ms = wait_max.item() * 1_000
        throughput = (
            measured_steps * self.config.batch_size / elapsed_seconds
            if not self.distributed
            else measured * self.config.batch_size / elapsed
        )
        throughput_value = (
            throughput if isinstance(throughput, float) else throughput.item()
        )
        if not self.rank:
            epoch = step / self.steps_per_epoch
            print(
                f"step={step}/{self.total_steps} epoch={epoch:.4f} "
                f"loss={losses['loss_total'].item():.5f} "
                f"data_wait_mean_ms={mean_wait_ms:.2f} "
                f"data_wait_max_ms={maximum_wait_ms:.2f} "
                f"clips_per_second={throughput_value:.2f}",
                flush=True,
            )
            if self.wandb:
                self.wandb.log(
                    {
                        **{
                            f"train/{name}": value.item()
                            for name, value in losses.items()
                        },
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        "train/latent_mean": self.latent_mean.value,
                        "train/latent_std": self.latent_std.value,
                        "performance/data_wait_mean_ms": mean_wait_ms,
                        "performance/data_wait_max_ms": maximum_wait_ms,
                        "performance/clips_per_second": throughput_value,
                        "progress/epoch": epoch,
                    },
                    step=step,
                )

    def run(self) -> None:
        self.model.train()
        self.raw_model.encoder.dinov3.eval()
        step = 0
        wait_total = 0.0
        wait_max = 0.0
        measured_steps = 0
        interval_started = time.monotonic()
        iterator = iter(self.train_loader)
        try:
            while step < self.total_steps:
                wait_started = time.monotonic()
                video = next(iterator)
                data_wait = time.monotonic() - wait_started
                wait_total += data_wait
                wait_max = max(wait_max, data_wait)
                measured_steps += 1

                step += 1
                train_losses = self._train_step(video)
                should_log = (
                    step <= 10
                    or step % self.config.log_every == 0
                    or step == self.total_steps
                )
                if should_log:
                    now = time.monotonic()
                    self._log_train(
                        step,
                        train_losses,
                        data_wait_seconds=wait_total,
                        maximum_data_wait_seconds=wait_max,
                        measured_steps=measured_steps,
                        elapsed_seconds=now - interval_started,
                    )
                    wait_total = 0.0
                    wait_max = 0.0
                    measured_steps = 0
                    interval_started = now

                epoch_finished = step % self.steps_per_epoch == 0
                training_finished = step == self.total_steps
                if epoch_finished or training_finished:
                    self._evaluate_and_log(step)
                    self._save(step)
        finally:
            if not self.rank:
                self._finish_uploads()
            if self.wandb:
                self.wandb.finish()
            if self.distributed:
                dist.destroy_process_group()


def _loaders(
    config: TrainConfig, *, rank: int, world_size: int
) -> tuple[_BatchLoader, _BatchLoader | None, int]:
    if dist.is_initialized():
        payload = [
            committed_archives(
                config.data_dir,
                frames=config.frames,
                stride_frames=config.stride_frames,
                require_index=True,
            )
            if not rank
            else None
        ]
        dist.broadcast_object_list(payload, src=0)
        archives = payload[0]
        assert archives is not None
    else:
        archives = committed_archives(
            config.data_dir,
            frames=config.frames,
            stride_frames=config.stride_frames,
            require_index=True,
        )
    train_archives, eval_archives = split_archives(
        archives, config.train_fraction, config.split_seed
    )
    train_clips = sum(archive.clips or 0 for archive in train_archives)
    eval_clips = sum(archive.clips or 0 for archive in eval_archives)
    global_batch = config.batch_size * world_size
    steps_per_epoch = train_clips // global_batch
    if steps_per_epoch < 1:
        raise ValueError(
            f"Only {train_clips} train clips for global batch {global_batch}"
        )

    train_dataset = TarZstdClipDataset(
        train_archives,
        frames=config.frames,
        stride_frames=config.stride_frames,
        fps=config.fps,
        size=(config.height, config.width),
        training=True,
        rank=rank,
        world_size=world_size,
        seed=config.split_seed,
        spool_bytes=config.spool_mb * 1024**2,
        decoder_threads=config.decoder_threads,
        video_prefetch=config.video_prefetch,
    )
    eval_loader = None
    if not rank:
        eval_dataset = TarZstdClipDataset(
            eval_archives,
            frames=config.frames,
            stride_frames=config.stride_frames,
            fps=config.fps,
            size=(config.height, config.width),
            training=False,
            seed=config.split_seed,
            spool_bytes=config.spool_mb * 1024**2,
            decoder_threads=config.decoder_threads,
            video_prefetch=config.video_prefetch,
        )
        eval_loader = _data_loader(config, eval_dataset, training=False)
        print(
            f"{len(train_archives)} train archives / {train_clips:,} clips; "
            f"{len(eval_archives)} eval archives / {eval_clips:,} clips; "
            f"{steps_per_epoch:,} steps/epoch at global batch {global_batch}; "
            f"clips are {config.frames / config.fps:g}s with "
            f"{config.stride_frames / config.fps:g}s stride",
            flush=True,
        )
    return (
        _data_loader(config, train_dataset, training=True),
        eval_loader,
        steps_per_epoch,
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
        batches_ahead=config.batch_prefetch,
    )


@dataclass(frozen=True)
class _LoaderFailure:
    error: BaseException


_LOADER_END = object()


class _BatchLoader:
    """Collate in the rank process and keep complete pinned batches ahead of the GPU."""

    def __init__(
        self,
        samples: DataLoader[Tensor],
        *,
        batch_size: int,
        drop_last: bool,
        pin_memory: bool,
        batches_ahead: int,
    ) -> None:
        self.samples = samples
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.batches_ahead = batches_ahead

    def __iter__(self) -> Iterator[Tensor]:
        sample_iterator = iter(self.samples)
        if self.batches_ahead == 1:
            yield from self._batches(sample_iterator)
            return

        ready: queue.Queue[Tensor | _LoaderFailure | object] = queue.Queue(
            maxsize=self.batches_ahead
        )
        stop = threading.Event()

        def put(item: Tensor | _LoaderFailure | object) -> bool:
            while not stop.is_set():
                try:
                    ready.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def produce() -> None:
            try:
                for batch in self._batches(sample_iterator):
                    if not put(batch):
                        return
            except BaseException as error:
                put(_LoaderFailure(error))
            finally:
                put(_LOADER_END)

        producer = threading.Thread(
            target=produce, name="codec-batch-producer", daemon=True
        )
        producer.start()
        try:
            while True:
                item = ready.get()
                if item is _LOADER_END:
                    return
                if isinstance(item, _LoaderFailure):
                    raise item.error
                assert isinstance(item, Tensor)
                yield item
        finally:
            stop.set()
            producer.join(timeout=5)

    def _batches(self, samples: Iterator[Tensor]) -> Iterator[Tensor]:
        batch: list[Tensor] = []
        for sample in samples:
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
        stride_frames=config.stride_frames,
        fps=config.fps,
        size=(config.height, config.width),
        training=True,
        seed=config.split_seed,
        spool_bytes=config.spool_mb * 1024**2,
        decoder_threads=config.decoder_threads,
        video_prefetch=config.video_prefetch,
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
