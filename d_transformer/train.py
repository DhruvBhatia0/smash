"""Train with ``torchrun --nproc-per-node=8 -m d_transformer.train TRAIN EVAL CODEC``."""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from core.training.codec.codec import CodecEncoder
from core.training.codec.train import VideoFolderDataset

from .transformer import DiffusionTransformer, flow_matching_loss


@dataclass(frozen=True)
class TrainConfig:
    train_dir: Path
    eval_dir: Path
    codec_checkpoint: Path
    checkpoint_dir: Path = Path("checkpoints/d_transformer")
    steps: int = 250_000
    save_every: int = 5_000
    log_every: int = 10
    batch_size: int = 1
    workers: int = 6
    learning_rate: float = 1e-4
    frames: int = 40
    height: int = 288
    width: int = 512
    model_width: int = 2048
    model_depth: int = 16
    attention_heads: int = 16
    key_value_heads: int = 4
    time_attention_every: int = 4
    activation_checkpointing: bool = True
    compile: bool = True
    wandb_project: str = "smash-d-transformer"
    wandb_name: str | None = None
    wandb_mode: str = "online"

    @classmethod
    def from_cli(cls) -> TrainConfig:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("train_dir", type=Path)
        parser.add_argument("eval_dir", type=Path)
        parser.add_argument("codec_checkpoint", type=Path)
        parser.add_argument("--checkpoint-dir", type=Path, default=cls.checkpoint_dir)
        parser.add_argument("--steps", type=int, default=cls.steps)
        parser.add_argument("--save-every", type=int, default=cls.save_every)
        parser.add_argument("--log-every", type=int, default=cls.log_every)
        parser.add_argument("--batch-size", type=int, default=cls.batch_size)
        parser.add_argument("--workers", type=int, default=cls.workers)
        parser.add_argument("--learning-rate", type=float, default=cls.learning_rate)
        parser.add_argument("--model-width", type=int, default=cls.model_width)
        parser.add_argument("--model-depth", type=int, default=cls.model_depth)
        parser.add_argument("--attention-heads", type=int, default=cls.attention_heads)
        parser.add_argument("--key-value-heads", type=int, default=cls.key_value_heads)
        parser.add_argument(
            "--time-attention-every", type=int, default=cls.time_attention_every
        )
        parser.add_argument(
            "--activation-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=cls.activation_checkpointing,
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
        return cls(**vars(parser.parse_args()))


class DiffusionTrainer:
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
        self.train_loader, self.train_sampler = self._loader(
            config.train_dir, training=True
        )
        self.eval_loader, _ = self._loader(config.eval_dir, training=False)
        self.codec_encoder = self._load_codec_encoder().to(self.device)

        self.raw_model = DiffusionTransformer(
            width=config.model_width,
            depth=config.model_depth,
            heads=config.attention_heads,
            kv_heads=config.key_value_heads,
            time_attention_every=config.time_attention_every,
            activation_checkpointing=config.activation_checkpointing,
        ).to(self.device)
        self.model: nn.Module = self.raw_model
        if self.distributed:
            self.model = DistributedDataParallel(
                self.raw_model,
                device_ids=[self.local_rank],
                broadcast_buffers=False,
            )
        if config.compile:
            self.codec_encoder.compile()
            self.model.compile()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0.1,
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1 / 1_000,
            total_iters=min(1_000, config.steps),
        )
        self.mean = torch.tensor(self.DINO_MEAN, device=self.device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(self.DINO_STD, device=self.device).view(1, 1, 3, 1, 1)
        self.wandb: Any | None = None
        self._start_wandb()

    def _load_codec_encoder(self) -> CodecEncoder:
        encoder = CodecEncoder(desired_hidden_states=list(self.DINO_LAYERS))
        checkpoint = torch.load(
            self.config.codec_checkpoint, map_location="cpu", weights_only=False
        )
        encoder_state = {
            name.removeprefix("encoder."): value
            for name, value in checkpoint["model"].items()
            if name.startswith("encoder.")
        }
        encoder.load_state_dict(encoder_state)
        encoder.requires_grad_(False)
        return encoder.eval()

    def _loader(
        self, folder: Path, training: bool
    ) -> tuple[DataLoader[Tensor], DistributedSampler[Tensor] | None]:
        dataset = VideoFolderDataset(
            folder,
            frames=self.config.frames,
            size=(self.config.height, self.config.width),
        )
        sampler = (
            DistributedSampler(dataset, shuffle=training) if self.distributed else None
        )
        worker_options = (
            {"persistent_workers": True, "prefetch_factor": 2}
            if self.config.workers
            else {}
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
            raise ValueError(
                "The training folder must contain at least one full global batch"
            )
        return loader, sampler

    def _start_wandb(self) -> None:
        if self.rank or self.config.wandb_mode == "disabled":
            return
        import wandb

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

    def _encode(self, video: Tensor) -> Tensor:
        video = video.to(self.device, non_blocking=True).float().div_(255)
        with torch.no_grad(), self._autocast():
            # b t 3 h w -> b (t / 2) 32 (h / 32) (w / 32)
            return self.codec_encoder((video - self.mean) / self.std)

    def _train_step(self, video: Tensor) -> Tensor:
        latents = self._encode(video)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            loss = flow_matching_loss(self.model, latents)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach()

    @torch.inference_mode()
    def _evaluate(self) -> float:
        self.model.eval()
        totals = torch.zeros(2, device=self.device)
        for video in self.eval_loader:
            latents = self._encode(video)
            with self._autocast():
                loss = flow_matching_loss(self.model, latents)
            totals[0] += loss * video.shape[0]
            totals[1] += video.shape[0]
        if self.distributed:
            dist.all_reduce(totals)
        self.model.train()
        return (totals[0] / totals[1]).item()

    def _save_and_log(self, step: int) -> None:
        eval_loss = self._evaluate()
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
                    "eval_loss": eval_loss,
                },
                checkpoint_path,
            )
            if self.wandb:
                self.wandb.log({"eval/loss": eval_loss}, step=step)
        if self.distributed:
            dist.barrier()

    def run(self) -> None:
        self.model.train()
        step = 0
        try:
            for epoch in count():
                if self.train_sampler:
                    self.train_sampler.set_epoch(epoch)
                for video in self.train_loader:
                    step += 1
                    train_loss = self._train_step(video)
                    if step % self.config.log_every == 0:
                        if self.distributed:
                            dist.all_reduce(train_loss)
                            train_loss /= self.world_size
                        if self.wandb:
                            self.wandb.log(
                                {
                                    "train/loss": train_loss.item(),
                                    "train/learning_rate": self.scheduler.get_last_lr()[
                                        0
                                    ],
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
    DiffusionTrainer(TrainConfig.from_cli()).run()
