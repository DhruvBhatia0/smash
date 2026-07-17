"""Train the MIRA-aligned 1B world model for exactly four full data epochs."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from core.training.codec.optimization import ModelEMA

from .full_data import LatentShardDataset, load_evaluation
from .transformer import DiffusionTransformer, flow_matching_prediction


@dataclass(frozen=True)
class Config:
    manifest: Path
    output_dir: Path
    epochs: int = 4
    batch_size: int = 16
    workers: int = 6
    prefetch_factor: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 1_000
    log_every: int = 100
    eval_every: int = 5_000
    eval_samples: int = 64
    seed: int = 28
    ema_decay: float = 0.9999
    activation_checkpointing: bool = False
    resume: Path | None = None
    wandb_entity: str = "dhruvbhatia0"
    wandb_project: str = "smash-d-transformer-full"
    wandb_name: str = "mira-1b-full-4epochs"
    wandb_mode: str = "online"

    @classmethod
    def from_cli(cls) -> "Config":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("manifest", type=Path)
        parser.add_argument("output_dir", type=Path)
        for name in (
            "epochs",
            "batch_size",
            "workers",
            "prefetch_factor",
            "warmup_steps",
            "log_every",
            "eval_every",
            "eval_samples",
            "seed",
        ):
            parser.add_argument(
                f"--{name.replace('_', '-')}", type=int, default=getattr(cls, name)
            )
        parser.add_argument("--learning-rate", type=float, default=cls.learning_rate)
        parser.add_argument("--ema-decay", type=float, default=cls.ema_decay)
        parser.add_argument(
            "--activation-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=cls.activation_checkpointing,
        )
        parser.add_argument("--resume", type=Path)
        parser.add_argument("--wandb-entity", default=cls.wandb_entity)
        parser.add_argument("--wandb-project", default=cls.wandb_project)
        parser.add_argument("--wandb-name", default=cls.wandb_name)
        parser.add_argument(
            "--wandb-mode",
            choices=("online", "offline", "disabled"),
            default=cls.wandb_mode,
        )
        config = cls(**vars(parser.parse_args()))
        if config.epochs != 4:
            raise ValueError("This production recipe is locked to exactly four epochs")
        if min(
            config.batch_size,
            config.workers,
            config.prefetch_factor,
            config.log_every,
            config.eval_every,
            config.eval_samples,
        ) < 1:
            raise ValueError("Batch, worker, logging, and evaluation counts must be positive")
        return config


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world_size > 1:
            dist.init_process_group("nccl")
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        torch.manual_seed(config.seed)
        torch.set_float32_matmul_precision("high")

        self.manifest = json.loads(config.manifest.read_text())
        self.root = config.manifest.parent
        self.train_shards = [
            shard for shard in self.manifest["shards"] if shard["split"] == "train"
        ]
        self.eval_shards = [
            shard for shard in self.manifest["shards"] if shard["split"] == "eval"
        ]

        model = DiffusionTransformer(
            use_clean_past=True,
            activation_checkpointing=config.activation_checkpointing,
        ).to(self.device)
        self.ema = ModelEMA(model, config.ema_decay)
        self.model: nn.Module = (
            DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
            if self.world_size > 1
            else model
        )
        self.raw_model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0.1,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min(step / config.warmup_steps, 1.0)
            if config.warmup_steps
            else 1.0,
        )
        self.start_epoch = 0
        self.step = 0
        if config.resume:
            self._resume(config.resume)

        self.eval_latents: Tensor | None = None
        self.eval_actions: Tensor | None = None
        self.eval_noise: Tensor | None = None
        self.eval_time: Tensor | None = None
        if self.rank == 0:
            latents, actions = load_evaluation(
                self.root, self.eval_shards, config.eval_samples
            )
            self.eval_latents = latents.to(self.device)
            self.eval_actions = actions.to(self.device)
            generator = torch.Generator(device=self.device).manual_seed(config.seed + 1)
            self.eval_noise = torch.randn(
                self.eval_latents.shape,
                dtype=self.eval_latents.dtype,
                device=self.device,
                generator=generator,
            )
            base = torch.linspace(0.025, 0.975, 20, device=self.device)
            self.eval_time = torch.stack(
                [base.roll(3 * index) for index in range(config.eval_samples)]
            )
        self.wandb_run: Any | None = None
        self._start_wandb()

    def _start_wandb(self) -> None:
        if self.rank or self.config.wandb_mode == "disabled":
            return
        import wandb

        run_id_path = self.config.output_dir / "wandb-run-id.txt"
        run_id = run_id_path.read_text().strip() if run_id_path.is_file() else wandb.util.generate_id()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(run_id + "\n")
        values = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self.config).items()
        }
        values.update(
            {
                "model_parameters": sum(p.numel() for p in self.raw_model.parameters()),
                "train_unique_clips": self.manifest["train_samples"],
                "eval_archive": self.manifest["eval_archive"],
                "global_batch_size": self.config.batch_size * self.world_size,
                "action_representation": "three_ordered_60hz_states_per_20hz_transition",
                "use_clean_past": True,
            }
        )
        self.wandb_run = wandb.init(
            entity=self.config.wandb_entity,
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            id=run_id,
            resume="allow",
            mode=self.config.wandb_mode,
            config=values,
            tags=("mira", "full-data", "four-epochs"),
        )
        self.wandb_run.define_metric("*", step_metric="progress/step")
        print(f"Weights & Biases: {self.wandb_run.url}", flush=True)

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def _loader(self, epoch: int) -> tuple[DataLoader, LatentShardDataset]:
        dataset = LatentShardDataset(
            self.root,
            self.train_shards,
            rank=self.rank,
            world_size=self.world_size,
            workers=self.config.workers,
            batch_size=self.config.batch_size,
            epoch=epoch,
            seed=self.config.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )
        return loader, dataset

    @torch.inference_mode()
    def _eval_loss(self, zero_actions: bool = False) -> float:
        assert self.eval_latents is not None
        assert self.eval_actions is not None
        assert self.eval_noise is not None
        assert self.eval_time is not None
        losses = []
        self.raw_model.eval()
        for index in range(len(self.eval_latents)):
            actions = self.eval_actions[index : index + 1]
            if zero_actions:
                actions = torch.zeros_like(actions)
            with self._autocast():
                prediction, target, _ = flow_matching_prediction(
                    self.raw_model,
                    self.eval_latents[index : index + 1],
                    actions,
                    noise=self.eval_noise[index : index + 1],
                    flow_time=self.eval_time[index : index + 1],
                )
            losses.append((prediction.float() - target.float()).square().mean())
        return torch.stack(losses).mean().item()

    @torch.inference_mode()
    def _rollout(self, actions: Tensor) -> Tensor:
        assert self.eval_latents is not None
        latents = self.eval_latents[:2]
        horizon, denoising_steps = 6, 10
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 2)
        noise = torch.randn(
            (2, horizon, *latents.shape[2:]),
            device=self.device,
            dtype=latents.dtype,
            generator=generator,
        )
        generated = [latents[:, 0]]
        for target_index in range(1, horizon + 1):
            current = noise[:, target_index - 1].clone()
            for diffusion_step in range(denoising_steps):
                prefix = torch.stack([*generated, current], dim=1)
                time = prefix.shape[1]
                flow_time = torch.ones(2, time, device=self.device)
                flow_time[:, -1] = diffusion_step / denoising_steps
                with self._autocast():
                    velocity = self.raw_model(
                        prefix,
                        actions[:, : 2 * time - 1],
                        flow_time,
                    )[:, -1]
                current = current + velocity.float() / denoising_steps
            generated.append(current)
        return torch.stack(generated[1:], dim=1)

    def _evaluate(self, *, rollout: bool) -> dict[str, float]:
        if self.rank:
            return {}
        assert self.eval_actions is not None and self.eval_latents is not None
        with self.ema.average_parameters():
            metrics = {"eval/loss": self._eval_loss()}
            if rollout:
                correct = self._rollout(self.eval_actions[:2])
                zero = self._rollout(torch.zeros_like(self.eval_actions[:2]))
                target = self.eval_latents[:2, 1:7].float()
                correct_error = (correct - target).square().mean((0, 2, 3, 4))
                zero_error = (zero - target).square().mean((0, 2, 3, 4))
                metrics.update(
                    {
                        "eval/rollout_mse_h01": correct_error[0].item(),
                        "eval/rollout_mse_h06": correct_error[-1].item(),
                        "eval/action_zero_delta_h06": (
                            zero_error[-1] - correct_error[-1]
                        ).item(),
                    }
                )
        self.raw_model.train()
        self._log(metrics)
        return metrics

    def _log(self, metrics: dict[str, float]) -> None:
        if self.rank:
            return
        print(json.dumps({"step": self.step, **metrics}, sort_keys=True), flush=True)
        if self.wandb_run:
            self.wandb_run.log({"progress/step": self.step, **metrics})
        temporary = self.config.output_dir / "status.json.partial"
        temporary.write_text(json.dumps({"step": self.step, **metrics}, indent=2) + "\n")
        temporary.replace(self.config.output_dir / "status.json")

    def _sync_eval(self, *, rollout: bool) -> None:
        if dist.is_initialized():
            dist.barrier()
        self._evaluate(rollout=rollout)
        if dist.is_initialized():
            dist.barrier()

    def run(self) -> None:
        self.model.train()
        torch.cuda.reset_peak_memory_stats()
        self._sync_eval(rollout=True)
        global_examples = self.step * self.config.batch_size * self.world_size
        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                loader, dataset = self._loader(epoch)
                steps_per_epoch = len(dataset) // self.config.batch_size
                interval_loss = 0.0
                interval_wait = 0.0
                interval_count = 0
                interval_started = time.monotonic()
                iterator = iter(loader)
                for epoch_step in range(1, steps_per_epoch + 1):
                    wait_started = time.monotonic()
                    latents, actions = next(iterator)
                    interval_wait += time.monotonic() - wait_started
                    latents = latents.to(self.device, non_blocking=True)
                    actions = actions.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad(set_to_none=True)
                    with self._autocast():
                        prediction, target, _ = flow_matching_prediction(
                            self.model, latents, actions
                        )
                        loss = (prediction.float() - target.float()).square().mean()
                    loss.backward()
                    self.step += 1
                    should_log = self.step == 1 or self.step % self.config.log_every == 0
                    grad_norm = (
                        nn.utils.clip_grad_norm_(self.model.parameters(), math.inf)
                        if should_log
                        else None
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.ema.step()
                    interval_loss += loss.item() * len(latents)
                    interval_count += len(latents)
                    global_examples += len(latents) * self.world_size

                    if should_log:
                        elapsed = time.monotonic() - interval_started
                        values = torch.tensor(
                            [interval_loss, interval_wait, interval_count],
                            device=self.device,
                            dtype=torch.float64,
                        )
                        elapsed_value = torch.tensor(elapsed, device=self.device)
                        norm_value = grad_norm.detach().float()  # type: ignore[union-attr]
                        if dist.is_initialized():
                            dist.all_reduce(values)
                            dist.all_reduce(elapsed_value, op=dist.ReduceOp.MAX)
                            dist.all_reduce(norm_value)
                            norm_value /= self.world_size
                        metrics = {
                            "train/loss": (values[0] / values[2]).item(),
                            "optimization/learning_rate": self.optimizer.param_groups[0]["lr"],
                            "optimization/grad_norm": norm_value.item(),
                            "progress/epoch": epoch + epoch_step / steps_per_epoch,
                            "progress/examples": float(global_examples),
                            "performance/clips_per_second": (
                                values[2] / elapsed_value
                            ).item(),
                            "performance/data_wait_ms": (
                                1_000 * values[1] / values[2]
                            ).item(),
                        }
                        self._log(metrics)
                        interval_loss = interval_wait = 0.0
                        interval_count = 0
                        interval_started = time.monotonic()

                    if self.step % self.config.eval_every == 0:
                        self._sync_eval(rollout=False)
                        interval_started = time.monotonic()

                self._sync_eval(rollout=True)
                self._save(epoch + 1, dataset.padding_samples)
        finally:
            if self.wandb_run:
                self.wandb_run.finish()
            if dist.is_initialized():
                dist.destroy_process_group()

    def _save(self, epoch: int, padding_samples: int) -> None:
        if dist.is_initialized():
            dist.barrier()
        if self.rank == 0:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            destination = self.config.output_dir / f"checkpoint-epoch-{epoch:02d}.pt"
            temporary = destination.with_suffix(".pt.partial")
            torch.save(
                {
                    "model": self.raw_model.state_dict(),
                    "ema": self.ema.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": epoch,
                    "step": self.step,
                    "padding_samples_per_epoch": padding_samples,
                    "config": asdict(self.config),
                },
                temporary,
            )
            temporary.replace(destination)
            for old in self.config.output_dir.glob("checkpoint-epoch-*.pt"):
                if old != destination:
                    old.unlink()
            self._log({"progress/checkpoint_epoch": float(epoch)})
        if dist.is_initialized():
            dist.barrier()

    def _resume(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(checkpoint["model"])
        self.ema.count = checkpoint["ema"]["count"]
        averages = checkpoint["ema"]["averages"]
        for name, value in zip(self.ema.names, self.ema.averages, strict=True):
            value.copy_(averages[name])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = int(checkpoint["epoch"])
        self.step = int(checkpoint["step"])


if __name__ == "__main__":
    Trainer(Config.from_cli()).run()
