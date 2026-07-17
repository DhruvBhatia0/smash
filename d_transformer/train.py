"""Train a cached-latent MIRA ablation with fixed, action-sensitive evaluation."""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, TensorDataset

from .transformer import DiffusionTransformer, flow_matching_prediction


ACTION_REPRESENTATIONS = (
    "ordered_60hz",
    "summary_60hz",
    "last_20hz",
    "mean_20hz",
    "none",
)
TAU_BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))


@dataclass(frozen=True)
class TrainConfig:
    cache: Path
    output: Path = Path("d-transformer-ablation.pt")
    steps: int = 1_000
    learning_rate: float = 1e-4
    warmup_steps: int = 1_000
    log_every: int = 20
    eval_every: int = 100
    batch_size: int = 1
    eval_batch_size: int = 2
    train_samples: int = 16
    eval_samples: int = 8
    rollout_samples: int = 2
    rollout_horizon: int = 6
    rollout_steps: int = 8
    seed: int = 28
    model_width: int = 2_048
    model_depth: int = 16
    attention_heads: int = 16
    key_value_heads: int = 4
    time_attention_every: int = 4
    action_representation: str = "ordered_60hz"
    use_clean_past: bool = True
    ema_decay: float = 0.9999
    activation_checkpointing: bool = False
    compile: bool = False
    wandb_project: str = "smash-d-transformer"
    wandb_entity: str | None = "dhruvbhatia0"
    wandb_name: str | None = None
    wandb_group: str | None = None
    wandb_tags: str = "mira,action-conditioning,data-efficiency"
    wandb_mode: str = "online"

    @classmethod
    def from_cli(cls) -> "TrainConfig":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("cache", type=Path)
        parser.add_argument("--output", type=Path, default=cls.output)
        for name in (
            "steps",
            "warmup_steps",
            "log_every",
            "eval_every",
            "batch_size",
            "eval_batch_size",
            "train_samples",
            "eval_samples",
            "rollout_samples",
            "rollout_horizon",
            "rollout_steps",
            "seed",
            "model_width",
            "model_depth",
            "attention_heads",
            "key_value_heads",
            "time_attention_every",
        ):
            parser.add_argument(
                f"--{name.replace('_', '-')}", type=int, default=getattr(cls, name)
            )
        parser.add_argument("--learning-rate", type=float, default=cls.learning_rate)
        parser.add_argument("--ema-decay", type=float, default=cls.ema_decay)
        parser.add_argument(
            "--action-representation",
            choices=ACTION_REPRESENTATIONS,
            default=cls.action_representation,
        )
        parser.add_argument(
            "--activation-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=cls.activation_checkpointing,
        )
        parser.add_argument(
            "--use-clean-past",
            action=argparse.BooleanOptionalAction,
            default=cls.use_clean_past,
        )
        parser.add_argument(
            "--compile", action=argparse.BooleanOptionalAction, default=cls.compile
        )
        parser.add_argument("--wandb-project", default=cls.wandb_project)
        parser.add_argument("--wandb-entity", default=cls.wandb_entity)
        parser.add_argument("--wandb-name")
        parser.add_argument("--wandb-group")
        parser.add_argument("--wandb-tags", default=cls.wandb_tags)
        parser.add_argument(
            "--wandb-mode",
            choices=("online", "offline", "disabled"),
            default=cls.wandb_mode,
        )
        config = cls(**vars(parser.parse_args()))
        counts = (
            config.steps,
            config.log_every,
            config.eval_every,
            config.batch_size,
            config.eval_batch_size,
            config.train_samples,
            config.eval_samples,
            config.rollout_samples,
            config.rollout_horizon,
            config.rollout_steps,
        )
        if min(counts) < 1:
            raise ValueError("Step and batch counts must be positive")
        if config.action_representation not in ACTION_REPRESENTATIONS:
            raise ValueError("Unknown action representation")
        return config


def action_representation(actions: Tensor, kind: str) -> Tensor:
    if kind == "ordered_60hz":
        return actions
    if kind == "last_20hz":
        return actions[:, :, -1:].expand_as(actions)
    if kind == "mean_20hz":
        return actions.mean(dim=2, keepdim=True).expand_as(actions)
    if kind == "summary_60hz":
        return torch.cat(
            (actions.mean(dim=2, keepdim=True), actions[:, :, :1], actions[:, :, -1:]),
            dim=2,
        )
    if kind == "none":
        return torch.zeros_like(actions)
    raise ValueError(f"Unknown action representation: {kind}")


def _module_norm(module: nn.Module, *, gradients: bool = False) -> Tensor:
    values = [
        (parameter.grad if gradients else parameter).detach().float().norm()
        for parameter in module.parameters()
        if not gradients or parameter.grad is not None
    ]
    return torch.stack(values).norm() if values else torch.tensor(0.0)


def _tau_metrics(prefix: str, errors: Tensor, flow_time: Tensor) -> dict[str, float]:
    metrics = {}
    for low, high in TAU_BINS:
        mask = (flow_time >= low) & (flow_time < high)
        metrics[f"{prefix}/tau_{low:.1f}_{high:.1f}"] = errors[mask].mean().item()
    return metrics


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)
        torch.set_float32_matmul_precision("high")

        cache = torch.load(
            config.cache, map_location="cpu", weights_only=False, mmap=True
        )
        latents, actions = cache["latents"].float(), cache["actions"].float()
        action_differences = (actions[:, :, 1:] - actions[:, :, :-1]).abs()
        self.action_statistics = {
            "action_substep_pair_any_change_fraction": (action_differences > 1e-6)
            .any(-1)
            .float()
            .mean()
            .item(),
            "action_transition_player_any_change_fraction": (action_differences > 1e-6)
            .any(-1)
            .any(2)
            .float()
            .mean()
            .item(),
            "action_transition_any_change_fraction": (action_differences > 1e-6)
            .any(-1)
            .any(2)
            .any(2)
            .float()
            .mean()
            .item(),
            "action_analog_pair_any_change_fraction": (
                action_differences[..., :8] > 1e-6
            )
            .any(-1)
            .float()
            .mean()
            .item(),
            "action_processed_button_pair_any_change_fraction": (
                action_differences[..., 8:40] > 1e-6
            )
            .any(-1)
            .float()
            .mean()
            .item(),
            "action_physical_button_pair_any_change_fraction": (
                action_differences[..., 40:] > 1e-6
            )
            .any(-1)
            .float()
            .mean()
            .item(),
            "action_last_only_feature_mse": (
                actions - actions[:, :, -1:].expand_as(actions)
            )
            .square()
            .mean()
            .item(),
        }
        required = config.train_samples + config.eval_samples
        if required > len(latents):
            raise ValueError(f"Cache has {len(latents)} samples; run needs {required}")
        self.cache_metadata = {
            key: value
            for key, value in cache.items()
            if key not in {"latents", "actions"}
        }
        self.train_names = cache["samples"][: config.train_samples]
        self.eval_names = cache["samples"][-config.eval_samples :]
        train_latents, train_actions = (
            latents[: config.train_samples],
            actions[: config.train_samples],
        )
        self.eval_latents = latents[-config.eval_samples :].to(self.device)
        self.eval_actions = actions[-config.eval_samples :].to(self.device)
        generator = torch.Generator().manual_seed(config.seed)
        self.loader = DataLoader(
            TensorDataset(
                train_latents,
                train_actions,
                torch.arange(config.train_samples),
            ),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=self.device.type == "cuda",
        )

        self.model = DiffusionTransformer(
            latent_height=latents.shape[-2],
            latent_width=latents.shape[-1],
            width=config.model_width,
            depth=config.model_depth,
            heads=config.attention_heads,
            kv_heads=config.key_value_heads,
            time_attention_every=config.time_attention_every,
            use_clean_past=config.use_clean_past,
            activation_checkpointing=config.activation_checkpointing,
        ).to(self.device)
        if config.compile:
            self.model.compile()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0.1,
        )
        self.scheduler = (
            torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1 / config.warmup_steps,
                total_iters=config.warmup_steps,
            )
            if config.warmup_steps
            else None
        )
        self.ema = AveragedModel(
            self.model,
            multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay),
            use_buffers=True,
        )
        eval_generator = torch.Generator(device=self.device).manual_seed(
            config.seed + 1
        )
        self.eval_noise = torch.randn(
            self.eval_latents.shape,
            device=self.device,
            generator=eval_generator,
        )
        base_time = torch.linspace(
            0.025,
            0.975,
            self.eval_latents.shape[1],
            device=self.device,
        )
        self.eval_time = torch.stack(
            [base_time.roll(3 * index) for index in range(config.eval_samples)]
        )
        self.wandb: Any | None = None
        self.wandb_run: Any | None = None
        self.eval_table_rows: list[list[Any]] = []
        self.history: list[dict[str, float]] = []
        self._start_wandb()

    def _start_wandb(self) -> None:
        if self.config.wandb_mode == "disabled":
            return
        import wandb

        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self.config).items()
        }
        config.update(
            {
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "latent_shape": tuple(self.eval_latents.shape[1:]),
                "action_shape": tuple(self.eval_actions.shape[1:]),
                "video_frames_per_clip": self.cache_metadata["video_frames"],
                "train_replays": self.train_names,
                "eval_replays": self.eval_names,
                **self.action_statistics,
            }
        )
        self.wandb = wandb
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_name,
            group=self.config.wandb_group,
            tags=[tag for tag in self.config.wandb_tags.split(",") if tag],
            job_type="action-data-efficiency-ablation",
            mode=self.config.wandb_mode,
            config=config,
        )
        if self.wandb_run.url:
            print(f"Weights & Biases: {self.wandb_run.url}", flush=True)

    def _autocast(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )

    def _interventions(self, actions: Tensor) -> dict[str, Tensor]:
        kind = self.config.action_representation
        return {
            "correct": action_representation(actions, kind),
            "transition_roll": action_representation(actions.roll(1, dims=1), kind),
            "microstep_reverse": action_representation(actions.flip(2), kind),
            "player_swap": action_representation(actions.flip(3), kind),
            "last_20hz": action_representation(actions, "last_20hz"),
            "zero": torch.zeros_like(actions),
        }

    @torch.inference_mode()
    def _evaluate(self, model: nn.Module, *, diagnostics: bool) -> dict[str, float]:
        model.eval()
        intervention_errors: dict[str, list[Tensor]] = {}
        correct_predictions, targets = [], []
        for start in range(0, self.config.eval_samples, self.config.eval_batch_size):
            stop = start + self.config.eval_batch_size
            latents = self.eval_latents[start:stop]
            noise = self.eval_noise[start:stop]
            flow_time = self.eval_time[start:stop]
            interventions = self._interventions(self.eval_actions[start:stop])
            if not diagnostics:
                interventions = {"correct": interventions["correct"]}
            for name, inputs in interventions.items():
                with self._autocast():
                    prediction, target, _ = flow_matching_prediction(
                        model,
                        latents,
                        inputs,
                        noise=noise,
                        flow_time=flow_time,
                    )
                errors = (prediction.float() - target.float()).square().mean((2, 3, 4))
                intervention_errors.setdefault(name, []).append(errors.cpu())
                if name == "correct":
                    correct_predictions.append(prediction.float().cpu())
                    targets.append(target.float().cpu())

        errors = {name: torch.cat(rows) for name, rows in intervention_errors.items()}
        correct = errors["correct"]
        prediction = torch.cat(correct_predictions)
        target = torch.cat(targets)
        metrics = {
            "loss": correct.mean().item(),
            "loss_std_across_replays": correct.mean(1).std(correction=0).item(),
            "zero_velocity_loss": target.square().mean().item(),
            "loss_over_zero_velocity": correct.mean()
            .div(target.square().mean())
            .item(),
            "velocity_cosine": F.cosine_similarity(
                prediction.flatten(2), target.flatten(2), dim=2
            )
            .mean()
            .item(),
            "prediction_velocity_norm": prediction.flatten(2).norm(dim=2).mean().item(),
            "target_velocity_norm": target.flatten(2).norm(dim=2).mean().item(),
        }
        metrics.update(_tau_metrics("loss_by_noise", correct, self.eval_time.cpu()))
        metrics.update(
            {
                f"loss_by_latent_position/t_{index:02d}": value.item()
                for index, value in enumerate(correct.mean(0))
            }
        )
        metrics.update(
            {
                f"loss_by_replay/replay_{index:02d}": value.item()
                for index, value in enumerate(correct.mean(1))
            }
        )
        for name, values in errors.items():
            if name == "correct":
                continue
            delta = values.mean() - correct.mean()
            metrics[f"action_intervention/{name}_loss"] = values.mean().item()
            metrics[f"action_intervention/{name}_delta"] = delta.item()
            metrics[f"action_intervention/{name}_relative_delta"] = (
                delta / correct.mean()
            ).item()
            intervention_bins = _tau_metrics("unused", values, self.eval_time.cpu())
            correct_bins = _tau_metrics("unused", correct, self.eval_time.cpu())
            for key in intervention_bins:
                label = key.rsplit("/", 1)[-1]
                metrics[f"action_intervention_by_noise/{name}_{label}_delta"] = (
                    intervention_bins[key] - correct_bins[key]
                )
        return metrics

    def _log(self, values: dict[str, float], step: int) -> None:
        row = {"step": step, **values}
        print(json.dumps(row, sort_keys=True), flush=True)
        if self.wandb_run:
            self.wandb_run.log(values, step=step)

    @torch.inference_mode()
    def _rollout_metrics(self) -> dict[str, float]:
        samples = min(self.config.rollout_samples, self.config.eval_samples)
        horizon = min(
            self.config.rollout_horizon,
            self.eval_latents.shape[1] - 1,
        )
        latents = self.eval_latents[:samples]
        actions = self.eval_actions[:samples]
        generator = torch.Generator(device=self.device).manual_seed(
            self.config.seed + 2
        )
        initial_noise = torch.randn(
            (samples, horizon, *latents.shape[2:]),
            device=self.device,
            generator=generator,
        )
        model = self.ema.module.eval()
        rollouts: dict[str, Tensor] = {}
        for name, inputs in self._interventions(actions).items():
            generated = [latents[:, 0]]
            for target_index in range(1, horizon + 1):
                current = initial_noise[:, target_index - 1].clone()
                for diffusion_step in range(self.config.rollout_steps):
                    flow_time = torch.ones(
                        samples,
                        len(generated) + 1,
                        device=self.device,
                    )
                    flow_time[:, -1] = diffusion_step / self.config.rollout_steps
                    prefix = torch.stack([*generated, current], dim=1)
                    action_stop = 2 * prefix.shape[1] - 1
                    with self._autocast():
                        velocity = model(
                            prefix,
                            inputs[:, :action_stop],
                            flow_time,
                        )[:, -1]
                    current = current + velocity.float() / self.config.rollout_steps
                generated.append(current)
            rollouts[name] = torch.stack(generated[1:], dim=1).cpu()

        target = latents[:, 1 : horizon + 1].float().cpu()
        correct = rollouts["correct"]
        correct_errors = (correct - target).square().mean((0, 2, 3, 4))
        metrics: dict[str, float] = {}
        for name, rollout in rollouts.items():
            errors = (rollout - target).square().mean((0, 2, 3, 4))
            cosine = F.cosine_similarity(
                rollout.flatten(2), target.flatten(2), dim=2
            ).mean(0)
            for index, (error, similarity) in enumerate(zip(errors, cosine), start=1):
                metrics[f"rollout/{name}_mse_h{index:02d}"] = error.item()
                metrics[f"rollout/{name}_cosine_h{index:02d}"] = similarity.item()
            if name != "correct":
                metrics[f"rollout/{name}_final_error_delta"] = (
                    errors[-1] - correct_errors[-1]
                ).item()
                metrics[f"rollout/{name}_final_divergence_from_correct"] = (
                    (rollout[:, -1] - correct[:, -1]).square().mean().item()
                )
        return metrics

    def _evaluate_and_log(self, step: int) -> None:
        started = time.monotonic()
        raw = self._evaluate(self.model, diagnostics=False)
        ema = self._evaluate(self.ema.module, diagnostics=True)
        metrics = {
            "eval/raw_loss": raw["loss"],
            **{f"eval/{name}": value for name, value in ema.items()},
            "performance/eval_seconds": time.monotonic() - started,
        }
        self.history.append({"step": float(step), **metrics})
        for name in self._interventions(self.eval_actions[:1]):
            loss = (
                ema["loss"]
                if name == "correct"
                else ema[f"action_intervention/{name}_loss"]
            )
            self.eval_table_rows.append([step, name, loss, loss - ema["loss"]])
        self._log(metrics, step)
        self.model.train()

    def run(self) -> None:
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._evaluate_and_log(0)
        iterator = iter(self.loader)
        interval_started = time.monotonic()
        interval_losses: list[float] = []
        interval_errors: list[Tensor] = []
        interval_times: list[Tensor] = []
        interval_grad_norms: list[float] = []
        interval_action_grad_norms: list[float] = []
        samples_seen: set[int] = set()
        examples_seen = 0

        try:
            for step in range(1, self.config.steps + 1):
                try:
                    latents, actions, indices = next(iterator)
                except StopIteration:
                    iterator = iter(self.loader)
                    latents, actions, indices = next(iterator)
                samples_seen.update(indices.tolist())
                examples_seen += len(latents)
                latents = latents.to(self.device, non_blocking=True)
                actions = action_representation(
                    actions.to(self.device, non_blocking=True),
                    self.config.action_representation,
                )

                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    prediction, target, flow_time = flow_matching_prediction(
                        self.model, latents, actions
                    )
                    frame_errors = (
                        (prediction.float() - target.float()).square().mean((2, 3, 4))
                    )
                    loss = frame_errors.mean()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), math.inf)
                action_grad_norm = _module_norm(self.model.actions, gradients=True)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.ema.update_parameters(self.model)

                interval_losses.append(loss.item())
                interval_errors.append(frame_errors.detach().cpu())
                interval_times.append(flow_time.detach().cpu())
                interval_grad_norms.append(grad_norm.item())
                interval_action_grad_norms.append(action_grad_norm.item())

                should_log = (
                    step == 1
                    or step % self.config.log_every == 0
                    or step == self.config.steps
                )
                if should_log:
                    elapsed = time.monotonic() - interval_started
                    losses = torch.tensor(interval_losses)
                    errors = torch.cat(interval_errors)
                    times = torch.cat(interval_times)
                    parameter_norm = _module_norm(self.model).item()
                    grad_norm_mean = torch.tensor(interval_grad_norms).mean().item()
                    measured = len(interval_losses) * self.config.batch_size
                    metrics = {
                        "train/loss_mean": losses.mean().item(),
                        "train/loss_std": losses.std(correction=0).item(),
                        "train/loss_min": losses.min().item(),
                        "train/loss_max": losses.max().item(),
                        **_tau_metrics("train/loss_by_noise", errors, times),
                        "optimization/learning_rate": self.optimizer.param_groups[0][
                            "lr"
                        ],
                        "optimization/grad_norm": grad_norm_mean,
                        "optimization/grad_norm_max": max(interval_grad_norms),
                        "optimization/action_encoder_grad_norm": torch.tensor(
                            interval_action_grad_norms
                        )
                        .mean()
                        .item(),
                        "optimization/parameter_norm": parameter_norm,
                        "optimization/raw_step_to_weight": (
                            self.optimizer.param_groups[0]["lr"]
                            * grad_norm_mean
                            / parameter_norm
                        ),
                        "data/examples_seen": float(examples_seen),
                        "data/equivalent_epochs": examples_seen
                        / self.config.train_samples,
                        "data/unique_replays_seen": float(len(samples_seen)),
                        "data/replay_coverage": len(samples_seen)
                        / self.config.train_samples,
                        "data/video_frames_seen": float(
                            examples_seen * self.cache_metadata["video_frames"]
                        ),
                        "data/source_video_seconds_seen": float(
                            examples_seen * self.cache_metadata["video_frames"] / 20
                        ),
                        "data/player_micro_inputs_seen": float(
                            examples_seen
                            * self.eval_actions.shape[1]
                            * self.eval_actions.shape[2]
                            * self.eval_actions.shape[3]
                        ),
                        "performance/steps_per_second": len(interval_losses) / elapsed,
                        "performance/clips_per_second": measured / elapsed,
                        "performance/video_frames_per_second": (
                            measured * self.cache_metadata["video_frames"] / elapsed
                        ),
                        "performance/step_ms": 1_000 * elapsed / len(interval_losses),
                    }
                    if self.device.type == "cuda":
                        metrics.update(
                            {
                                "performance/gpu_peak_allocated_gib": torch.cuda.max_memory_allocated()
                                / 1024**3,
                                "performance/gpu_peak_reserved_gib": torch.cuda.max_memory_reserved()
                                / 1024**3,
                            }
                        )
                    self._log(metrics, step)
                    interval_losses.clear()
                    interval_errors.clear()
                    interval_times.clear()
                    interval_grad_norms.clear()
                    interval_action_grad_norms.clear()
                    interval_started = time.monotonic()

                evaluate = (
                    step % self.config.eval_every == 0 or step == self.config.steps
                )
                if evaluate:
                    self._evaluate_and_log(step)
                    self.model.train()
                    interval_started = time.monotonic()
                    if self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()

            rollout_metrics = self._rollout_metrics()
            self.history[-1].update(rollout_metrics)
            self._log(rollout_metrics, self.config.steps)
            self.config.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "ema_model": self.ema.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config": asdict(self.config),
                    "history": self.history,
                    "train_samples": self.train_names,
                    "eval_samples": self.eval_names,
                },
                self.config.output,
            )
            if self.wandb_run:
                table = self.wandb.Table(
                    columns=["step", "intervention", "loss", "delta_from_correct"],
                    data=self.eval_table_rows,
                )
                self.wandb_run.log({"evaluation/action_interventions": table})
                best = min(row["eval/loss"] for row in self.history)
                self.wandb_run.summary["best_eval_loss"] = best
                self.wandb_run.summary["final_eval_loss"] = self.history[-1][
                    "eval/loss"
                ]
                self.wandb_run.summary["checkpoint"] = str(self.config.output)
        finally:
            if self.wandb_run:
                self.wandb_run.finish()


if __name__ == "__main__":
    Trainer(TrainConfig.from_cli()).run()
