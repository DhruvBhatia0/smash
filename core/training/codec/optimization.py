"""Optimization utilities matching MIRA's released codec training recipe."""

from __future__ import annotations

import math
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    """Linear warmup followed by cosine decay to ``min_lr``."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        decay_steps: int,
        min_lr: float,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0 or decay_steps < 0 or min_lr < 0:
            raise ValueError(
                "warmup steps, decay steps, and minimum LR cannot be negative"
            )
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        decay_step = step - self.warmup_steps
        if decay_step < self.decay_steps:
            cosine = 0.5 * (1 + math.cos(math.pi * decay_step / self.decay_steps))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]
        return [self.min_lr for _ in self.base_lrs]


class ModelEMA:
    """Bias-corrected EMA of trainable model parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.model = model
        self.decay = decay
        self.count = 0.0
        self.names: list[str] = []
        self.parameters: list[nn.Parameter] = []
        self.averages: list[Tensor] = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                self.names.append(name)
                self.parameters.append(parameter)
                self.averages.append(parameter.detach().clone())

    @torch.no_grad()
    def step(self) -> None:
        self.count = self.count * self.decay + 1
        weight = 1 / self.count
        torch._foreach_lerp_(
            self.averages,
            [parameter.detach() for parameter in self.parameters],
            weight,
        )

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        # Swap storages rather than copying the complete model twice for every evaluation.
        for index, parameter in enumerate(self.parameters):
            parameter.data, self.averages[index] = self.averages[index], parameter.data
        try:
            yield
        finally:
            for index, parameter in enumerate(self.parameters):
                parameter.data, self.averages[index] = (
                    self.averages[index],
                    parameter.data,
                )

    def state_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "averages": dict(zip(self.names, self.averages, strict=True)),
        }


class ScalarEMA:
    """Small EMA used for codec latent normalization statistics."""

    def __init__(self, decay: float, initial_value: float = 0.0) -> None:
        self.decay = decay
        self.value = initial_value

    def update(self, value: Tensor) -> None:
        self.value = (
            self.decay * self.value
            + (1 - self.decay) * value.detach().float().mean().item()
        )

    def synchronize(self, device: torch.device) -> float:
        if dist.is_available() and dist.is_initialized():
            value = torch.tensor(self.value, dtype=torch.float64, device=device)
            dist.all_reduce(value, op=dist.ReduceOp.AVG)
            self.value = value.item()
        return self.value
