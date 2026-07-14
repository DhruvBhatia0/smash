"""MIRA's codec reconstruction objective."""

from __future__ import annotations

import weakref

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ReconstructionLoss(nn.Module):
    """L1 + VGG-LPIPS + normalized DINO consistency with adaptive weights."""

    def __init__(
        self,
        *,
        dino: nn.Module,
        last_layer: Tensor,
        frame_fraction: float = 0.25,
        max_adaptive_weight: float = 1e4,
    ) -> None:
        super().__init__()
        if not 0 < frame_fraction <= 1:
            raise ValueError("frame_fraction must be in (0, 1]")

        import lpips

        self.lpips = lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False)
        self.frame_fraction = frame_fraction
        self.max_adaptive_weight = max_adaptive_weight

        # The model already owns these objects. Weak/non-registered references avoid duplicating
        # the frozen DINO backbone and decoder parameter in this module's state dict.
        self._dino = weakref.ref(dino)
        object.__setattr__(self, "_last_layer", last_layer)

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        dino_prediction: Tensor | None = None,
        target_dino_features: tuple[Tensor, ...] | None = None,
    ) -> dict[str, Tensor]:
        """Compute the released MIRA objective.

        ``prediction`` and ``target`` are the native-resolution videos in ``[-1, 1]``. A padded
        ``dino_prediction`` and its encoder-produced target features may be supplied when the native
        resolution is not divisible by the codec's spatial stride.
        """
        prediction = prediction.float()
        target = target.float()
        loss_mae = F.l1_loss(prediction, target)

        frame_count = max(1, round(prediction.shape[1] * self.frame_fraction))
        lpips_frames = self._sample_frames(
            prediction.shape[1], frame_count, prediction.device
        )
        loss_lpips = self.lpips(
            prediction[:, lpips_frames].flatten(0, 1),
            target[:, lpips_frames].flatten(0, 1),
        ).mean()

        dino_video = prediction if dino_prediction is None else dino_prediction.float()
        dino_frames = self._sample_frames(
            dino_video.shape[1], frame_count, dino_video.device
        )
        predicted_features = self._dino_features(dino_video[:, dino_frames])
        if target_dino_features is None:
            with torch.no_grad():
                target_features = self._dino_features(target[:, dino_frames])
        else:
            target_features = tuple(
                features[:, dino_frames].detach() for features in target_dino_features
            )

        loss_dino = torch.stack(
            [
                F.mse_loss(
                    F.normalize(actual, dim=2, eps=1e-6),
                    F.normalize(expected, dim=2, eps=1e-6),
                )
                for actual, expected in zip(
                    predicted_features, target_features, strict=True
                )
            ]
        ).mean()

        lpips_weight = self._adaptive_weight(loss_mae, loss_lpips)
        dino_weight = self._adaptive_weight(loss_mae, loss_dino)
        loss_total = loss_mae + lpips_weight * loss_lpips + dino_weight * loss_dino
        return {
            "loss_total": loss_total,
            "loss_mae": loss_mae,
            "loss_lpips_perceptual": loss_lpips,
            "loss_lpips_perceptual_auto_w": lpips_weight,
            "loss_dino_latent_consistency": loss_dino,
            "loss_dino_latent_consistency_auto_w": dino_weight,
        }

    @staticmethod
    def _sample_frames(time: int, count: int, device: torch.device) -> Tensor:
        return torch.randperm(time, device=device)[:count].sort().values

    def _dino_features(self, video: Tensor) -> tuple[Tensor, ...]:
        # DINO receives ImageNet-normalized [0, 1] frames. Gradients must flow through this frozen
        # network to reconstructed pixels, so this path deliberately does not use no_grad().
        mean = video.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
        std = video.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
        normalized = ((video + 1) / 2 - mean) / std
        dino = self._dino()
        if dino is None:
            raise RuntimeError("The codec's DINO encoder no longer exists")
        return dino.intermediate_layers(normalized)

    def _adaptive_weight(self, anchor: Tensor, other: Tensor) -> Tensor:
        if not torch.is_grad_enabled() or not anchor.requires_grad:
            return anchor.new_ones(())
        last_layer: Tensor = self._last_layer
        anchor_gradient = torch.autograd.grad(anchor, last_layer, retain_graph=True)[0]
        other_gradient = torch.autograd.grad(other, last_layer, retain_graph=True)[0]
        return (
            anchor_gradient.norm()
            .div(other_gradient.norm() + 1e-6)
            .clamp(0, self.max_adaptive_weight)
            .detach()
        )
