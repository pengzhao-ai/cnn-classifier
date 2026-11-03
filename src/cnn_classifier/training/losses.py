"""Loss functions and batch-level MixUp/CutMix."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from cnn_classifier.config.schema import LossConfig


@dataclass(frozen=True)
class MixedBatch:
    images: torch.Tensor
    labels_a: torch.Tensor
    labels_b: torch.Tensor
    lam: float


def build_loss(config: LossConfig) -> nn.Module:
    """Build the configured classification loss."""
    if config.name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    raise ValueError(f"unknown loss {config.name!r}")


def mixup_cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
    generator: torch.Generator,
) -> MixedBatch:
    """Apply MixUp, CutMix, or neither, returning labels for both mixtures."""
    if images.ndim != 4:
        raise ValueError("MixUp/CutMix expects [B,C,H,W] images")
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return MixedBatch(images, labels, labels, 1.0)
    batch_size = images.shape[0]
    permutation = torch.randperm(batch_size, generator=generator, device=images.device)
    use_cutmix = cutmix_alpha > 0 and (
        mixup_alpha <= 0 or float(torch.rand(1, generator=generator).item()) < 0.5
    )
    if use_cutmix:
        lam = float(torch.rand(1, generator=generator).item())
        cut_ratio = math.sqrt(1.0 - lam)
        cut_h = int(images.shape[2] * cut_ratio)
        cut_w = int(images.shape[3] * cut_ratio)
        center_y = int(torch.randint(images.shape[2], (1,), generator=generator).item())
        center_x = int(torch.randint(images.shape[3], (1,), generator=generator).item())
        y1 = max(center_y - cut_h // 2, 0)
        y2 = min(center_y + cut_h // 2, images.shape[2])
        x1 = max(center_x - cut_w // 2, 0)
        x2 = min(center_x + cut_w // 2, images.shape[3])
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (images.shape[2] * images.shape[3]))
        return MixedBatch(mixed, labels, labels[permutation], lam)
    lam = float(
        torch.distributions.Beta(torch.tensor(mixup_alpha), torch.tensor(mixup_alpha))
        .sample()
        .item()
    )
    mixed = lam * images + (1.0 - lam) * images[permutation]
    return MixedBatch(mixed, labels, labels[permutation], lam)
