"""Learning-rate scheduler construction."""

from __future__ import annotations

import math

import torch

from cnn_classifier.config.schema import SchedulerConfig


def _cosine(epoch: int, epochs: int, min_ratio: float) -> float:
    progress = min(epoch / max(1, epochs - 1), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the configured scheduler; warmup is explicit."""
    if config.name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
    if config.name == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: _cosine(epoch, epochs, config.min_lr_ratio),
        )
    if config.name == "cosine_warmup":
        if config.warmup_epochs <= 0:
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda epoch: _cosine(epoch, epochs, config.min_lr_ratio),
            )

        def warmup_lambda(epoch: int) -> float:
            if epoch < config.warmup_epochs:
                return (epoch + 1) / config.warmup_epochs
            progress = (epoch - config.warmup_epochs) / max(1, epochs - config.warmup_epochs - 1)
            progress = min(progress, 1.0)
            return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    if config.name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    if config.name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    raise ValueError(f"unknown scheduler {config.name!r}")
