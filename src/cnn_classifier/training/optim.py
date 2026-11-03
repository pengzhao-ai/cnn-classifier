"""Optimizer construction."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from cnn_classifier.config.schema import TrainingConfig


def build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    param_groups: list[dict[str, Any]] | None = None,
) -> torch.optim.Optimizer:
    """Build Adam, AdamW, or SGD with the configured hyperparameters."""
    parameters: Any = model.parameters() if param_groups is None else param_groups
    optimizer = config.optimizer
    if optimizer.name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=optimizer.lr,
            momentum=optimizer.momentum,
            weight_decay=optimizer.weight_decay,
        )
    if optimizer.name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=optimizer.lr,
            weight_decay=optimizer.weight_decay,
        )
    if optimizer.name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=optimizer.lr,
            weight_decay=optimizer.weight_decay,
        )
    raise ValueError(f"unknown optimizer {optimizer.name!r}")
