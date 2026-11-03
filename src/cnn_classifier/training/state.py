"""Mutable state shared by the trainer and callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from cnn_classifier.config.schema import RunConfig
from cnn_classifier.runs.context import RunContext


@dataclass
class TrainingState:
    model: nn.Module
    eval_model: nn.Module | None
    optimizer: torch.optim.Optimizer | None
    scheduler: Any
    scaler: Any
    epoch: int
    global_step: int
    history: list[dict[str, float]]
    metrics: dict[str, float]
    best_metric: float
    best_epoch: int
    stop_requested: bool
    run: RunContext
    config: RunConfig
    device: torch.device
    class_names: tuple[str, ...] = ()
    normalization: Any = None
    model_contract: Any = None
    extra: dict[str, Any] = field(default_factory=dict)
