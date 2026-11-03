"""Activation statistics for ReLU layers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ActivationStat:
    name: str
    type: str
    mean: float
    std: float
    zero_fraction: float


def activation_report(model: nn.Module, batch: torch.Tensor) -> list[ActivationStat]:
    """Record mean, std, and zero fraction for each ReLU activation."""
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    was_training = model.training
    records: list[ActivationStat] = []
    handles = []

    def hook(name: str, module: nn.Module) -> Callable[[nn.Module, Any, Any], None]:
        def _record(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output.detach().float()
            records.append(
                ActivationStat(
                    name=name,
                    type=type(module).__name__,
                    mean=float(tensor.mean().item()),
                    std=float(tensor.std().item()) if tensor.numel() > 1 else 0.0,
                    zero_fraction=float((tensor == 0).float().mean().item()),
                )
            )

        return _record

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(hook(name, module)))
    try:
        model.eval()
        with torch.no_grad():
            model(batch.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return records
