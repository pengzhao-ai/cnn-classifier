"""Gradient statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GradientStat:
    name: str
    norm: float
    zero_fraction: float


def gradient_report(model: nn.Module) -> list[GradientStat]:
    """Return norm and zero fraction for every parameter with a gradient."""
    report: list[GradientStat] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        report.append(
            GradientStat(
                name=name,
                norm=float(torch.linalg.vector_norm(gradient).item()),
                zero_fraction=float((gradient == 0).float().mean().item()),
            )
        )
    return report
