"""Model parameter and shape summaries."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def model_summary(model: nn.Module, input_shape: tuple[int, int, int]) -> dict[str, Any]:
    """Return parameter counts, module details, and output shape."""
    was_training = model.training
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    model.eval()
    try:
        with torch.no_grad():
            output = model(torch.zeros(1, *input_shape, device=device))
    finally:
        model.train(was_training)
    modules = [
        {
            "name": name or "model",
            "type": type(module).__name__,
            "params": sum(parameter.numel() for parameter in module.parameters()),
        }
        for name, module in model.named_modules()
    ]
    return {
        "input_shape": list(input_shape),
        "output_shape": list(output.shape),
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "modules": modules,
        "macs": None,
    }
