"""Shape tracing with forward hooks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LayerShape:
    name: str
    type: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    params: int


def trace_shapes(model: nn.Module, input_shape: tuple[int, int, int]) -> list[LayerShape]:
    """Run one dummy input and report each module's shapes."""
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    was_training = model.training
    records: list[LayerShape] = []
    handles: list[Any] = []

    def hook(name: str, module: nn.Module) -> Callable[[nn.Module, Any, Any], None]:
        def _record(_module: nn.Module, inputs: Any, output: Any) -> None:
            input_tensor = inputs[0] if isinstance(inputs, tuple) else inputs
            output_tensor = output[0] if isinstance(output, tuple) else output
            records.append(
                LayerShape(
                    name=name or "model",
                    type=type(module).__name__,
                    input_shape=tuple(int(value) for value in input_tensor.shape),
                    output_shape=tuple(int(value) for value in output_tensor.shape),
                    params=sum(p.numel() for p in module.parameters()),
                )
            )

        return _record

    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(hook(name, module)))
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, *input_shape, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return records
