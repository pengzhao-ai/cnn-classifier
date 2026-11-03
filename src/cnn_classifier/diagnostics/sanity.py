"""One-batch overfit sanity check."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OverfitResult:
    initial_loss: float
    final_loss: float
    steps: int


def overfit_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int = 100,
    target_loss: float = 0.01,
) -> OverfitResult:
    """Train one batch to a small loss, restoring the model mode."""
    images = batch[0].to(device)
    labels = batch[1].to(device)
    was_training = model.training
    model.train()
    initial_loss = 0.0
    final_loss = 0.0
    steps = 0
    try:
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            if step == 1:
                initial_loss = float(loss.item())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
            steps = step
            if final_loss <= target_loss:
                break
    finally:
        model.train(was_training)
    return OverfitResult(initial_loss=initial_loss, final_loss=final_loss, steps=steps)
