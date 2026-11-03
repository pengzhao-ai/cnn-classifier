"""Inference latency and throughput measurement."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LatencyReport:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    throughput: float


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    device: torch.device | str,
    warmup: int = 20,
    iterations: int = 100,
) -> LatencyReport:
    """Measure single-image forward-pass latency with warmup and sync."""
    runtime_device = torch.device(device) if isinstance(device, str) else device
    was_training = model.training
    model.eval()
    sample = torch.zeros(1, *input_shape, device=runtime_device)
    for _ in range(max(0, warmup)):
        model(sample)
    _synchronize(runtime_device)
    timings = []
    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        model(sample)
        _synchronize(runtime_device)
        timings.append((time.perf_counter() - start) * 1000.0)
    model.train(was_training)
    values = np.array(timings, dtype=np.float64)
    mean_ms = float(values.mean())
    return LatencyReport(
        mean_ms=mean_ms,
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        throughput=float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    )
