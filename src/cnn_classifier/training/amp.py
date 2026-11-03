"""Device-aware mixed-precision policy."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch

from cnn_classifier.core.errors import ConfigError

AmpRequest = Literal["auto", "off", "fp16", "bf16"]


@dataclass(frozen=True)
class AmpPolicy:
    enabled: bool
    dtype: torch.dtype | None
    scaler: torch.amp.GradScaler | None
    device_type: str


def resolve_amp(device: torch.device, requested: AmpRequest) -> AmpPolicy:
    """Resolve the requested AMP mode against the actual device."""
    if requested == "off":
        return AmpPolicy(False, None, None, device.type)
    if requested == "auto":
        if device.type == "cuda":
            return AmpPolicy(True, torch.float16, torch.amp.GradScaler("cuda"), "cuda")
        return AmpPolicy(False, None, None, device.type)
    if requested == "fp16":
        if device.type != "cuda":
            raise ConfigError("fp16 AMP is only supported on CUDA devices")
        return AmpPolicy(True, torch.float16, torch.amp.GradScaler("cuda"), "cuda")
    if requested == "bf16":
        if device.type == "mps":
            raise ConfigError("bf16 AMP is not enabled for MPS in this release")
        return AmpPolicy(True, torch.bfloat16, None, device.type)
    raise ConfigError(f"unknown AMP request {requested!r}")


@contextlib.contextmanager
def autocast_context(policy: AmpPolicy) -> Iterator[None]:
    """Enter autocast when the policy is enabled; otherwise do nothing."""
    if not policy.enabled or policy.dtype is None:
        yield
        return
    with torch.autocast(device_type=policy.device_type, dtype=policy.dtype):
        yield
