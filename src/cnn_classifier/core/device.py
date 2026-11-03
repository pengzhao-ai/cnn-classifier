"""Device resolution for CUDA, Apple MPS, and CPU."""

from __future__ import annotations

import platform

import torch

from cnn_classifier.core.errors import ConfigError


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve ``auto`` or an explicit device string into a ``torch.device``.

    ``auto`` prefers CUDA, then MPS, then CPU. Explicit CUDA or MPS requests are
    rejected when the backend is unavailable, so a run never silently changes
    hardware.
    """
    value = spec.strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ConfigError(f"invalid device {spec!r}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError(f"CUDA device {spec!r} requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ConfigError(f"MPS device {spec!r} requested but MPS is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ConfigError(f"unsupported device type {device.type!r}")
    return device


def device_report() -> dict[str, object]:
    """Return runtime and device availability information."""
    return {
        "device": str(resolve_device("auto")),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
