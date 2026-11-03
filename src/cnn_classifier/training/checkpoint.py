"""Checkpoint serialization with config identity and RNG state."""

from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from cnn_classifier.core.errors import RunError

CheckpointPayload = dict[str, Any]


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return cast(Any, asdict)(value)
    return value


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    config_hash: str,
    class_names: tuple[str, ...],
    normalization: Any,
    model_contract: Any,
) -> None:
    """Save a complete, resumable checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: CheckpointPayload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "config_hash": config_hash,
        "class_names": tuple(class_names),
        "normalization": _jsonable(normalization),
        "model_contract": _jsonable(model_contract),
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    if torch.cuda.is_available():
        payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
    expected_config_hash: str | None = None,
    allow_config_change: bool = False,
) -> CheckpointPayload:
    """Load a trusted checkpoint and enforce its config identity."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RunError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunError(f"checkpoint {path} does not contain a mapping")
    if (
        expected_config_hash is not None
        and payload.get("config_hash") != expected_config_hash
        and not allow_config_change
    ):
        raise RunError(
            f"checkpoint config hash {payload.get('config_hash')!r} does not match "
            f"expected {expected_config_hash!r}"
        )
    return payload
