"""Checkpoint serialization with config identity and RNG state."""

from __future__ import annotations

import random
from collections.abc import Sequence
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


def _normalization_dict(value: Any) -> dict[str, Any] | None:
    data = _jsonable(value)
    if data is None:
        return None
    if not isinstance(data, dict):
        return {"value": data}
    return {
        "mean": tuple(round(float(item), 8) for item in data.get("mean", ())),
        "std": tuple(round(float(item), 8) for item in data.get("std", ())),
        "source": data.get("source"),
    }


def _contract_dict(value: Any) -> dict[str, Any] | None:
    data = _jsonable(value)
    if data is None or not isinstance(data, dict):
        return None
    return {
        "input_shape": list(data.get("input_shape", ())),
        "dtype": data.get("dtype"),
        "value_range": list(data.get("value_range", ())),
        "class_names": list(data.get("class_names", ())),
        "label_map": dict(data.get("label_map", {})),
        "channel_policy": data.get("channel_policy"),
        "stem_adaptation": data.get("stem_adaptation"),
    }


def validate_checkpoint_identity(
    payload: CheckpointPayload,
    *,
    class_names: Sequence[str] | None = None,
    normalization: Any = None,
    model_contract: Any = None,
) -> None:
    """Reject a checkpoint whose class map, normalization, or contract changed."""
    stored_names = payload.get("class_names")
    if (
        class_names is not None
        and stored_names is not None
        and tuple(stored_names) != tuple(class_names)
    ):
        raise RunError(
            f"checkpoint class names {tuple(stored_names)!r} do not match "
            f"current class names {tuple(class_names)!r}"
        )
    if (
        payload.get("normalization") is not None or normalization is not None
    ) and _normalization_dict(payload.get("normalization")) != _normalization_dict(normalization):
        raise RunError("checkpoint normalization does not match the current run")
    stored_contract = payload.get("model_contract")
    if (
        stored_contract is not None
        and model_contract is not None
        and _contract_dict(stored_contract) != _contract_dict(model_contract)
    ):
        raise RunError("checkpoint model contract does not match the current run")


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
