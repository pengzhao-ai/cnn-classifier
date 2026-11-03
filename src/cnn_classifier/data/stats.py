"""Channel normalization statistics with caching helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from cnn_classifier.core.errors import DataError
from cnn_classifier.core.hashing import stable_hash
from cnn_classifier.core.types import NormalizationStats


def _as_batch(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 3:
        return images.unsqueeze(0)
    if images.ndim != 4:
        raise DataError(f"normalization expects [C,H,W] or [B,C,H,W]; got {tuple(images.shape)}")
    return images


def compute_normalization(
    dataset: Dataset[Any],
    batch_size: int = 256,
) -> NormalizationStats:
    """Compute per-channel mean and standard deviation over a dataset.

    Accumulation is in float64. A constant channel gets a standard deviation of
    1.0 so downstream normalization never divides by zero.
    """
    if len(dataset) == 0:  # type: ignore[arg-type]
        raise DataError("cannot compute normalization for an empty dataset")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sums: torch.Tensor | None = None
    squares: torch.Tensor | None = None
    count = 0
    for item in loader:
        images = item[0] if isinstance(item, (tuple, list)) else item
        batch = _as_batch(images).to(torch.float64)
        if sums is None:
            sums = torch.zeros(batch.shape[1], dtype=torch.float64)
            squares = torch.zeros(batch.shape[1], dtype=torch.float64)
        assert sums is not None and squares is not None
        sums += batch.sum(dim=(0, 2, 3))
        squares += (batch * batch).sum(dim=(0, 2, 3))
        count += batch.shape[0] * batch.shape[2] * batch.shape[3]
    if sums is None or squares is None or count == 0:
        raise DataError("dataset produced no samples for normalization")
    mean = sums / count
    variance = (squares / count) - (mean * mean)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    std = torch.where(std == 0, torch.ones_like(std), std)
    return NormalizationStats(
        mean=tuple(float(value) for value in mean.tolist()),
        std=tuple(float(value) for value in std.tolist()),
        source="computed",
    )


def save_normalization(stats: NormalizationStats, path: Path) -> None:
    """Write normalization statistics as deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mean": list(stats.mean), "std": list(stats.std), "source": stats.source}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_normalization(path: Path) -> NormalizationStats:
    """Load normalization statistics written by :func:`save_normalization`."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mean = tuple(float(value) for value in payload["mean"])
        std = tuple(float(value) for value in payload["std"])
        source = str(payload["source"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot load normalization {path}: {exc}") from exc
    if not all(math.isfinite(value) for value in (*mean, *std)):
        raise DataError(f"normalization {path} contains non-finite values")
    return NormalizationStats(mean=mean, std=std, source=source)  # type: ignore[arg-type]


def normalization_cache_key(
    dataset_fingerprint: str,
    image_size: tuple[int, int],
    channels: str,
    value_range: Mapping[str, Any] | Any,
) -> str:
    """Return a stable cache key for a normalization computation."""
    if hasattr(value_range, "model_dump"):
        policy = value_range.model_dump()
    elif isinstance(value_range, Mapping):
        policy = dict(value_range)
    else:
        policy = str(value_range)
    return stable_hash(
        {
            "fingerprint": dataset_fingerprint,
            "image_size": list(image_size),
            "channels": channels,
            "value_range": policy,
        }
    )
