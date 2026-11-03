"""NumPy array adapter for scientific image tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.core.errors import ConfigError, DataError
from cnn_classifier.core.hashing import stable_hash
from cnn_classifier.core.types import DatasetBundle
from cnn_classifier.data.adapters._common import (
    RawSample,
    assign_split,
    build_bundle,
    resolve_class_names,
)
from cnn_classifier.data.adapters.registry import register_dataset


def _load_arrays(array_path: Path, labels_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    try:
        if array_path.suffix == ".npz":
            payload = np.load(array_path)
            images = np.asarray(payload["images"])
            labels = np.asarray(payload["labels"] if labels_path is None else np.load(labels_path))
        else:
            images = np.asarray(np.load(array_path))
            if labels_path is None:
                raise ConfigError("numpy adapter requires labels_path unless the array is .npz")
            labels = np.asarray(np.load(labels_path))
    except OSError as exc:
        raise DataError(f"cannot load NumPy dataset: {exc}") from exc
    if images.ndim == 3:
        images = images[:, None, :, :]
    if images.ndim != 4:
        raise DataError(f"numpy images must be [N,C,H,W] or [N,H,W]; got {images.shape}")
    if len(images) != len(labels):
        raise DataError("numpy images and labels have different lengths")
    return images, labels


def _to_float(images: np.ndarray) -> np.ndarray:
    data = images.astype(np.float64)
    if np.issubdtype(images.dtype, np.integer):
        data = data / float(np.iinfo(images.dtype).max)
    return np.clip(data, 0.0, 1.0).astype(np.float32)


@register_dataset("numpy")
class NumpyAdapter:
    """Read images and labels from .npy or .npz arrays."""

    @staticmethod
    def build(config: DataConfig) -> DatasetBundle:
        raw_path = config.adapter_params.get("array_path")
        if raw_path is None:
            raise ConfigError("numpy adapter requires adapter_params.array_path")
        array_path = Path(raw_path)
        if not array_path.is_absolute() and config.root is not None:
            array_path = config.root / array_path
        labels_raw = config.adapter_params.get("labels_path")
        labels_path = Path(labels_raw) if labels_raw is not None else None
        if labels_path is not None and not labels_path.is_absolute() and config.root is not None:
            labels_path = config.root / labels_path
        images, labels = _load_arrays(array_path, labels_path)
        images = _to_float(images)
        class_values = sorted({int(value) for value in labels.tolist()})
        class_names = resolve_class_names([f"class_{value}" for value in class_values], config)
        samples = [
            RawSample(
                source=torch.from_numpy(images[index]),
                label=f"class_{int(labels[index])}",
                split="train",
            )
            for index in range(len(images))
        ]
        split, mode = assign_split(samples, config)
        fingerprint = stable_hash(
            {
                "adapter": "numpy",
                "shape": list(images.shape),
                "labels": labels.tolist(),
                "seed": config.split.seed,
            }
        )
        return build_bundle(
            samples,
            config,
            class_names=class_names,
            split=split,
            mode=mode,
            fingerprint=fingerprint,
        )
