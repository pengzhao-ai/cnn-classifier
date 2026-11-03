"""Inference, TTA, metrics, and prediction artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn_classifier.config.schema import EvaluationConfig
from cnn_classifier.core.errors import DataError
from cnn_classifier.evaluation.bootstrap import ConfidenceInterval, bootstrap_ci
from cnn_classifier.evaluation.latency import LatencyReport, measure_latency
from cnn_classifier.evaluation.metrics import MetricsReport, accuracy, classification_metrics


@dataclass(frozen=True)
class EvaluationResult:
    metrics: MetricsReport
    probabilities: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    indices: np.ndarray
    paths: tuple[str, ...]
    latency: LatencyReport | None
    base_metrics: MetricsReport | None = None
    tta_metrics: MetricsReport | None = None
    confidence_intervals: dict[str, ConfidenceInterval] = field(default_factory=dict)


@torch.no_grad()
def _predict(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    config: EvaluationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    base_parts: list[np.ndarray] = []
    tta_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    use_hflip = config.tta.enabled and "hflip" in config.tta.transforms
    for images, labels, indices in loader:
        images = images.to(device)
        base = torch.softmax(model(images), dim=1)
        base_parts.append(base.cpu().numpy())
        if use_hflip:
            flipped = torch.flip(images, dims=[3])
            tta = (base + torch.softmax(model(flipped), dim=1)) / 2.0
        else:
            tta = base
        tta_parts.append(tta.cpu().numpy())
        labels_parts.append(labels.numpy())
        index_parts.append(indices.numpy())
    if not base_parts:
        raise DataError("evaluation loader produced no samples")
    return (
        np.concatenate(base_parts),
        np.concatenate(tta_parts),
        np.concatenate(labels_parts),
        np.concatenate(index_parts),
        np.concatenate(index_parts),
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device | str,
    class_names: tuple[str, ...] | list[str],
    config: EvaluationConfig,
) -> EvaluationResult:
    """Evaluate on a loader without touching training or test state."""
    runtime_device = torch.device(device) if isinstance(device, str) else device
    was_training = model.training
    base_probs, tta_probs, y_true, indices, _ = _predict(model, loader, runtime_device, config)
    y_pred = base_probs.argmax(axis=1)
    names = tuple(class_names)
    base_metrics = classification_metrics(y_true, y_pred, base_probs, names, top_k=config.top_k)
    tta_metrics = None
    if config.tta.enabled:
        tta_pred = tta_probs.argmax(axis=1)
        tta_metrics = classification_metrics(y_true, tta_pred, tta_probs, names, top_k=config.top_k)
    metrics = tta_metrics or base_metrics
    intervals: dict[str, ConfidenceInterval] = {}
    if config.bootstrap.enabled:
        used_probs = tta_probs if config.tta.enabled else base_probs
        used_pred = used_probs.argmax(axis=1)
        intervals["accuracy"] = bootstrap_ci(
            y_true,
            used_pred,
            accuracy,
            n_resamples=config.bootstrap.n_resamples,
            confidence=config.bootstrap.confidence,
            seed=0,
            stratify=config.bootstrap.stratify,
        )
    latency = None
    if config.latency.enabled:
        sample_shape = loader.dataset[0][0].shape
        input_shape = (
            int(sample_shape[0]),
            int(sample_shape[1]),
            int(sample_shape[2]),
        )
        latency = measure_latency(
            model,
            input_shape,
            runtime_device,
            config.latency.warmup,
            config.latency.iterations,
        )
    model.train(was_training)
    return EvaluationResult(
        metrics=metrics,
        probabilities=tta_probs if config.tta.enabled else base_probs,
        y_true=y_true,
        y_pred=tta_probs.argmax(axis=1) if config.tta.enabled else y_pred,
        indices=indices,
        paths=tuple(str(int(index)) for index in indices),
        latency=latency,
        base_metrics=base_metrics,
        tta_metrics=tta_metrics,
        confidence_intervals=intervals,
    )


def save_predictions(result: EvaluationResult, path: Path) -> Path:
    """Save prediction arrays in a portable ``.npz`` archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probabilities=result.probabilities,
        y_true=result.y_true,
        y_pred=result.y_pred,
        indices=result.indices,
    )
    return path
