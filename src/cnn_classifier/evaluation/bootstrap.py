"""Bootstrap confidence intervals and paired comparisons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

MetricFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float
    estimate: float


def _validate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must have the same length")
    return truth, predictions


def _resample_indices(
    truth: np.ndarray,
    rng: np.random.Generator,
    stratify: bool,
) -> np.ndarray:
    if not stratify:
        return rng.integers(0, len(truth), size=len(truth))
    pieces = []
    for label in np.unique(truth):
        class_indices = np.flatnonzero(truth == label)
        pieces.append(rng.choice(class_indices, size=len(class_indices), replace=True))
    return np.concatenate(pieces)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: MetricFunction,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    stratify: bool = False,
) -> ConfidenceInterval:
    """Return a percentile bootstrap interval for one prediction set."""
    truth, predictions = _validate(y_true, y_pred)
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        indices = _resample_indices(truth, rng, stratify)
        values.append(metric_fn(truth[indices], predictions[indices]))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(values, [100 * alpha, 100 * (1 - alpha)])
    return ConfidenceInterval(
        low=float(low), high=float(high), estimate=float(metric_fn(truth, predictions))
    )


def paired_bootstrap_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn: MetricFunction,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    stratify: bool = False,
) -> ConfidenceInterval:
    """Return a bootstrap interval for ``metric(a) - metric(b)``."""
    truth, predictions_a = _validate(y_true, pred_a)
    _, predictions_b = _validate(y_true, pred_b)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        indices = _resample_indices(truth, rng, stratify)
        values.append(
            metric_fn(truth[indices], predictions_a[indices])
            - metric_fn(truth[indices], predictions_b[indices])
        )
    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(values, [100 * alpha, 100 * (1 - alpha)])
    return ConfidenceInterval(
        low=float(low),
        high=float(high),
        estimate=float(metric_fn(truth, predictions_a) - metric_fn(truth, predictions_b)),
    )
