"""Calibration metrics: ECE, MCE, Brier score, and reliability bins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float


@dataclass(frozen=True)
class CalibrationReport:
    ece: float
    mce: float
    bins: list[CalibrationBin]
    brier: float


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> CalibrationReport:
    """Compute ECE, MCE, Brier score, and reliability bins."""
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.int64)
    if probs.ndim != 2 or len(probs) != len(truth):
        raise ValueError("probabilities must be [N,C] and match y_true")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == truth).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(confidence, edges, right=False) - 1, 0, n_bins - 1)
    bins: list[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    total = len(truth)
    for index in range(n_bins):
        mask = bin_indices == index
        count = int(mask.sum())
        if count:
            mean_conf = float(confidence[mask].mean())
            mean_acc = float(correct[mask].mean())
            gap = abs(mean_acc - mean_conf)
        else:
            mean_conf = 0.0
            mean_acc = 0.0
            gap = 0.0
        bins.append(
            CalibrationBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                confidence=mean_conf,
                accuracy=mean_acc,
            )
        )
        if total:
            ece += (count / total) * gap
            mce = max(mce, gap)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(truth)), truth] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return CalibrationReport(ece=float(ece), mce=float(mce), bins=bins, brier=brier)
