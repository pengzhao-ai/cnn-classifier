"""Classification metrics with JSON-friendly reports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cnn_classifier.evaluation.calibration import CalibrationReport, expected_calibration_error


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class MetricsReport:
    accuracy: float
    top_k_accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]
    nll: float | None
    calibration: CalibrationReport | None
    support: int


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return top-1 accuracy for equal-length arrays."""
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must have the same length")
    if len(truth) == 0:
        return 0.0
    return float(np.mean(truth == predictions))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    top_k: int = 3,
) -> MetricsReport:
    """Compute top-k, per-class, confusion, NLL, and calibration metrics."""
    truth = np.asarray(y_true, dtype=np.int64)
    predictions = np.asarray(y_pred, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    names = tuple(class_names)
    if len(truth) != len(predictions) or len(truth) != len(probs):
        raise ValueError("y_true, y_pred, and probabilities must have equal length")
    if probs.ndim != 2 or probs.shape[1] != len(names):
        raise ValueError("probabilities must be [N, len(class_names)]")
    count = len(names)
    confusion = [[0 for _ in range(count)] for _ in range(count)]
    for true_label, predicted in zip(truth, predictions, strict=True):
        confusion[int(true_label)][int(predicted)] += 1
    per_class: dict[str, ClassMetrics] = {}
    f1_scores: list[float] = []
    for index, name in enumerate(names):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(count) if row != index)
        false_negative = sum(confusion[index][column] for column in range(count) if column != index)
        predicted_positive = true_positive + false_positive
        actual_positive = true_positive + false_negative
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion[index])
        per_class[name] = ClassMetrics(
            precision=float(precision), recall=float(recall), f1=float(f1), support=int(support)
        )
        f1_scores.append(f1)
    top_k = max(1, min(top_k, count))
    top_indices = np.argsort(probs, axis=1)[:, -top_k:]
    top_k_accuracy = float(
        np.mean([truth[index] in top_indices[index] for index in range(len(truth))])
    )
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(truth)), truth], 1e-12, 1.0))))
    return MetricsReport(
        accuracy=accuracy(truth, predictions),
        top_k_accuracy=top_k_accuracy,
        macro_f1=float(np.mean(f1_scores)),
        per_class=per_class,
        confusion_matrix=confusion,
        nll=nll,
        calibration=expected_calibration_error(probs, truth),
        support=len(truth),
    )
