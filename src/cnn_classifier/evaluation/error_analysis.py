"""Error analysis helpers for evaluation reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorExample:
    index: int
    true: str
    predicted: str
    confidence: float


def most_confident_errors(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    class_names: Sequence[str],
    max_examples: int = 32,
) -> list[ErrorExample]:
    """Return wrong predictions ranked by model confidence."""
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.int64)
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    wrong = np.flatnonzero(predictions != truth)
    order = wrong[np.argsort(confidence[wrong])[::-1]][:max_examples]
    return [
        ErrorExample(
            index=int(index),
            true=class_names[int(truth[index])],
            predicted=class_names[int(predictions[index])],
            confidence=float(confidence[index]),
        )
        for index in order
    ]


def confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> list[tuple[str, str, int]]:
    """Return off-diagonal confusion pairs sorted by count."""
    truth = np.asarray(y_true, dtype=np.int64)
    predictions = np.asarray(y_pred, dtype=np.int64)
    counts: dict[tuple[int, int], int] = {}
    for actual, predicted in zip(truth, predictions, strict=True):
        if actual == predicted:
            continue
        key = (int(actual), int(predicted))
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        (class_names[actual], class_names[predicted], count)
        for (actual, predicted), count in ordered
    ]
