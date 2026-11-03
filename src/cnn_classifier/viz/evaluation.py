"""Evaluation figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from cnn_classifier.viz.style import PALETTE, save_figure


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]] | np.ndarray,
    class_names: Sequence[str],
    path: Path,
    title: str = "Confusion matrix",
) -> Path:
    counts = np.asarray(matrix, dtype=np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, data, panel_title in (
        (axes[0], counts, "Counts"),
        (axes[1], normalized, "Row-normalized"),
    ):
        image = axis.imshow(data, cmap="Blues")
        axis.set_title(panel_title)
        axis.set_xticks(range(len(class_names)))
        axis.set_yticks(range(len(class_names)))
        axis.set_xticklabels(class_names, rotation=45, ha="right")
        axis.set_yticklabels(class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                value = (
                    f"{data[row, column]:.2f}"
                    if data is normalized
                    else f"{int(data[row, column])}"
                )
                axis.text(column, row, value, ha="center", va="center", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result


def _f1(value: Any) -> float:
    if hasattr(value, "f1"):
        return float(value.f1)
    if isinstance(value, Mapping):
        return float(value.get("f1", 0.0))
    return float(value)


def plot_per_class_metrics(
    per_class: Mapping[str, Any],
    path: Path,
    title: str = "Per-class F1",
) -> Path:
    names = list(per_class)
    values = [_f1(per_class[name]) for name in names]
    fig, axis = plt.subplots(figsize=(8, max(3, 0.35 * len(names) + 1)))
    axis.barh(names[::-1], values[::-1], color=PALETTE[0])
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("F1")
    axis.set_title(title)
    for index, value in enumerate(values[::-1]):
        axis.text(value + 0.01, index, f"{value:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result


def plot_calibration(report: Any, path: Path, title: str = "Calibration") -> Path:
    bins = [item for item in report.bins if item.count > 0]
    confidence = [item.confidence for item in bins]
    accuracy = [item.accuracy for item in bins]
    counts = [item.count for item in bins]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Perfect")
    axis.plot(confidence, accuracy, marker="o", color=PALETTE[0], label="Model")
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title(f"{title} (ECE={report.ece:.3f}, Brier={report.brier:.3f})")
    axis.legend()
    if counts:
        twin = axis.twinx()
        twin.bar(confidence, counts, width=0.05, alpha=0.15, color=PALETTE[1])
        twin.set_ylabel("Count")
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result


def _to_image(image: torch.Tensor) -> np.ndarray:
    array: np.ndarray = np.asarray(image.detach().cpu().numpy())
    if array.shape[0] == 1:
        return cast(np.ndarray, array[0])
    return np.transpose(array, (1, 2, 0))


def plot_error_gallery(
    images: torch.Tensor,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    path: Path,
    title: str = "Misclassifications",
) -> Path:
    count = min(len(images), len(y_true), len(y_pred))
    columns = min(4, count) or 1
    rows = max(1, (count + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows))
    flat = np.asarray(axes).reshape(-1)
    for index in range(count):
        axis = flat[index]
        axis.imshow(_to_image(images[index]))
        axis.set_title(f"True: {y_true[index]}\nPred: {y_pred[index]}", fontsize=8)
        axis.axis("off")
    for axis in flat[count:]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result
