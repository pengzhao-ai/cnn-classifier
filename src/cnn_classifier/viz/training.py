"""Training-history figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from cnn_classifier.viz.style import PALETTE, save_figure


def plot_training_history(
    history: Sequence[Mapping[str, float]],
    path: Path,
    title: str = "Training history",
) -> Path:
    """Plot loss, accuracy, learning rate, and gradient norm panels."""
    epochs = [int(row.get("epoch", index + 1)) for index, row in enumerate(history)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    loss_ax, accuracy_ax, lr_ax, grad_ax = axes.flat

    loss_ax.plot(epochs, [row.get("train_loss", float("nan")) for row in history], label="Train")
    loss_ax.plot(epochs, [row.get("val_loss", float("nan")) for row in history], label="Val")
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.legend()

    accuracy_ax.plot(
        epochs, [row.get("train_accuracy", float("nan")) for row in history], label="Train"
    )
    accuracy_ax.plot(
        epochs, [row.get("val_accuracy", float("nan")) for row in history], label="Val"
    )
    accuracy_ax.set_title("Accuracy")
    accuracy_ax.set_xlabel("Epoch")
    accuracy_ax.legend()

    lr_ax.plot(epochs, [row.get("lr", float("nan")) for row in history], color=PALETTE[2])
    lr_ax.set_title("Learning rate")
    lr_ax.set_xlabel("Epoch")

    grad_values = [row.get("grad_norm") for row in history]
    if any(value is not None for value in grad_values):
        grad_ax.plot(epochs, [float(value or 0.0) for value in grad_values], color=PALETTE[3])
        grad_ax.set_title("Gradient norm")
        grad_ax.set_xlabel("Epoch")
    else:
        grad_ax.axis("off")
        grad_ax.text(0.5, 0.5, "gradient norm not logged", ha="center", va="center")
    for axis in (loss_ax, accuracy_ax, lr_ax, grad_ax):
        axis.grid(True)
    fig.suptitle(title)
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result
