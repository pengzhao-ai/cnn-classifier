"""Dataset-oriented figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from cnn_classifier.viz.style import PALETTE, save_figure


def _as_image(tensor: torch.Tensor) -> np.ndarray:
    array: np.ndarray = np.asarray(tensor.detach().cpu().numpy())
    if array.ndim == 3 and array.shape[0] in (1, 3):
        if array.shape[0] == 1:
            return cast(np.ndarray, array[0])
        return np.transpose(array, (1, 2, 0))
    return array


def plot_dataset_overview(
    counts: Mapping[str, int], path: Path, title: str = "Class counts"
) -> Path:
    names = list(counts)
    values = [counts[name] for name in names]
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(names, values, color=PALETTE[0])
    axis.set_ylabel("Samples")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result


def plot_augmentation_gallery(images: Sequence[torch.Tensor], path: Path) -> Path:
    count = len(images)
    columns = min(4, count) or 1
    rows = max(1, (count + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    flat = np.asarray(axes).reshape(-1)
    for index, image in enumerate(images[: len(flat)]):
        flat[index].imshow(_as_image(image))
        flat[index].axis("off")
    for axis in flat[count:]:
        axis.axis("off")
    fig.suptitle("Augmentation samples")
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result
