"""Single-image and folder prediction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from cnn_classifier.core.errors import DataError


@dataclass(frozen=True)
class Predictions:
    paths: tuple[Path, ...]
    probabilities: np.ndarray
    predicted_labels: np.ndarray
    class_names: tuple[str, ...]

    def top_k(self, k: int = 3) -> list[list[tuple[str, float]]]:
        order = np.argsort(self.probabilities, axis=1)[:, -k:][:, ::-1]
        return [
            [(self.class_names[int(index)], float(row[index])) for index in row] for row in order
        ]


@torch.no_grad()
def predict_paths(
    model: nn.Module,
    paths: Sequence[Path],
    transform: Any,
    device: torch.device | str,
    class_names: Sequence[str],
) -> Predictions:
    """Predict a sequence of images without requiring a DataLoader."""
    runtime_device = torch.device(device) if isinstance(device, str) else device
    was_training = model.training
    model.eval()
    batch = []
    for path in paths:
        try:
            with Image.open(path) as image:
                tensor = transform(image.copy())
        except (OSError, ValueError) as exc:
            raise DataError(f"cannot read prediction image {path}: {exc}") from exc
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        batch.append(tensor)
    if not batch:
        raise DataError("predict_paths received no paths")
    probabilities = torch.softmax(model(torch.cat(batch).to(runtime_device)), dim=1)
    model.train(was_training)
    labels = probabilities.argmax(dim=1).cpu().numpy()
    return Predictions(
        paths=tuple(Path(path) for path in paths),
        probabilities=probabilities.cpu().numpy(),
        predicted_labels=np.asarray(labels),
        class_names=tuple(class_names),
    )
