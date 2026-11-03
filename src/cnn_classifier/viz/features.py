"""Filters, feature maps, and Grad-CAM."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from cnn_classifier.viz.style import save_figure


def plot_filters(filters: torch.Tensor, path: Path, max_filters: int = 32) -> Path:
    array = filters.detach().cpu()
    if array.ndim != 4:
        raise ValueError("filters must be [N,C,H,W]")
    count = min(len(array), max_filters)
    columns = min(8, count)
    rows = max(1, (count + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(1.6 * columns, 1.6 * rows))
    flat = np.asarray(axes).reshape(-1)
    for index in range(count):
        kernel = array[index].numpy()
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
        if kernel.shape[0] == 3:
            flat[index].imshow(np.transpose(kernel, (1, 2, 0)))
        else:
            flat[index].imshow(kernel[0], cmap="gray")
        flat[index].axis("off")
        flat[index].set_title(f"F{index}", fontsize=7)
    for axis in flat[count:]:
        axis.axis("off")
    fig.suptitle("First-layer filters")
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result


def plot_feature_maps(feature_maps: torch.Tensor, path: Path, max_maps: int = 16) -> Path:
    array = feature_maps.detach().cpu()
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("feature maps must be [C,H,W] or [1,C,H,W]")
    count = min(len(array), max_maps)
    columns = min(8, count)
    rows = max(1, (count + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(1.6 * columns, 1.6 * rows))
    flat = np.asarray(axes).reshape(-1)
    for index in range(count):
        flat[index].imshow(array[index].numpy(), cmap="viridis")
        flat[index].axis("off")
        flat[index].set_title(f"Ch {index}", fontsize=7)
    for axis in flat[count:]:
        axis.axis("off")
    fig.suptitle("Feature maps")
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result


def compute_gradcam(model: nn.Module, image: torch.Tensor, target_class: int) -> torch.Tensor:
    """Return a normalized Grad-CAM heatmap for one image."""
    if image.ndim != 3:
        raise ValueError("image must be [C,H,W]")
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    model.eval()
    target_layer = next(
        (module for module in reversed(list(model.modules())) if isinstance(module, nn.Conv2d)),
        None,
    )
    if target_layer is None:
        raise ValueError("Grad-CAM requires at least one Conv2d layer")
    activations: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output.retain_grad()
        activations.append(output)

    handle = target_layer.register_forward_hook(hook)
    try:
        output = model(image.unsqueeze(0).to(device))
        output[0, target_class].backward()
        activation = activations[0]
        gradients = activation.grad
        if gradients is None:
            raise RuntimeError("Grad-CAM target activation has no gradient")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu()
    finally:
        handle.remove()


def plot_gradcam(image: torch.Tensor, cam: torch.Tensor, path: Path) -> Path:
    array = image.detach().cpu().numpy()
    if array.shape[0] == 1:
        display = array[0]
        cmap = "gray"
    else:
        display = np.transpose(array, (1, 2, 0))
        cmap = None
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(display, cmap=cmap)
    axis.imshow(cam.detach().cpu().numpy(), cmap="viridis", alpha=0.5)
    axis.axis("off")
    axis.set_title("Grad-CAM")
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result
