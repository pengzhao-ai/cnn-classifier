"""Model card figures without optional architecture backends."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from cnn_classifier.viz.style import save_figure


def plot_model_card(summary: Mapping[str, Any], path: Path) -> Path:
    """Render a contract-style summary table as a figure."""
    modules = list(summary.get("modules", []))[:24]
    lines = [
        f"Input: {summary.get('input_shape')}",
        f"Output: {summary.get('output_shape')}",
        f"Total params: {summary.get('total_params'):,}"
        if isinstance(summary.get("total_params"), int)
        else f"Total params: {summary.get('total_params')}",
        f"Trainable: {summary.get('trainable_params')}",
        "",
        f"{'Module':<28} {'Type':<18} {'Params':>10}",
        "-" * 60,
    ]
    for module in modules:
        lines.append(
            f"{str(module.get('name', ''))[:27]:<28} "
            f"{str(module.get('type', ''))[:17]:<18} "
            f"{int(module.get('params', 0)):>10,}"
        )
    fig, axis = plt.subplots(figsize=(10, max(3.0, 0.26 * len(lines) + 1.0)))
    axis.axis("off")
    axis.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        family="DejaVu Sans Mono",
        fontsize=7.5,
        transform=axis.transAxes,
    )
    axis.set_title("Model card")
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result
