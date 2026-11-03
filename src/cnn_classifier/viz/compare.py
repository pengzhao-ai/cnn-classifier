"""Multi-run comparison figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from cnn_classifier.viz.style import PALETTE, save_figure


def plot_model_comparison(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    valid = [row for row in rows if row.get("accuracy") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, row in enumerate(valid):
        color = PALETTE[index % len(PALETTE)]
        params = row.get("params")
        latency = row.get("latency_ms")
        if params is not None:
            axes[0].scatter(params, row["accuracy"], color=color)
            axes[0].annotate(str(row.get("run", "")), (params, row["accuracy"]), fontsize=8)
        if latency is not None:
            axes[1].scatter(latency, row["accuracy"], color=color)
            axes[1].annotate(str(row.get("run", "")), (latency, row["accuracy"]), fontsize=8)
    axes[0].set_xlabel("Parameters")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs parameters")
    axes[1].set_xlabel("Latency (ms)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs latency")
    fig.tight_layout()
    result = save_figure(fig, path)
    plt.close(fig)
    return result
