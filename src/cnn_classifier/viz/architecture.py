"""Model card and stage-pipeline figures without optional backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cnn_classifier.diagnostics.architecture import ArchitectureStage
from cnn_classifier.viz.style import save_figure

_COLORS = {
    "input": "#e5e7eb",
    "stem": "#dbeafe",
    "stage": "#bfdbfe",
    "head": "#dcfce7",
    "output": "#fef3c7",
}
_FROZEN = "#f3f4f6"


@dataclass(frozen=True)
class _Card:
    name: str
    kind: str
    shape: tuple[int, ...]
    params: int
    frozen: bool
    residual: bool


def _format_shape(shape: Sequence[int]) -> str:
    if not shape:
        return "?"
    return "×".join(str(int(value)) for value in shape)  # noqa: RUF001


def _format_params(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _draw_card(
    axis: Any,
    x: float,
    name: str,
    kind: str,
    shape: Sequence[int],
    params: int,
    *,
    frozen: bool,
    residual: bool,
    offset: float = 0.0,
) -> None:
    width, height, bottom = 1.12, 0.9, 0.55
    face = _FROZEN if frozen else _COLORS.get(kind, _COLORS["stage"])
    hatch = "///" if frozen else None
    for plate in (0.12, 0.06):
        axis.add_patch(
            FancyBboxPatch(
                (x + plate, bottom + plate),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=face,
                edgecolor="#94a3b8",
                linewidth=0.6,
                alpha=0.55,
            )
        )
    axis.add_patch(
        FancyBboxPatch(
            (x + offset, bottom),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=face,
            edgecolor="#475569",
            linewidth=1.0,
            hatch=hatch,
        )
    )
    axis.text(
        x + width / 2,
        1.69,
        name,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#0f172a",
    )
    axis.text(
        x + width / 2,
        1.47,
        _format_shape(shape),
        ha="center",
        va="bottom",
        fontsize=7,
        color="#334155",
    )
    detail = f"{_format_params(params)}" if params else ""
    if frozen:
        detail = "frozen" if not detail else f"{detail} · frozen"
    if detail:
        axis.text(
            x + width / 2,
            0.34,
            detail,
            ha="center",
            va="top",
            fontsize=6.5,
            color="#64748b",
        )
    if residual:
        axis.add_patch(
            FancyArrowPatch(
                (x + 0.08, 1.78),
                (x + width - 0.08, 1.78),
                connectionstyle="arc3,rad=-0.45",
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.9,
                linestyle=(0, (2, 2)),
                color="#64748b",
            )
        )


def plot_cnn_pipeline(
    stages: Sequence[ArchitectureStage],
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    path: Path,
    title: str,
) -> Path:
    """Draw a concise, publication-style CNN stage pipeline."""
    if not stages:
        raise ValueError("at least one architecture stage is required")
    cards = [
        _Card(
            name="input",
            kind="input",
            shape=input_shape,
            params=0,
            frozen=False,
            residual=False,
        )
    ]
    cards.extend(
        _Card(
            name=stage.name,
            kind=stage.kind,
            shape=stage.output_shape,
            params=stage.params,
            frozen=stage.frozen,
            residual=stage.residual,
        )
        for stage in stages
    )
    cards.append(
        _Card(
            name="output",
            kind="output",
            shape=(num_classes,),
            params=0,
            frozen=False,
            residual=False,
        )
    )

    spacing = 1.55
    n = len(cards)
    fig, axis = plt.subplots(figsize=(max(11.0, 1.45 * n + 1.6), 3.9))
    axis.set_xlim(-0.55, (n - 1) * spacing + 1.65)
    axis.set_ylim(-0.1, 2.25)
    axis.axis("off")

    for index, card in enumerate(cards):
        x = index * spacing
        _draw_card(
            axis,
            x,
            card.name,
            card.kind,
            card.shape,
            card.params,
            frozen=card.frozen,
            residual=card.residual,
        )
        if index < n - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (x + 1.12, 1.0),
                    (x + spacing, 1.0),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.0,
                    color="#475569",
                )
            )

    total = sum(stage.params for stage in stages)
    trainable = sum(stage.trainable_params for stage in stages)
    notes = []
    if any(stage.residual for stage in stages):
        notes.append("dashed arc = residual skip")
    if any(stage.frozen for stage in stages):
        notes.append("hatched = frozen")
    if notes:
        axis.text(
            -0.35,
            -0.02,
            "   ·   ".join(notes),
            ha="left",
            va="top",
            fontsize=6.5,
            color="#64748b",
        )
    fig.suptitle(
        f"{title}\n{_format_params(total)} parameters · {_format_params(trainable)} trainable",
        fontsize=10,
        fontweight="bold",
        color="#0f172a",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    result = save_figure(fig, path)
    plt.close(fig)
    return result


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
