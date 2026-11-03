"""Shared publication-style Matplotlib defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, cast

import matplotlib as mpl
from matplotlib.figure import Figure

PALETTE: Final[tuple[str, ...]] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

_STYLE: Final[dict[str, Any]] = {
    "figure.figsize": (8.0, 4.5),
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.titlesize": 11.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.0,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": "cycler('color', [" + ", ".join(repr(color) for color in PALETTE) + "])",
    "grid.color": "#b0b0b0",
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.7,
    "legend.frameon": False,
    "legend.fontsize": 9.0,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "image.cmap": "viridis",
}


def set_style() -> None:
    """Apply the package's colorblind-safe plotting defaults."""
    mpl.rcParams.update(cast(Any, _STYLE))


def save_figure(fig: Figure, path: Path) -> Path:
    """Save a figure as PNG and its SVG sibling; return the requested path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = path if path.suffix.lower() == ".png" else path.with_suffix(".png")
    svg_path = path if path.suffix.lower() == ".svg" else path.with_suffix(".svg")
    fig.savefig(png_path)
    fig.savefig(svg_path)
    return path
