"""Generate the deterministic 16-bit bring-your-own-data fixture.

The fixture is committed so the quickstart needs no network. It contains three
classes, twenty 16x16 16-bit grayscale images per class, and a manifest with
path, label, split, and group columns.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

LABELS = ("circle", "square", "stripe")
IMAGE_SIZE = 16
TRAIN_PER_CLASS = 15
VAL_PER_CLASS = 5
DEFAULT_ROOT = Path(__file__).resolve().parent


def _circle(index: int) -> np.ndarray:
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    radius = 3.0 + (index % 3)
    distance = np.sqrt((xx - 7.5) ** 2 + (yy - 7.5) ** 2)
    inside = distance <= radius
    background = 1500.0 + yy * 120.0
    foreground = 25000.0 + index * 1800.0
    return np.where(inside, foreground, background)


def _square(index: int) -> np.ndarray:
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    half = 2 + (index % 3)
    inside = (np.abs(xx - 7.5) <= half) & (np.abs(yy - 7.5) <= half)
    background = 1200.0 + xx * 100.0
    foreground = 27000.0 + index * 1700.0
    return np.where(inside, foreground, background)


def _stripe(index: int) -> np.ndarray:
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    period = 2.5 + 0.5 * (index % 4)
    wave = (np.sin(xx * period + yy * 0.2) + 1.0) / 2.0
    return 4000.0 + wave * (28000.0 + index * 1600.0)


GENERATORS = {"circle": _circle, "square": _square, "stripe": _stripe}


def generate_fixture(root: Path = DEFAULT_ROOT, seed: int = 7) -> Path:
    """Write the fixture images and manifest, returning the manifest path."""
    root = Path(root)
    fixture_root = root / "fixture"
    rows: list[dict[str, str]] = []
    for label in LABELS:
        generator = GENERATORS[label]
        for index in range(TRAIN_PER_CLASS + VAL_PER_CLASS):
            split = "train" if index < TRAIN_PER_CLASS else "val"
            directory = fixture_root / split / label
            directory.mkdir(parents=True, exist_ok=True)
            value_index = (index + seed) % (TRAIN_PER_CLASS + VAL_PER_CLASS)
            values = generator(value_index)
            array = np.clip(values, 0, 65535).astype(np.uint16)
            relative = Path(split) / label / f"{label}_{index:02d}.png"
            Image.fromarray(array).save(directory / f"{label}_{index:02d}.png")
            rows.append(
                {
                    "path": relative.as_posix(),
                    "label": label,
                    "split": split,
                    "group": f"{label}_{index:02d}",
                }
            )
    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split", "group"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    manifest = generate_fixture(args.root, args.seed)
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
