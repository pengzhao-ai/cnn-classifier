"""Generate a deterministic optical-degradation classification dataset.

The generator treats each base image as a group. Groups are assigned to
train/validation/test before any degraded image is produced, so a base image can
never appear in more than one split.

With no ``base_images`` argument the module creates a small synthetic base
fixture for tests. The command-line path uses selected CIFAR-10 training images
as the base source.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from cnn_classifier.core.hashing import stable_hash

LABELS = (
    "clean",
    "defocus",
    "motion_blur",
    "chromatic_aberration",
    "vignetting",
    "sensor_noise",
    "jpeg",
)
IMAGE_SIZE = (32, 32)
DEFAULT_BASE_COUNT = 300
QUICK_BASE_COUNT = 60
SYNTHETIC_BASE_COUNT = 16
REPO_ROOT = Path(__file__).resolve().parents[2]

# Severity is sampled uniformly inside each degradation's range. Clean samples
# are the zero-severity control.
SEVERITY_RANGES = {
    "clean": (0.0, 0.0),
    "defocus": (0.2, 1.0),
    "motion_blur": (0.2, 1.0),
    "chromatic_aberration": (0.2, 1.0),
    "vignetting": (0.2, 1.0),
    "sensor_noise": (0.2, 1.0),
    "jpeg": (0.2, 1.0),
}


@dataclass(frozen=True)
class GeneratedDataset:
    manifest_digest: str
    labels: tuple[str, ...]
    group_leakage: list[str]
    manifest_path: Path
    image_root: Path
    record_count: int


def _to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _from_pil(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _synthetic_base_images(seed: int, count: int = SYNTHETIC_BASE_COUNT) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    height, width = IMAGE_SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    images: list[np.ndarray] = []
    for index in range(count):
        kind = index % 5
        if kind == 0:
            base = np.stack([xx / width, yy / height, np.full_like(xx, 0.5)], axis=-1)
        elif kind == 1:
            stripes = (np.sin((xx + yy) * (0.4 + 0.1 * (index % 4))) + 1.0) / 2.0
            base = np.stack([stripes, stripes**2, 1.0 - stripes], axis=-1)
        elif kind == 2:
            checker = ((xx // 4 + yy // 4) % 2).astype(np.float32)
            base = np.stack([checker, 1.0 - checker, np.full_like(xx, 0.3)], axis=-1)
        elif kind == 3:
            distance = np.sqrt((xx - width / 2) ** 2 + (yy - height / 2) ** 2)
            rings = (np.sin(distance * 0.8) + 1.0) / 2.0
            base = np.stack([rings, np.full_like(xx, 0.4), 1.0 - rings], axis=-1)
        else:
            base = rng.random((height, width, 3)).astype(np.float32)
        tint = rng.uniform(0.7, 1.0, size=3).astype(np.float32)
        images.append(np.clip(base * tint, 0.0, 1.0))
    return images


def _load_base_images(paths: Sequence[Path]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            images.append(np.asarray(resized, dtype=np.float32) / 255.0)
    return images


def _split_groups(groups: Sequence[str], seed: int) -> dict[str, str]:
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    count = len(ordered)
    if count >= 3:
        test_count = max(1, round(count * 0.2))
        val_count = max(1, round(count * 0.2))
    else:
        test_count = 1 if count > 1 else 0
        val_count = 1 if count > 2 else 0
    train_count = count - val_count - test_count
    assignment: dict[str, str] = {}
    for position, group in enumerate(ordered):
        if position < test_count:
            assignment[group] = "test"
        elif position < test_count + val_count:
            assignment[group] = "val"
        else:
            assignment[group] = "train"
    if train_count < 1:
        raise ValueError("the generator needs at least one base image for training")
    return assignment


def _motion_blur(array: np.ndarray, severity: float) -> np.ndarray:
    length = 3 + round(severity * 4.0)
    if length % 2 == 0:
        length += 1
    angle = math.radians(severity * 180.0)
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    for offset in range(-center, center + 1):
        x = round(center + offset * math.cos(angle))
        y = round(center + offset * math.sin(angle))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    kernel /= kernel.sum()
    height, width = array.shape[:2]
    padded = np.pad(array, ((center, center), (center, center), (0, 0)), mode="edge")
    out = np.zeros_like(array)
    for dy in range(length):
        for dx in range(length):
            if kernel[dy, dx]:
                out += kernel[dy, dx] * padded[dy : dy + height, dx : dx + width]
    return np.clip(out, 0.0, 1.0)


def _chromatic_aberration(array: np.ndarray, severity: float) -> np.ndarray:
    height, width = array.shape[:2]
    shift = 0.02 + 0.06 * severity
    scales = (1.0 + shift, 1.0, 1.0 - shift)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    channels = []
    for channel_index, scale in enumerate(scales):
        channel = array[..., channel_index]
        image = Image.fromarray(np.clip(channel * 255.0, 0, 255).astype(np.uint8), mode="L")
        transformed = image.transform(
            (width, height),
            Image.Transform.AFFINE,
            (
                scale,
                0.0,
                center_x * (1.0 - scale),
                0.0,
                scale,
                center_y * (1.0 - scale),
            ),
            resample=Image.Resampling.BILINEAR,
        )
        channels.append(np.asarray(transformed, dtype=np.float32) / 255.0)
    return np.stack(channels, axis=-1)


def _vignetting(array: np.ndarray, severity: float) -> np.ndarray:
    height, width = array.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    radius = ((xx - (width - 1) / 2.0) / (width / 2.0)) ** 2 + (
        (yy - (height - 1) / 2.0) / (height / 2.0)
    ) ** 2
    mask = 1.0 - severity * 0.8 * np.clip(radius, 0.0, 1.0)
    return np.clip(array * mask[..., None], 0.0, 1.0)


def _sensor_noise(array: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    photons = np.clip(array, 0.0, 1.0) * 255.0
    shot = rng.poisson(photons) / 255.0 - array
    gaussian = rng.normal(0.0, 0.05, size=array.shape)
    return np.clip(array + severity * (shot + gaussian), 0.0, 1.0)


def _jpeg(array: np.ndarray, severity: float) -> np.ndarray:
    quality = round(95 - 70 * severity)
    buffer = io.BytesIO()
    _to_pil(array).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return _from_pil(image)


def _degrade(
    array: np.ndarray,
    label: str,
    severity: float,
    noise_rng: np.random.Generator,
) -> np.ndarray:
    if label == "clean":
        return array
    if label == "defocus":
        return _from_pil(_to_pil(array).filter(ImageFilter.GaussianBlur(0.5 + severity * 2.5)))
    if label == "motion_blur":
        return _motion_blur(array, severity)
    if label == "chromatic_aberration":
        return _chromatic_aberration(array, severity)
    if label == "vignetting":
        return _vignetting(array, severity)
    if label == "sensor_noise":
        return _sensor_noise(array, severity, noise_rng)
    if label == "jpeg":
        return _jpeg(array, severity)
    raise ValueError(f"unknown degradation label {label!r}")


def generate_fixture(
    root: Path,
    seed: int,
    base_images: Sequence[Path] | None = None,
) -> GeneratedDataset:
    """Generate degraded images and a manifest under ``root``."""
    root = Path(root)
    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    arrays = _synthetic_base_images(seed) if base_images is None else _load_base_images(base_images)
    groups = [f"base_{index:04d}" for index in range(len(arrays))]
    assignment = _split_groups(groups, seed)

    severity_rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for group, array in zip(groups, arrays, strict=True):
        split = assignment[group]
        for label in LABELS:
            low, high = SEVERITY_RANGES[label]
            severity = 0.0 if high == 0.0 else severity_rng.uniform(low, high)
            degraded = _degrade(array, label, severity, noise_rng)
            relative_path = Path("images") / f"{group}_{label}.png"
            _to_pil(degraded).save(root / relative_path)
            records.append(
                {
                    "path": relative_path.as_posix(),
                    "label": label,
                    "split": split,
                    "group": group,
                    "severity": round(float(severity), 6),
                }
            )

    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split", "group", "severity"])
        writer.writeheader()
        writer.writerows(records)

    group_splits: dict[str, set[str]] = {}
    for record in records:
        group_splits.setdefault(str(record["group"]), set()).add(str(record["split"]))
    group_leakage = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    manifest_digest = stable_hash(records)
    return GeneratedDataset(
        manifest_digest=manifest_digest,
        labels=LABELS,
        group_leakage=group_leakage,
        manifest_path=manifest_path,
        image_root=image_root,
        record_count=len(records),
    )


def _cifar_base_images(root: Path, count: int, seed: int) -> list[Path]:
    import torchvision

    dataset = torchvision.datasets.CIFAR10(root=str(REPO_ROOT / "data"), train=True, download=True)
    base_dir = root / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    indices = random.Random(seed).sample(range(len(dataset)), min(count, len(dataset)))
    paths: list[Path] = []
    for index in indices:
        image, _ = dataset[index]
        path = base_dir / f"cifar_{index:05d}.png"
        image.save(path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "data" / "optics_degradation",
        help="Output dataset directory (default: <repo>/data/optics_degradation).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Deterministic seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Use {QUICK_BASE_COUNT} base images instead of {DEFAULT_BASE_COUNT}.",
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    base_count = QUICK_BASE_COUNT if args.quick else DEFAULT_BASE_COUNT
    base_images = _cifar_base_images(root, base_count, args.seed)
    dataset = generate_fixture(root, seed=args.seed, base_images=base_images)
    print(f"Manifest: {dataset.manifest_path}")
    print(f"Images: {dataset.image_root}")
    print(f"Records: {dataset.record_count}")
    print(f"Manifest digest: {dataset.manifest_digest}")
    print(f"Group leakage: {dataset.group_leakage}")


if __name__ == "__main__":
    main()
