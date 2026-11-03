"""Dataset inspection for class balance, duplicates, and leakage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from cnn_classifier.core.errors import DataError
from cnn_classifier.core.types import DatasetBundle


@dataclass(frozen=True)
class DataInspectionReport:
    class_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    duplicates: dict[str, int] = field(default_factory=dict)
    corrupt: list[str] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)
    split_overlap: list[str] = field(default_factory=list)
    group_leakage: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _content_hash(image: torch.Tensor) -> str:
    data = image.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def inspect_dataset(bundle: DatasetBundle, max_samples: int | None = None) -> DataInspectionReport:
    """Inspect split class balance, duplicate content, and group leakage."""
    class_counts: dict[str, dict[str, int]] = {}
    content_hashes: dict[str, int] = {}
    corrupt: list[str] = []
    sizes: dict[str, int] = {}
    path_sets: dict[str, set[str]] = {}
    group_sets: dict[str, set[str]] = {}

    splits = {"train": bundle.train, "val": bundle.val}
    if bundle.test is not None:
        splits["test"] = bundle.test
    for split_name, dataset in splits.items():
        counts: dict[str, int] = dict.fromkeys(bundle.class_names, 0)
        path_sets[split_name] = set()
        group_sets[split_name] = set()
        limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)  # type: ignore[arg-type]
        for index in range(limit):
            source_index = -1
            try:
                image, label, source_index = dataset[index]
            except (DataError, OSError, ValueError) as exc:
                path = (
                    bundle.sample_metadata.paths[source_index]
                    if 0 <= source_index < len(bundle.sample_metadata.paths)
                    else f"<sample:{index}>"
                )
                corrupt.append(f"{path}: {exc}")
                continue
            class_name = bundle.class_names[int(label)]
            counts[class_name] += 1
            digest = _content_hash(image)
            content_hashes[digest] = content_hashes.get(digest, 0) + 1
            _, height, width = image.shape
            sizes[f"{height}x{width}x{image.shape[0]}"] = (
                sizes.get(f"{height}x{width}x{image.shape[0]}", 0) + 1
            )
            source_path = (
                bundle.sample_metadata.paths[source_index]
                if 0 <= source_index < len(bundle.sample_metadata.paths)
                else None
            )
            if source_path is not None:
                path_sets[split_name].add(str(source_path))
            group = (
                bundle.sample_metadata.groups[source_index]
                if 0 <= source_index < len(bundle.sample_metadata.groups)
                else None
            )
            if group is not None:
                group_sets[split_name].add(group)
        class_counts[split_name] = counts

    duplicates = {digest: count for digest, count in content_hashes.items() if count > 1}
    split_overlap: list[str] = []
    if "train" in path_sets and "val" in path_sets:
        overlap = path_sets["train"] & path_sets["val"]
        split_overlap = sorted(overlap)
    group_leakage = sorted(group_sets.get("train", set()) & group_sets.get("val", set()))
    warnings: list[str] = []
    if split_overlap:
        warnings.append(f"{len(split_overlap)} path(s) appear in both train and val")
    if group_leakage:
        warnings.append(f"group leakage across train and val: {', '.join(group_leakage)}")
    if duplicates:
        warnings.append(f"{len(duplicates)} duplicate content hash(es) found")
    if corrupt:
        warnings.append(f"{len(corrupt)} unreadable sample(s) found")
    for split_name, counts in class_counts.items():
        nonzero = [count for count in counts.values() if count > 0]
        if nonzero and max(nonzero) > 2 * max(1, min(nonzero)):
            warnings.append(f"{split_name} classes are imbalanced: {counts}")

    return DataInspectionReport(
        class_counts=class_counts,
        duplicates=duplicates,
        corrupt=corrupt,
        sizes=sizes,
        split_overlap=split_overlap,
        group_leakage=group_leakage,
        warnings=warnings,
    )
