"""Deterministic split policies and group-leakage checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cnn_classifier.core.errors import DataError
from cnn_classifier.core.types import SplitInfo


@dataclass(frozen=True)
class SplitIndices:
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...] | None = None


def _validate_fraction(val_fraction: float) -> None:
    if not 0.0 < val_fraction < 1.0:
        raise DataError("val_fraction must be in (0, 1)")


def random_split_indices(
    labels: Sequence[Any],
    val_fraction: float,
    seed: int,
) -> SplitIndices:
    """Split indices with a seeded permutation, without stratification."""
    _validate_fraction(val_fraction)
    count = len(labels)
    if count < 2:
        return SplitIndices(tuple(range(count)), ())
    rng = np.random.default_rng(seed)
    order = rng.permutation(count).tolist()
    val_count = min(count - 1, max(1, round(count * val_fraction)))
    val = tuple(sorted(order[:val_count]))
    train = tuple(sorted(order[val_count:]))
    return SplitIndices(train_indices=train, val_indices=val)


def stratified_split_indices(
    labels: Sequence[Any],
    val_fraction: float,
    seed: int,
) -> SplitIndices:
    """Split each class proportionally, then combine the class splits."""
    _validate_fraction(val_fraction)
    by_class: dict[Any, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    for label in sorted(by_class):
        indices = np.array(by_class[label])
        order = rng.permutation(len(indices))
        shuffled = indices[order].tolist()
        val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_fraction)))
        if len(shuffled) == 1:
            val_count = 0
        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])
    return SplitIndices(tuple(sorted(train)), tuple(sorted(val)))


def group_split_indices(
    groups: Sequence[str],
    val_fraction: float,
    seed: int,
) -> SplitIndices:
    """Assign whole groups to train or validation, never splitting a group."""
    _validate_fraction(val_fraction)
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_to_indices[group].append(index)
    unique_groups = sorted(group_to_indices)
    if len(unique_groups) < 2:
        raise DataError("group split requires at least two groups")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_groups)).tolist()
    target = round(len(groups) * val_fraction)
    val_groups: list[str] = []
    val_count = 0
    for position in order:
        if val_count >= target and val_groups:
            break
        if len(val_groups) >= len(unique_groups) - 1:
            break
        group = unique_groups[position]
        val_groups.append(group)
        val_count += len(group_to_indices[group])
    val_group_set = set(val_groups)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for group in unique_groups:
        target_list = val_indices if group in val_group_set else train_indices
        target_list.extend(group_to_indices[group])
    return SplitIndices(tuple(sorted(train_indices)), tuple(sorted(val_indices)))


def assert_no_group_overlap(split: SplitIndices, groups: Sequence[str]) -> None:
    """Raise when a group appears in both train and validation."""
    train_groups = {groups[index] for index in split.train_indices}
    val_groups = {groups[index] for index in split.val_indices}
    overlap = train_groups & val_groups
    if overlap:
        raise DataError(f"groups leak across train and validation: {sorted(overlap)}")


def make_split_info(
    split: SplitIndices,
    *,
    mode: str,
    seed: int,
    group_column: str | None,
    fingerprint: str,
) -> SplitInfo:
    """Package a split for the dataset bundle."""
    return SplitInfo(
        mode=mode,  # type: ignore[arg-type]
        train_indices=split.train_indices,
        val_indices=split.val_indices,
        test_indices=split.test_indices,
        seed=seed,
        group_column=group_column,
        fingerprint=fingerprint,
    )
