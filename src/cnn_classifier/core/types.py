"""Shared data contracts exchanged between adapters, models, and runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from torch.utils.data import Dataset

from cnn_classifier.core.errors import DataError

DatasetLike = Dataset[Any]


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: str
    split: Literal["train", "val", "test"]
    group: str | None = None
    severity: float | None = None


@dataclass(frozen=True)
class SampleMetadata:
    paths: tuple[Path, ...]
    labels: tuple[int, ...]
    groups: tuple[str | None, ...]
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class NormalizationStats:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    source: Literal["computed", "supplied", "none"]


@dataclass(frozen=True)
class ModelInputContract:
    input_shape: tuple[int, int, int]
    dtype: Literal["float32"]
    value_range: tuple[float, float]
    normalization: NormalizationStats | None
    class_names: tuple[str, ...]
    label_map: dict[str, int]
    channel_policy: Literal["rgb", "gray", "as_is"]
    stem_adaptation: str | None


@dataclass(frozen=True)
class SplitInfo:
    mode: Literal["random", "random_stratified", "group", "explicit", "train_only"]
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...] | None
    seed: int
    group_column: str | None
    fingerprint: str

    def __post_init__(self) -> None:
        all_indices = [*self.train_indices, *self.val_indices]
        if self.test_indices is not None:
            all_indices.extend(self.test_indices)
        if len(all_indices) != len(set(all_indices)):
            raise DataError("train, val, and test indices overlap")
        if any(index < 0 for index in all_indices):
            raise DataError("split indices must be non-negative")
        if not self.fingerprint:
            raise DataError("split fingerprint must not be empty")


@dataclass(frozen=True)
class DatasetBundle:
    train: DatasetLike
    val: DatasetLike
    test: DatasetLike | None
    class_names: tuple[str, ...]
    in_channels: int
    image_size: tuple[int, int]
    normalization: NormalizationStats | None
    split_info: SplitInfo
    fingerprint: str
    sample_metadata: SampleMetadata
    model_contract: ModelInputContract
    test_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.class_names:
            raise DataError("class_names must not be empty")
        if len(set(self.class_names)) != len(self.class_names):
            raise DataError("class_names must be unique")
        if self.in_channels <= 0:
            raise DataError("in_channels must be positive")
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise DataError("image_size must be positive")
