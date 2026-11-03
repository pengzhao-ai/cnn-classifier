"""Shared dataset construction helpers used by the concrete adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset, Subset

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.core.errors import ConfigError, DataError
from cnn_classifier.core.hashing import stable_hash
from cnn_classifier.core.types import (
    DatasetBundle,
    ModelInputContract,
    NormalizationStats,
    SampleMetadata,
    SplitInfo,
)
from cnn_classifier.data.image_io import ChannelPolicy, read_image
from cnn_classifier.data.splits import (
    SplitIndices,
    group_split_indices,
    random_split_indices,
    stratified_split_indices,
)
from cnn_classifier.data.stats import compute_normalization
from cnn_classifier.data.transforms import EnsureTensor, build_transforms

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class RawSample:
    """A source image (path or tensor) and its logical label/split."""

    source: Any
    label: str
    split: SplitName
    group: str | None = None
    severity: float | None = None


class RawDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """Apply label mapping, image decoding, and an optional transform."""

    def __init__(
        self,
        samples: Sequence[RawSample],
        label_to_index: dict[str, int],
        channels: ChannelPolicy,
        value_range: Any,
        transform: Any = None,
    ) -> None:
        self.samples = tuple(samples)
        self.label_to_index = label_to_index
        self.channels = channels
        self.value_range = value_range
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        sample = self.samples[index]
        if isinstance(sample.source, Path):
            image = read_image(sample.source, channels=self.channels, value_range=self.value_range)
        elif torch.is_tensor(sample.source):
            image = sample.source.to(torch.float32)
        else:
            image = EnsureTensor()(sample.source)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label_to_index[sample.label], index


def resolve_class_names(
    labels: Sequence[str],
    config: DataConfig,
) -> tuple[str, ...]:
    """Resolve explicit class names or derive a stable sorted order."""
    if isinstance(config.class_names, tuple):
        names = config.class_names
        if config.num_classes is not None and len(names) != config.num_classes:
            raise ConfigError("data.class_names length must match data.num_classes")
        missing = set(labels) - set(names)
        if missing:
            raise ConfigError(f"data.class_names is missing labels: {sorted(missing)}")
        return names
    return tuple(sorted(set(labels)))


def assign_split(samples: Sequence[RawSample], config: DataConfig) -> tuple[SplitIndices, str]:
    """Resolve the configured split policy into source indices."""
    mode = config.split.mode
    if mode == "explicit":
        train = tuple(i for i, sample in enumerate(samples) if sample.split == "train")
        val = tuple(i for i, sample in enumerate(samples) if sample.split == "val")
        test = tuple(i for i, sample in enumerate(samples) if sample.split == "test")
        return SplitIndices(train, val, test or None), "explicit"
    labels = [sample.label for sample in samples]
    if mode == "random":
        return random_split_indices(labels, config.split.val_fraction, config.split.seed), "random"
    if mode == "random_stratified":
        return (
            stratified_split_indices(labels, config.split.val_fraction, config.split.seed),
            "random_stratified",
        )
    if mode == "group":
        groups = [sample.group for sample in samples]
        if any(group is None for group in groups):
            raise DataError("group split requires a group for every sample")
        return (
            group_split_indices(
                [str(group) for group in groups],
                config.split.val_fraction,
                config.split.seed,
            ),
            "group",
        )
    if mode == "train_only":
        return SplitIndices(tuple(range(len(samples))), ()), "train_only"
    raise ConfigError(f"unsupported split mode {mode!r}")


def resolve_normalization(
    config: DataConfig,
    base: Dataset[Any],
    train_indices: tuple[int, ...],
) -> NormalizationStats | None:
    """Resolve computed, supplied, or disabled normalization statistics."""
    if config.normalization.mode == "none":
        return None
    if config.normalization.mode == "supplied":
        if config.normalization.mean is None or config.normalization.std is None:
            raise ConfigError("supplied normalization requires mean and std")
        return NormalizationStats(
            mean=config.normalization.mean,
            std=config.normalization.std,
            source="supplied",
        )
    if not train_indices:
        raise DataError("cannot compute normalization without training samples")
    return compute_normalization(Subset(base, train_indices))


def build_bundle(
    samples: Sequence[RawSample],
    config: DataConfig,
    *,
    class_names: Sequence[str],
    split: SplitIndices,
    mode: str,
    fingerprint: str,
    group_column: str | None = None,
) -> DatasetBundle:
    """Build the final bundle, computing normalization before augmentation."""
    names = tuple(class_names)
    label_to_index = {name: index for index, name in enumerate(names)}
    base = RawDataset(samples, label_to_index, config.channels, config.value_range)
    normalization = resolve_normalization(config, base, split.train_indices)
    train_base = RawDataset(
        samples,
        label_to_index,
        config.channels,
        config.value_range,
        transform=build_transforms(config, normalization, train=True),
    )
    eval_base = RawDataset(
        samples,
        label_to_index,
        config.channels,
        config.value_range,
        transform=build_transforms(config, normalization, train=False),
    )
    test_indices = split.test_indices
    metadata = SampleMetadata(
        paths=tuple(
            sample.source if isinstance(sample.source, Path) else Path(f"<sample:{index}>")
            for index, sample in enumerate(samples)
        ),
        labels=tuple(label_to_index[sample.label] for sample in samples),
        groups=tuple(sample.group for sample in samples),
        source_indices=tuple(range(len(samples))),
    )
    contract = ModelInputContract(
        input_shape=(3 if config.channels == "rgb" else 1, *config.image_size),
        dtype="float32",
        value_range=(0.0, 1.0),
        normalization=normalization,
        class_names=names,
        label_map=label_to_index,
        channel_policy=config.channels,
        stem_adaptation=None,
    )
    return DatasetBundle(
        train=Subset(train_base, split.train_indices),
        val=Subset(eval_base, split.val_indices),
        test=Subset(eval_base, test_indices) if test_indices else None,
        class_names=names,
        in_channels=int(contract.input_shape[0]),
        image_size=config.image_size,
        normalization=normalization,
        split_info=SplitInfo(
            mode=mode,  # type: ignore[arg-type]
            train_indices=split.train_indices,
            val_indices=split.val_indices,
            test_indices=test_indices,
            seed=config.split.seed,
            group_column=group_column,
            fingerprint=fingerprint,
        ),
        fingerprint=fingerprint,
        sample_metadata=metadata,
        model_contract=contract,
    )


def stable_sample_fingerprint(samples: Sequence[RawSample], seed: int) -> str:
    """Fingerprint a sample list without requiring source files to be read."""
    payload = [
        {
            "source": str(sample.source) if isinstance(sample.source, Path) else "<tensor>",
            "label": sample.label,
            "split": sample.split,
            "group": sample.group,
            "severity": sample.severity,
        }
        for sample in samples
    ]
    return stable_hash({"samples": payload, "seed": seed})
