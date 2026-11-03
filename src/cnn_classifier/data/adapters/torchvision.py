"""Torchvision dataset adapter for CIFAR-10 and similar built-ins."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchvision
from torch.utils.data import Dataset, Subset

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.core.errors import ConfigError, DataError
from cnn_classifier.core.hashing import stable_hash
from cnn_classifier.core.types import (
    DatasetBundle,
    ModelInputContract,
    SampleMetadata,
    SplitInfo,
)
from cnn_classifier.data.adapters._common import resolve_normalization
from cnn_classifier.data.adapters.registry import register_dataset
from cnn_classifier.data.splits import (
    SplitIndices,
    random_split_indices,
    stratified_split_indices,
)
from cnn_classifier.data.transforms import EnsureTensor, build_transforms


class _TripleDataset(Dataset[tuple[Any, int, int]]):
    def __init__(self, base: Dataset[Any]) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        image, label = self.base[index]
        return image, int(label), index


class _TensorTripleDataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(self, base: Dataset[Any]) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, label, source_index = self.base[index]
        return EnsureTensor()(image), int(label), int(source_index)


class _TransformedTripleDataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(self, base: Dataset[Any], transform: Any) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, label, source_index = self.base[index]
        return self.transform(image), int(label), int(source_index)


def _dataset_targets(dataset: Dataset[Any]) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        raise DataError("torchvision dataset exposes neither targets nor labels")
    return [int(value) for value in targets]


def _sample_limit(config: DataConfig, key: str) -> int | None:
    value = config.adapter_params.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"adapter_params.{key} must be a positive integer")
    return value


@register_dataset("torchvision")
class TorchvisionAdapter:
    """Wrap a torchvision classification dataset with the package contracts."""

    @staticmethod
    def build(config: DataConfig) -> DatasetBundle:
        if config.root is None:
            raise ConfigError("torchvision adapter requires data.root")
        dataset_name = str(config.adapter_params.get("dataset", "CIFAR10"))
        dataset_class = getattr(torchvision.datasets, dataset_name, None)
        if dataset_class is None:
            raise ConfigError(f"unknown torchvision dataset {dataset_name!r}")
        download = bool(config.adapter_params.get("download", True))
        try:
            train_base = dataset_class(root=str(config.root), train=True, download=download)
            test_base = dataset_class(root=str(config.root), train=False, download=download)
        except (OSError, RuntimeError) as exc:
            raise DataError(f"cannot load torchvision dataset {dataset_name}: {exc}") from exc
        class_names = tuple(str(name) for name in train_base.classes)
        if isinstance(config.class_names, tuple) and tuple(config.class_names) != class_names:
            raise ConfigError("data.class_names must match the torchvision class order")
        label_map = {name: index for index, name in enumerate(class_names)}
        targets = _dataset_targets(train_base)
        test_targets = _dataset_targets(test_base)[: len(test_base)]
        test_fingerprint = stable_hash(
            {
                "adapter": "torchvision",
                "dataset": dataset_name,
                "test_size": len(test_targets),
                "test_targets": test_targets,
                "class_names": class_names,
            }
        )
        if config.split.mode == "random":
            split = random_split_indices(targets, config.split.val_fraction, config.split.seed)
            mode = "random"
        elif config.split.mode == "random_stratified":
            split = stratified_split_indices(targets, config.split.val_fraction, config.split.seed)
            mode = "random_stratified"
        else:
            raise ConfigError("torchvision adapter supports random or random_stratified splits")
        max_train = _sample_limit(config, "max_train_samples")
        max_val = _sample_limit(config, "max_val_samples")
        if max_train is not None or max_val is not None:
            split = SplitIndices(
                split.train_indices[:max_train] if max_train is not None else split.train_indices,
                split.val_indices[:max_val] if max_val is not None else split.val_indices,
            )
        train_triple = _TripleDataset(train_base)
        stats_source = _TensorTripleDataset(train_triple)
        normalization = resolve_normalization(config, stats_source, split.train_indices)
        train_transform = build_transforms(config, normalization, train=True)
        eval_transform = build_transforms(config, normalization, train=False)
        train = Subset(
            _TransformedTripleDataset(train_triple, train_transform), split.train_indices
        )
        val = Subset(_TransformedTripleDataset(train_triple, eval_transform), split.val_indices)
        test = _TransformedTripleDataset(_TripleDataset(test_base), eval_transform)
        fingerprint = stable_hash(
            {
                "adapter": "torchvision",
                "dataset": dataset_name,
                "train_size": len(train_base),
                "test_size": len(test_base),
                "train_samples": len(split.train_indices),
                "val_samples": len(split.val_indices),
                "seed": config.split.seed,
            }
        )
        contract = ModelInputContract(
            input_shape=(3, *config.image_size),
            dtype="float32",
            value_range=(0.0, 1.0),
            normalization=normalization,
            class_names=class_names,
            label_map=label_map,
            channel_policy="rgb",
            stem_adaptation=None,
        )
        metadata = SampleMetadata(
            paths=tuple(Path(f"<torchvision:{index}>") for index in range(len(train_base))),
            labels=tuple(targets),
            groups=tuple(None for _ in targets),
            source_indices=tuple(range(len(train_base))),
        )
        return DatasetBundle(
            train=train,
            val=val,
            test=test,
            class_names=class_names,
            in_channels=3,
            image_size=config.image_size,
            normalization=normalization,
            split_info=SplitInfo(
                mode=mode,  # type: ignore[arg-type]
                train_indices=split.train_indices,
                val_indices=split.val_indices,
                test_indices=None,
                seed=config.split.seed,
                group_column=None,
                fingerprint=fingerprint,
            ),
            fingerprint=fingerprint,
            sample_metadata=metadata,
            model_contract=contract,
            test_fingerprint=test_fingerprint,
        )
