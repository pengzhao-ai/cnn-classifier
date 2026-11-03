"""DataLoader construction with device-aware worker defaults."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from cnn_classifier.config.schema import RuntimeConfig, TrainingConfig
from cnn_classifier.core.seeding import make_generator
from cnn_classifier.core.types import DatasetBundle


@dataclass(frozen=True)
class DataLoaderBundle:
    train: DataLoader[Any]
    val: DataLoader[Any]
    test: DataLoader[Any] | None


def _worker_count(runtime: RuntimeConfig, device: torch.device) -> int:
    if isinstance(runtime.num_workers, int):
        return runtime.num_workers
    if device.type == "mps":
        return 0
    if device.type == "cuda" and platform.system() == "Linux":
        return 4
    return 2


def _pin_memory(runtime: RuntimeConfig, device: torch.device) -> bool:
    if isinstance(runtime.pin_memory, bool):
        return runtime.pin_memory
    return device.type == "cuda"


def build_loaders(
    bundle: DatasetBundle,
    training: TrainingConfig,
    runtime: RuntimeConfig,
    device: torch.device,
) -> DataLoaderBundle:
    """Build train/val/test loaders with seeded shuffling and worker policy."""
    workers = _worker_count(runtime, device)
    pin_memory = _pin_memory(runtime, device)
    persistent = runtime.persistent_workers and workers > 0
    train = DataLoader(
        bundle.train,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        generator=make_generator(bundle.split_info.seed),
    )
    val = DataLoader(
        bundle.val,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test = (
        DataLoader(
            bundle.test,
            batch_size=training.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
        if bundle.test is not None
        else None
    )
    return DataLoaderBundle(train=train, val=val, test=test)
