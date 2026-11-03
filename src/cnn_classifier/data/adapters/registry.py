"""Dataset adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.core.errors import ConfigError, RegistryError
from cnn_classifier.core.registry import Registry
from cnn_classifier.core.types import DatasetBundle
from cnn_classifier.data.base import DatasetAdapter

AdapterClass: TypeAlias = type[DatasetAdapter]
DATASETS: Registry[AdapterClass] = Registry("dataset")


def register_dataset(name: str) -> Callable[[AdapterClass], AdapterClass]:
    """Register a dataset adapter class under ``name``."""
    return DATASETS.register(name)


def build_dataset(config: DataConfig) -> DatasetBundle:
    """Resolve and build the configured dataset adapter."""
    try:
        adapter = DATASETS.get(config.adapter)
    except RegistryError as exc:
        raise ConfigError(f"unknown dataset adapter {config.adapter!r}") from exc
    return adapter.build(config)
