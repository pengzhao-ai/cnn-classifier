"""Dataset adapter protocol and shared bundle types."""

from __future__ import annotations

from typing import Protocol

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.core.types import DatasetBundle


class DatasetAdapter(Protocol):
    """Builds a train/val/test bundle from a data configuration."""

    def build(self, config: DataConfig) -> DatasetBundle:
        """Return the resolved datasets and their contracts."""
        ...
