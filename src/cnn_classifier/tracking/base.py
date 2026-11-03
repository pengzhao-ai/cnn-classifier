"""Tracker protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class Tracker(Protocol):
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Record one step of metrics."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...
