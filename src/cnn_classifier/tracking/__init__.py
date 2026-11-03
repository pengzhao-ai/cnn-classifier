"""Tracker construction."""

from __future__ import annotations

import contextlib

from cnn_classifier.config.schema import TrackingConfig
from cnn_classifier.runs.context import RunContext
from cnn_classifier.tracking.base import Tracker
from cnn_classifier.tracking.composite import CompositeTracker
from cnn_classifier.tracking.console import ConsoleTracker
from cnn_classifier.tracking.csv import CSVTracker
from cnn_classifier.tracking.tensorboard import TensorBoardTracker

__all__ = [
    "CSVTracker",
    "CompositeTracker",
    "ConsoleTracker",
    "TensorBoardTracker",
    "build_trackers",
]


def build_trackers(config: TrackingConfig, run: RunContext) -> CompositeTracker:
    """Build the enabled trackers for one run."""
    trackers: list[Tracker] = []
    if config.console:
        trackers.append(ConsoleTracker())
    if config.csv:
        trackers.append(CSVTracker(run))
    if config.tensorboard:
        with contextlib.suppress(RuntimeError):
            trackers.append(TensorBoardTracker(str(run.directory / "tensorboard")))
    return CompositeTracker(trackers)
