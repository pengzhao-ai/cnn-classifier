"""Composite tracker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cnn_classifier.tracking.base import Tracker


class CompositeTracker:
    def __init__(self, trackers: Sequence[Tracker]) -> None:
        self.trackers = list(trackers)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)

    def close(self) -> None:
        for tracker in self.trackers:
            tracker.close()
