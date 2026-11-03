"""CSV/JSONL tracker."""

from __future__ import annotations

from collections.abc import Mapping

from cnn_classifier.runs.artifacts import MetricsWriter
from cnn_classifier.runs.context import RunContext


class CSVTracker:
    def __init__(self, run: RunContext) -> None:
        self.writer = MetricsWriter(run)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self.writer.append(metrics, step)

    def close(self) -> None:
        return None
