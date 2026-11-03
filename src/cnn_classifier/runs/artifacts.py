"""CSV and JSONL metrics writers."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cnn_classifier.runs.context import RunContext


@dataclass
class MetricsWriter:
    run: RunContext
    _header: list[str] | None = field(default=None, init=False)

    def _metrics_path(self) -> Path:
        return self.run.directory / "metrics.jsonl"

    def _history_path(self) -> Path:
        return self.run.directory / "history.csv"

    def append(self, metrics: Mapping[str, float], step: int) -> None:
        payload = {"step": int(step), **{key: float(value) for key, value in metrics.items()}}
        with self._metrics_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self._header is None:
            self._header = self._existing_header()
            if self._header is None:
                self._header = ["step", *sorted(metrics)]
                with self._history_path().open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=self._header)
                    writer.writeheader()
        with self._history_path().open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._header, extrasaction="ignore")
            writer.writerow(payload)

    def _existing_header(self) -> list[str] | None:
        path = self._history_path()
        if not path.is_file() or path.stat().st_size == 0:
            return None
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return None
        return header or None
