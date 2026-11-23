"""CSV and JSONL metrics writers."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
            self._write_header()
        else:
            new_keys = sorted(set(payload) - set(self._header))
            if new_keys:
                self._extend_header(new_keys)
        header = self._header
        assert header is not None
        with self._history_path().open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
            writer.writerow(payload)

    def _write_header(self) -> None:
        header = self._header
        assert header is not None
        with self._history_path().open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=header).writeheader()

    def _extend_header(self, new_keys: list[str]) -> None:
        header = self._header
        assert header is not None
        path = self._history_path()
        rows: list[dict[str, Any]] = []
        if path.is_file() and path.stat().st_size > 0:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        new_header = [*header, *new_keys]
        self._header = new_header
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=new_header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

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
