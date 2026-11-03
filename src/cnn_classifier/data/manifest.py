"""CSV manifest parsing and deterministic dataset fingerprints."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cnn_classifier.core.errors import DataError
from cnn_classifier.core.hashing import stable_hash
from cnn_classifier.core.types import SampleRecord

SplitName = Literal["train", "val", "test"]
_REQUIRED_COLUMNS = ("path", "label", "split")


@dataclass(frozen=True)
class Manifest:
    records: tuple[SampleRecord, ...]

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(sorted({record.label for record in self.records}))

    @classmethod
    def from_csv(
        cls,
        path: Path,
        image_root: Path,
        columns: dict[str, str],
    ) -> Manifest:
        """Read a CSV manifest and validate every referenced image path."""
        for logical in _REQUIRED_COLUMNS:
            if logical not in columns:
                raise DataError(f"manifest columns must include {logical!r}")
        records: list[SampleRecord] = []
        try:
            handle = path.open("r", encoding="utf-8", newline="")
        except OSError as exc:
            raise DataError(f"cannot read manifest {path}: {exc}") from exc
        with handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                raw_path = row.get(columns["path"], "").strip()
                label = row.get(columns["label"], "").strip()
                split = row.get(columns["split"], "").strip()
                if not raw_path or not label:
                    raise DataError(f"manifest row {row_number} is missing path or label")
                if split not in {"train", "val", "test"}:
                    raise DataError(f"manifest row {row_number} has invalid split {split!r}")
                image_path = Path(raw_path)
                if not image_path.is_absolute():
                    image_path = image_root / image_path
                if not image_path.is_file():
                    raise DataError(f"manifest row {row_number} references missing file {image_path}")
                group = None
                if "group" in columns:
                    group = row.get(columns["group"], "").strip() or None
                severity = None
                if "severity" in columns:
                    raw_severity = row.get(columns["severity"], "").strip()
                    if raw_severity:
                        try:
                            severity = float(raw_severity)
                        except ValueError as exc:
                            raise DataError(
                                f"manifest row {row_number} has invalid severity {raw_severity!r}"
                            ) from exc
                records.append(
                    SampleRecord(
                        path=image_path,
                        label=label,
                        split=split,  # type: ignore[arg-type]
                        group=group,
                        severity=severity,
                    )
                )
        if not records:
            raise DataError(f"manifest {path} contains no records")
        return cls(records=tuple(records))


def fingerprint_records(records: tuple[SampleRecord, ...]) -> str:
    """Return a deterministic fingerprint independent of manifest row order."""
    payload = [
        {
            "path": str(record.path),
            "label": record.label,
            "split": record.split,
            "group": record.group,
            "severity": record.severity,
        }
        for record in sorted(records, key=lambda item: str(item.path))
    ]
    return stable_hash(payload)
