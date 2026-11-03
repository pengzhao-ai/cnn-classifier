"""Fair multi-run comparison with test-fingerprint enforcement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cnn_classifier.core.errors import RunError


@dataclass(frozen=True)
class ComparisonReport:
    rows: list[dict[str, Any]]
    warnings: list[str]
    markdown: str
    pareto_data: list[dict[str, Any]]


def _load_run(run_dir: Path) -> tuple[str, dict[str, Any]]:
    summary_path = run_dir / "summary.json"
    fingerprint_path = run_dir / "data" / "fingerprint.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"cannot read summary for {run_dir}: {exc}") from exc
    fingerprint = summary.get("test_fingerprint") or summary.get("dataset_fingerprint")
    if fingerprint_path.is_file():
        try:
            fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8")).get(
                "fingerprint", fingerprint
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RunError(f"cannot read fingerprint for {run_dir}: {exc}") from exc
    if not fingerprint:
        raise RunError(f"run {run_dir} has no test fingerprint")
    return str(fingerprint), summary


def _row(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    interval = summary.get("accuracy_ci") or [None, None]
    return {
        "run": run_dir.name,
        "accuracy": summary.get("test_accuracy", summary.get("best_metric")),
        "macro_f1": summary.get("macro_f1"),
        "ci_low": interval[0],
        "ci_high": interval[1],
        "params": summary.get("params"),
        "trainable_params": summary.get("trainable_params"),
        "train_seconds": summary.get("train_seconds"),
        "latency_ms": summary.get("latency_ms"),
        "peak_memory_mb": summary.get("peak_memory_mb"),
    }


def compare_runs(
    run_dirs: list[Path] | tuple[Path, ...],
    output_dir: Path,
) -> ComparisonReport:
    """Compare runs only when they share the same test fingerprint."""
    if not run_dirs:
        raise RunError("compare_runs requires at least one run")
    fingerprints: dict[str, str] = {}
    summaries: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in run_dirs:
        fingerprint, summary = _load_run(Path(run_dir))
        fingerprints[run_dir.name] = fingerprint
        summaries.append((Path(run_dir), summary))
    if len(set(fingerprints.values())) != 1:
        details = ", ".join(f"{name}={fingerprint}" for name, fingerprint in fingerprints.items())
        raise RunError(f"test fingerprint mismatch across runs: {details}")
    rows = sorted(
        (_row(run_dir, summary) for run_dir, summary in summaries),
        key=lambda row: float(row.get("accuracy") or 0.0),
        reverse=True,
    )
    warnings: list[str] = []
    for row in rows:
        if row["accuracy"] is None:
            warnings.append(f"{row['run']} has no accuracy")
    header = "| Run | Accuracy | Macro F1 | Params | Latency (ms) |"
    separator = "|---|---:|---:|---:|---:|"
    lines = [header, separator]
    for row in rows:
        accuracy = row["accuracy"]
        macro_f1 = row["macro_f1"]
        lines.append(
            f"| {row['run']} | "
            f"{accuracy if accuracy is not None else 'n/a'} | "
            f"{macro_f1 if macro_f1 is not None else 'n/a'} | "
            f"{row['params'] if row['params'] is not None else 'n/a'} | "
            f"{row['latency_ms'] if row['latency_ms'] is not None else 'n/a'} |"
        )
    markdown = "\n".join(lines) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    (output_dir / "comparison.json").write_text(
        json.dumps({"rows": rows, "warnings": warnings}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pareto = [
        {
            "run": row["run"],
            "accuracy": row["accuracy"],
            "params": row["params"],
            "latency_ms": row["latency_ms"],
        }
        for row in rows
    ]
    return ComparisonReport(rows=rows, warnings=warnings, markdown=markdown, pareto_data=pareto)
