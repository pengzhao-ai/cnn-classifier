"""Self-contained experiment report generated only from saved artifacts."""

from __future__ import annotations

import base64
import csv
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "report.html.j2"

_CONFIG_SECTIONS = (
    "experiment",
    "data",
    "model",
    "training",
    "evaluation",
    "tracking",
    "report",
    "runtime",
)

_HISTORY_LABELS = {
    "epoch": "Epoch",
    "train_loss": "Train loss",
    "val_loss": "Validation loss",
    "train_accuracy": "Train accuracy",
    "val_accuracy": "Validation accuracy",
    "lr": "Learning rate",
}

_FIGURE_CAPTIONS = {
    "training_history": "Training history",
    "confusion_matrix": "Confusion matrix",
    "per_class": "Per-class metrics",
    "calibration": "Calibration",
    "errors": "Misclassifications",
    "error_gallery": "Misclassifications",
    "model_card": "Model card",
    "model_summary": "Model summary",
    "dataset_overview": "Dataset overview",
    "augmentation_gallery": "Augmentation samples",
    "comparison": "Model comparison",
}

_EVALUATION_CANDIDATES = (
    "evaluation.json",
    "test_evaluation.json",
    "evaluation/test.json",
)


@dataclass(frozen=True)
class _Figure:
    name: str
    caption: str
    data_uri: str


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_history(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                parsed: dict[str, float] = {}
                for key, value in row.items():
                    if key is None or value is None or value == "":
                        continue
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        continue
                if parsed:
                    rows.append(parsed)
    except OSError:
        return []
    return rows


def _read_metrics_jsonl(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            parsed = {
                str(key): float(value)
                for key, value in payload.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if parsed:
                rows.append(parsed)
    except OSError:
        return []
    return rows


def _read_predictions(run_dir: Path) -> list[dict[str, str]]:
    directory = run_dir / "predictions"
    if not directory.is_dir():
        return []
    rows: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.npz")):
        try:
            with np.load(path) as archive:
                labels = archive.get("y_true", None)
                predictions = archive.get("y_pred", None)
                samples = len(labels) if labels is not None else None
                if samples is None and archive.files:
                    samples = len(archive[archive.files[0]])
                accuracy = (
                    float(np.mean(labels == predictions))
                    if labels is not None
                    and predictions is not None
                    and len(labels) == len(predictions)
                    and len(labels) > 0
                    else None
                )
        except (OSError, ValueError, KeyError):
            continue
        rows.append(
            {
                "name": path.name,
                "samples": _format(samples),
                "accuracy": _format_metric(accuracy) if accuracy is not None else "—",
            }
        )
    return rows


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _format_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _flatten(prefix: str, payload: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                rows.extend(_flatten(label, value))
            else:
                rows.append((label, _format(value)))
    elif isinstance(payload, list):
        rows.append((prefix or "value", _format(payload)))
    else:
        rows.append((prefix or "value", _format(payload)))
    return rows


def _config_sections(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not config:
        return []
    sections: list[dict[str, Any]] = []
    for name in _CONFIG_SECTIONS:
        value = config.get(name)
        if not value:
            continue
        rows = _flatten("", value)
        sections.append({"title": name.replace("_", " ").title(), "rows": rows})
    extras = [
        (key, value)
        for key, value in config.items()
        if key not in _CONFIG_SECTIONS and key != "defaults"
    ]
    if extras:
        sections.append(
            {
                "title": "Other",
                "rows": _flatten("", dict(extras)),
            }
        )
    return sections


def _contract_rows(contract: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not contract:
        return []
    rows: list[tuple[str, str]] = []
    for key, value in contract.items():
        if key == "label_map" and isinstance(value, dict):
            rows.append(
                ("label_map", ", ".join(f"{name}={index}" for name, index in value.items()))
            )
        elif key == "normalization" and isinstance(value, dict):
            mean = value.get("mean")
            std = value.get("std")
            rows.append(("normalization.mean", _format(mean)))
            rows.append(("normalization.std", _format(std)))
            rows.append(("normalization.source", _format(value.get("source"))))
        else:
            rows.append((key, _format(value)))
    return rows


def _split_section(
    fingerprint: dict[str, Any] | None,
) -> tuple[list[dict[str, str]], list[str]]:
    if not fingerprint:
        return [], ["Split integrity: not evaluated (no fingerprint artifact found)."]
    rows: list[dict[str, str]] = []
    splits = fingerprint.get("splits")
    if isinstance(splits, dict):
        for name, details in splits.items():
            if isinstance(details, dict):
                rows.append(
                    {
                        "split": str(name),
                        "count": _format(details.get("count")),
                        "fingerprint": _format(details.get("fingerprint")),
                    }
                )
            else:
                rows.append({"split": str(name), "count": _format(details), "fingerprint": "—"})
    notes: list[str] = []
    for key in ("split_mode", "group_overlap", "duplicate_overlap", "dataset"):
        if key in fingerprint:
            notes.append(f"{key.replace('_', ' ').capitalize()}: {_format(fingerprint[key])}")
    if not rows:
        rows = [
            {"split": key, "count": _format(value), "fingerprint": "—"}
            for key, value in fingerprint.items()
        ]
        notes = []
    return rows, notes


def _model_section(
    summary: dict[str, Any] | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not summary:
        return [], []
    modules = summary.get("modules")
    rows: list[dict[str, str]] = []
    if isinstance(modules, list):
        for module in modules[:24]:
            if isinstance(module, dict):
                rows.append(
                    {
                        "name": _format(module.get("name")),
                        "type": _format(module.get("type")),
                        "params": _format(module.get("params")),
                    }
                )
    facts = [
        {"label": "Total parameters", "value": _format(summary.get("total_params"))},
        {"label": "Trainable parameters", "value": _format(summary.get("trainable_params"))},
        {"label": "Input shape", "value": _format(summary.get("input_shape"))},
        {"label": "Output shape", "value": _format(summary.get("output_shape"))},
    ]
    return facts, rows


def _history_section(
    history: list[dict[str, float]], summary: dict[str, Any] | None
) -> dict[str, Any]:
    if not history:
        return {
            "rows": [],
            "columns": [],
            "best_epoch": None,
            "best_val_accuracy": None,
            "epochs_run": None,
        }
    columns = [
        (key, label) for key, label in _HISTORY_LABELS.items() if any(key in row for row in history)
    ]
    rows = []
    for row in history:
        rows.append(
            {
                key: (
                    str(int(row[key])) if key == "epoch" and key in row else _format(row.get(key))
                )
                for key, _ in columns
            }
        )
    best_epoch: Any = summary.get("best_epoch") if summary else None
    best_accuracy: Any = summary.get("best_metric") if summary else None
    if best_accuracy is None and any("val_accuracy" in row for row in history):
        best_row = max(history, key=lambda item: item.get("val_accuracy", float("-inf")))
        best_accuracy = best_row.get("val_accuracy")
        best_epoch = int(best_row.get("epoch", 0))
    return {
        "rows": rows,
        "columns": [{"key": key, "label": label} for key, label in columns],
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "epochs_run": len(history),
    }


def _find_evaluation(run_dir: Path) -> dict[str, Any] | None:
    for candidate in _EVALUATION_CANDIDATES:
        payload = _read_json(run_dir / candidate)
        if payload is not None:
            return payload
    return None


def _evaluation_sections(
    evaluation: dict[str, Any] | None,
    contract: dict[str, Any] | None,
) -> dict[str, Any]:
    if not evaluation:
        return {
            "available": False,
            "metrics": [],
            "per_class": [],
            "confusion": None,
            "class_names": [],
            "confidence": None,
            "latency": [],
            "split": None,
        }
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict):
        metrics = evaluation
    rows = [
        {"label": "Accuracy", "value": _format_metric(metrics.get("accuracy"))},
        {"label": "Top-k accuracy", "value": _format_metric(metrics.get("top_k_accuracy"))},
        {"label": "Macro F1", "value": _format_metric(metrics.get("macro_f1"))},
        {"label": "NLL", "value": _format_metric(metrics.get("nll"))},
        {"label": "ECE", "value": _format_metric(metrics.get("ece"))},
        {"label": "MCE", "value": _format_metric(metrics.get("mce"))},
        {"label": "Brier score", "value": _format_metric(metrics.get("brier"))},
        {"label": "Support", "value": _format_metric(metrics.get("support"))},
    ]
    calibration = metrics.get("calibration")
    if isinstance(calibration, dict):
        rows[4] = {"label": "ECE", "value": _format_metric(calibration.get("ece"))}
        rows[5] = {"label": "MCE", "value": _format_metric(calibration.get("mce"))}
        rows[6] = {"label": "Brier score", "value": _format_metric(calibration.get("brier"))}
    rows = [row for row in rows if row["value"] != "—"]

    per_class_rows: list[dict[str, str]] = []
    per_class = metrics.get("per_class")
    if isinstance(per_class, dict):
        for name, values in per_class.items():
            if isinstance(values, dict):
                per_class_rows.append(
                    {
                        "name": str(name),
                        "precision": _format_metric(values.get("precision")),
                        "recall": _format_metric(values.get("recall")),
                        "f1": _format_metric(values.get("f1")),
                        "support": _format_metric(values.get("support")),
                    }
                )

    confusion = metrics.get("confusion_matrix")
    if not isinstance(confusion, list):
        confusion = None
    names = []
    if contract and isinstance(contract.get("class_names"), list):
        names = [str(name) for name in contract["class_names"]]
    elif per_class:
        names = [row["name"] for row in per_class_rows]
    if confusion and not names:
        names = [str(index) for index in range(len(confusion))]

    confidence = None
    intervals = evaluation.get("confidence_intervals")
    if isinstance(intervals, dict) and isinstance(intervals.get("accuracy"), dict):
        interval = intervals["accuracy"]
        confidence = (
            f"{_format_metric(interval.get('confidence'))} CI "
            f"[{_format_metric(interval.get('low'))}, {_format_metric(interval.get('high'))}]"
        )
    elif isinstance(evaluation.get("accuracy_ci"), list):
        low, high = evaluation["accuracy_ci"]
        confidence = f"95% CI [{_format_metric(low)}, {_format_metric(high)}]"

    latency_rows: list[dict[str, str]] = []
    latency = evaluation.get("latency")
    if isinstance(latency, dict):
        for key, label in (
            ("mean_ms", "Mean latency (ms)"),
            ("p50_ms", "p50 latency (ms)"),
            ("p95_ms", "p95 latency (ms)"),
            ("throughput", "Throughput (images/s)"),
        ):
            if key in latency:
                latency_rows.append({"label": label, "value": _format_metric(latency[key])})

    return {
        "available": True,
        "metrics": rows,
        "per_class": per_class_rows,
        "confusion": confusion,
        "class_names": names,
        "confidence": confidence,
        "latency": latency_rows,
        "split": evaluation.get("split"),
    }


def _collect_figures(run_dir: Path) -> list[_Figure]:
    directory = run_dir / "figures"
    if not directory.is_dir():
        return []
    by_stem: dict[str, dict[str, Path]] = {}
    for path in sorted(directory.iterdir()):
        suffix = path.suffix.lower()
        if suffix in {".png", ".svg"}:
            by_stem.setdefault(path.stem, {})[suffix] = path
    figures: list[_Figure] = []
    for stem in sorted(by_stem):
        candidates = by_stem[stem]
        selected = candidates.get(".png") or candidates.get(".svg")
        if selected is None:
            continue
        mime = mimetypes.guess_type(selected.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(selected.read_bytes()).decode("ascii")
        figures.append(
            _Figure(
                name=selected.name,
                caption=_FIGURE_CAPTIONS.get(stem, stem.replace("_", " ").capitalize()),
                data_uri=f"data:{mime};base64,{encoded}",
            )
        )
    return figures


def _runtime_rows(
    environment: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    git: dict[str, Any] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if environment:
        for key, label in (
            ("python", "Python"),
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("platform", "Platform"),
            ("device", "Device"),
        ):
            if key in environment:
                rows.append({"label": label, "value": _format(environment[key])})
    if summary:
        if "device" in summary:
            rows.append({"label": "Training device", "value": _format(summary["device"])})
        if "config_hash" in summary:
            rows.append({"label": "Config hash", "value": _format(summary["config_hash"])})
        if "epochs_run" in summary:
            rows.append({"label": "Epochs run", "value": _format(summary["epochs_run"])})
    if git:
        if git.get("commit"):
            dirty = " (dirty)" if git.get("dirty") else ""
            rows.append({"label": "Git commit", "value": f"{git['commit']}{dirty}"})
        if git.get("branch"):
            rows.append({"label": "Git branch", "value": _format(git["branch"])})
    return rows


def _reproduction_commands(run_dir: Path) -> list[str]:
    return [
        f"cnncls train -c <config.yaml>  # produced {run_dir.name}",
        f"cnncls evaluate --run {run_dir} --checkpoint {run_dir / 'checkpoints' / 'best.pt'}",
        f"cnncls report --run {run_dir} --format html",
        f"cnncls compare {run_dir} <other-run>",
    ]


def _build_context(run_dir: Path) -> dict[str, Any]:
    config = _read_yaml(run_dir / "config.resolved.yaml")
    environment = _read_json(run_dir / "environment.json")
    status = _read_json(run_dir / "status.json")
    summary = _read_json(run_dir / "summary.json")
    contract = _read_json(run_dir / "data" / "model_contract.json")
    fingerprint = _read_json(run_dir / "data" / "fingerprint.json")
    model_summary = _read_json(run_dir / "model_summary.json")
    git = _read_json(run_dir / "git.json")
    evaluation = _find_evaluation(run_dir)
    history = _read_history(run_dir / "history.csv")
    if not history:
        history = _read_metrics_jsonl(run_dir / "metrics.jsonl")
    prediction_rows = _read_predictions(run_dir)
    split_rows, split_notes = _split_section(fingerprint)
    model_facts, module_rows = _model_section(model_summary)
    history_data = _history_section(history, summary)
    evaluation_data = _evaluation_sections(evaluation, contract)
    figures = _collect_figures(run_dir)
    figure_groups: dict[str, list[dict[str, str]]] = {
        "training": [],
        "confusion": [],
        "calibration": [],
        "errors": [],
        "other": [],
    }
    for figure in figures:
        payload = {
            "name": figure.name,
            "caption": figure.caption,
            "data_uri": figure.data_uri,
        }
        if "training_history" in figure.name:
            figure_groups["training"].append(payload)
        elif "confusion_matrix" in figure.name:
            figure_groups["confusion"].append(payload)
        elif "calibration" in figure.name:
            figure_groups["calibration"].append(payload)
        elif "error" in figure.name:
            figure_groups["errors"].append(payload)
        else:
            figure_groups["other"].append(payload)

    experiment = config.get("experiment") if config else None
    run_name = (
        str(experiment.get("name"))
        if isinstance(experiment, dict) and experiment.get("name")
        else run_dir.name
    )
    cards: list[dict[str, str]] = []
    if evaluation_data["available"]:
        metrics = evaluation.get("metrics") if evaluation else {}
        if not isinstance(metrics, dict):
            metrics = evaluation or {}
        cards.append({"label": "Test accuracy", "value": _format_metric(metrics.get("accuracy"))})
        cards.append({"label": "Macro F1", "value": _format_metric(metrics.get("macro_f1"))})
        if evaluation_data["confidence"]:
            cards.append({"label": "Accuracy interval", "value": evaluation_data["confidence"]})
    if model_summary and model_summary.get("total_params") is not None:
        cards.append({"label": "Parameters", "value": _format(model_summary["total_params"])})
    if history_data["best_val_accuracy"] is not None:
        cards.append(
            {
                "label": "Best validation accuracy",
                "value": _format_metric(history_data["best_val_accuracy"]),
            }
        )
    if evaluation_data["latency"]:
        cards.append(
            {
                "label": "Latency",
                "value": next(
                    row["value"] for row in evaluation_data["latency"] if "Mean" in row["label"]
                )
                + " ms",
            }
        )
    cards = [card for card in cards if card["value"] not in {"—", ""}]

    return {
        "title": f"Experiment report — {run_name}",
        "run_name": run_name,
        "run_directory": str(run_dir),
        "status": _format(status.get("status")) if status else "unknown",
        "config_hash": _format(summary.get("config_hash")) if summary else "—",
        "cards": cards,
        "config_sections": _config_sections(config),
        "contract_rows": _contract_rows(contract),
        "split_rows": split_rows,
        "split_notes": split_notes,
        "model_facts": model_facts,
        "module_rows": module_rows,
        "history": history_data,
        "evaluation": evaluation_data,
        "figures": [
            {"name": figure.name, "caption": figure.caption, "data_uri": figure.data_uri}
            for figure in figures
        ],
        "figure_groups": figure_groups,
        "prediction_rows": prediction_rows,
        "runtime_rows": _runtime_rows(environment, summary, git),
        "reproduction": _reproduction_commands(run_dir),
    }


def build_markdown_report(run_dir: Path) -> str:
    """Return a Markdown report with the same numbers as the HTML report."""
    context = _build_context(run_dir)
    lines: list[str] = [
        f"# Experiment report — {context['run_name']}",
        "",
        f"- Run directory: `{context['run_directory']}`",
        f"- Status: {context['status']}",
        f"- Config hash: {context['config_hash']}",
        "",
        "## Metrics",
        "",
    ]
    evaluation = context["evaluation"]
    if evaluation["available"]:
        lines.extend(["| Metric | Value |", "| --- | --- |"])
        lines.extend(f"| {row['label']} | {row['value']} |" for row in evaluation["metrics"])
        if evaluation["confidence"]:
            lines.append(f"| Accuracy interval | {evaluation['confidence']} |")
    else:
        lines.append("Test metrics: not evaluated (no evaluation artifact found).")
    if context["prediction_rows"]:
        lines.extend(
            [
                "",
                "| Prediction archive | Samples | Saved accuracy |",
                "| --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {row['name']} | {row['samples']} | {row['accuracy']} |"
            for row in context["prediction_rows"]
        )
    else:
        lines.extend(["", "Saved predictions: not evaluated."])
    lines.extend(["", "## Configuration", ""])
    if context["config_sections"]:
        for section in context["config_sections"]:
            lines.append(f"### {section['title']}")
            lines.append("")
            lines.extend(f"- `{key}`: {value}" for key, value in section["rows"])
            lines.append("")
    else:
        lines.append("Configuration: not evaluated (no resolved config found).")
        lines.append("")
    lines.extend(["## Data contract", ""])
    if context["contract_rows"]:
        lines.extend(f"- `{key}`: {value}" for key, value in context["contract_rows"])
    else:
        lines.append("Data contract: not evaluated.")
    lines.extend(["", "## Split integrity", ""])
    if context["split_rows"]:
        lines.extend(["| Split | Count | Fingerprint |", "| --- | --- | --- |"])
        lines.extend(
            f"| {row['split']} | {row['count']} | {row['fingerprint']} |"
            for row in context["split_rows"]
        )
        lines.extend(f"\n{note}" for note in context["split_notes"])
    else:
        lines.extend(context["split_notes"] or ["Split integrity: not evaluated."])
    lines.extend(["", "## Model", ""])
    if context["model_facts"]:
        lines.extend(f"- {row['label']}: {row['value']}" for row in context["model_facts"])
        if context["module_rows"]:
            lines.extend(["", "| Module | Type | Parameters |", "| --- | --- | --- |"])
            lines.extend(
                f"| {row['name']} | {row['type']} | {row['params']} |"
                for row in context["module_rows"]
            )
    else:
        lines.append("Model summary: not evaluated.")
    history = context["history"]
    lines.extend(["", "## Training", ""])
    if history["rows"]:
        headers = [column["label"] for column in history["columns"]]
        lines.append(f"Epochs run: {history['epochs_run']}")
        if history["best_val_accuracy"] is not None:
            lines.append(
                f"Best validation accuracy: {_format_metric(history['best_val_accuracy'])} "
                f"(epoch {_format(history['best_epoch'])})"
            )
        lines.append("")
        lines.extend(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
        )
        lines.extend(
            "| " + " | ".join(row[column["key"]] for column in history["columns"]) + " |"
            for row in history["rows"]
        )
    else:
        lines.append("Training history: not evaluated.")
    lines.extend(["", "## Test metrics", ""])
    if evaluation["per_class"]:
        lines.extend(
            ["| Class | Precision | Recall | F1 | Support |", "| --- | --- | --- | --- | --- |"]
        )
        lines.extend(
            f"| {row['name']} | {row['precision']} | {row['recall']} | "
            f"{row['f1']} | {row['support']} |"
            for row in evaluation["per_class"]
        )
    else:
        lines.append("Per-class metrics: not evaluated.")
    lines.extend(["", "## Calibration and error analysis", ""])
    calibration_figures = [
        figure for figure in context["figures"] if "calibration" in figure["name"]
    ]
    error_figures = [figure for figure in context["figures"] if "error" in figure["name"]]
    if calibration_figures:
        lines.extend(
            f"![{figure['caption']}](figures/{figure['name']})" for figure in calibration_figures
        )
    else:
        lines.append("Calibration figure: not evaluated.")
    if error_figures:
        lines.extend(
            f"![{figure['caption']}](figures/{figure['name']})" for figure in error_figures
        )
    else:
        lines.append("Error gallery: not evaluated.")
    lines.extend(["", "## Figures", ""])
    other_figures = [
        figure
        for figure in context["figures"]
        if not any(
            marker in figure["name"]
            for marker in ("training_history", "confusion_matrix", "calibration", "error")
        )
    ]
    if other_figures:
        lines.extend(
            f"![{figure['caption']}](figures/{figure['name']})" for figure in other_figures
        )
    else:
        lines.append("Additional figures: not evaluated.")
    lines.extend(["", "## Runtime", ""])
    if context["runtime_rows"]:
        lines.extend(f"- {row['label']}: {row['value']}" for row in context["runtime_rows"])
    else:
        lines.append("Runtime: not evaluated.")
    if evaluation["latency"]:
        lines.extend(f"- {row['label']}: {row['value']}" for row in evaluation["latency"])
    lines.extend(["", "## Reproduction", "", "```bash"])
    lines.extend(context["reproduction"])
    lines.extend(["```", ""])
    return "\n".join(lines)


def build_report(run_dir: Path, output_path: Path | None = None) -> Path:
    """Render a self-contained HTML report from saved run artifacts."""
    context = _build_context(run_dir)
    environment = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template(_TEMPLATE_NAME)
    html = template.render(**context)
    destination = output_path or (run_dir / "report.html")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination
