from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from cnn_classifier.viz.report import build_markdown_report, build_report

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def write_minimal_run(root: Path) -> Path:
    run = root / "run"
    (run / "data").mkdir(parents=True)
    (run / "predictions").mkdir()
    (run / "figures").mkdir()
    (run / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "demo", "seed": 7},
                "data": {"adapter": "synthetic", "image_size": [8, 8]},
                "model": {"name": "simple_cnn"},
                "training": {"epochs": 2, "batch_size": 2},
                "runtime": {"device": "cpu"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (run / "environment.json").write_text(
        json.dumps({"python": "3.13", "torch": "2.15", "device": {"selected": "cpu"}}),
        encoding="utf-8",
    )
    (run / "data" / "model_contract.json").write_text(
        json.dumps(
            {
                "input_shape": [1, 8, 8],
                "dtype": "float32",
                "value_range": [0.0, 1.0],
                "normalization": {"mean": [0.5], "std": [0.2], "source": "computed"},
                "class_names": ["a", "b"],
                "label_map": {"a": 0, "b": 1},
                "channel_policy": "gray",
            }
        ),
        encoding="utf-8",
    )
    (run / "data" / "fingerprint.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "split_mode": "random_stratified",
                "splits": {
                    "train": {"count": 6, "fingerprint": "abc123"},
                    "val": {"count": 2, "fingerprint": "def456"},
                    "test": {"count": 2, "fingerprint": "ghi789"},
                },
                "group_overlap": False,
                "duplicate_overlap": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "model_summary.json").write_text(
        json.dumps(
            {
                "input_shape": [1, 8, 8],
                "output_shape": [1, 2],
                "total_params": 1234,
                "trainable_params": 1200,
                "modules": [{"name": "features.0", "type": "Conv2d", "params": 100}],
            }
        ),
        encoding="utf-8",
    )
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "epoch",
                "train_loss",
                "val_loss",
                "train_accuracy",
                "val_accuracy",
                "lr",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step": 1,
                "epoch": 1.0,
                "train_loss": 0.8,
                "val_loss": 0.7,
                "train_accuracy": 0.7,
                "val_accuracy": 0.75,
                "lr": 0.001,
            }
        )
        writer.writerow(
            {
                "step": 2,
                "epoch": 2.0,
                "train_loss": 0.4,
                "val_loss": 0.5,
                "train_accuracy": 0.85,
                "val_accuracy": 0.8,
                "lr": 0.0005,
            }
        )
    (run / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "val_accuracy": 0.75}) + "\n",
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "best_metric": 0.8,
                "best_epoch": 2,
                "epochs_run": 2,
                "device": "cpu",
                "config_hash": "deadbeef",
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    (run / "evaluation.json").write_text(
        json.dumps(
            {
                "split": "test",
                "metrics": {
                    "accuracy": 0.9,
                    "top_k_accuracy": 1.0,
                    "macro_f1": 0.89,
                    "nll": 0.3,
                    "ece": 0.05,
                    "support": 10,
                    "confusion_matrix": [[4, 1], [0, 5]],
                    "per_class": {
                        "a": {"precision": 1.0, "recall": 0.8, "f1": 0.89, "support": 5},
                        "b": {"precision": 0.83, "recall": 1.0, "f1": 0.91, "support": 5},
                    },
                },
                "confidence_intervals": {
                    "accuracy": {"low": 0.8, "high": 0.98, "confidence": 0.95}
                },
                "latency": {"mean_ms": 1.2, "p50_ms": 1.1, "p95_ms": 1.5, "throughput": 833.0},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        run / "predictions" / "test.npz",
        probabilities=np.array([[0.9, 0.1], [0.2, 0.8]]),
        y_true=np.array([0, 1]),
        y_pred=np.array([0, 1]),
        indices=np.array([0, 1]),
    )
    (run / "figures" / "training_history.png").write_bytes(PNG_BYTES)
    return run


def test_report_is_self_contained_and_contains_sections(tmp_path):
    run = write_minimal_run(tmp_path)
    path = build_report(run)
    html = path.read_text(encoding="utf-8")
    assert "Experiment" in html
    assert "Metrics" in html
    assert "Configuration" in html
    assert "data:image/" in html
    assert "<script src=" not in html
    assert '<link rel="stylesheet" href=' not in html


def test_report_shows_metrics_and_missing_sections(tmp_path):
    run = write_minimal_run(tmp_path)
    html = build_report(run, tmp_path / "custom.html").read_text(encoding="utf-8")
    assert "0.900" in html
    assert "1,234" in html
    assert "not evaluated" in html
    assert not (run / "figures" / "confusion_matrix.png").exists()


def test_markdown_report_has_same_numbers(tmp_path):
    run = write_minimal_run(tmp_path)
    markdown = build_markdown_report(run)
    assert "# Experiment" in markdown
    assert "0.900" in markdown
    assert "1,234" in markdown
    assert "not evaluated" in markdown


def test_report_handles_run_without_optional_artifacts(tmp_path):
    run = tmp_path / "bare"
    run.mkdir()
    (run / "config.resolved.yaml").write_text(
        yaml.safe_dump({"experiment": {"name": "bare"}}),
        encoding="utf-8",
    )
    report = build_report(run)
    markdown = build_markdown_report(run)
    assert report.exists()
    assert "not evaluated" in report.read_text(encoding="utf-8")
    assert "not evaluated" in markdown
