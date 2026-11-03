import json
from pathlib import Path

import pytest

from cnn_classifier.core.errors import RunError
from cnn_classifier.evaluation.compare import compare_runs


def write_run(path: Path, fingerprint: str, accuracy: float) -> None:
    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "data" / "fingerprint.json").write_text(
        json.dumps({"fingerprint": fingerprint}), encoding="utf-8"
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "test_accuracy": accuracy,
                "macro_f1": accuracy - 0.01,
                "accuracy_ci": [accuracy - 0.02, accuracy + 0.02],
                "params": 1000,
                "trainable_params": 1000,
                "train_seconds": 10.0,
                "latency_ms": 1.0,
                "peak_memory_mb": 100.0,
            }
        ),
        encoding="utf-8",
    )


def test_compare_refuses_mismatched_fingerprints(tmp_path):
    write_run(tmp_path / "a", fingerprint="one", accuracy=0.8)
    write_run(tmp_path / "b", fingerprint="two", accuracy=0.9)
    with pytest.raises(RunError, match="fingerprint"):
        compare_runs([tmp_path / "a", tmp_path / "b"], tmp_path / "out")


def test_compare_writes_markdown_and_rows(tmp_path):
    write_run(tmp_path / "a", fingerprint="same", accuracy=0.8)
    write_run(tmp_path / "b", fingerprint="same", accuracy=0.9)
    report = compare_runs([tmp_path / "a", tmp_path / "b"], tmp_path / "out")
    assert len(report.rows) == 2
    assert "Accuracy" in report.markdown
    assert (tmp_path / "out" / "comparison.md").is_file()
    assert (tmp_path / "out" / "comparison.json").is_file()
