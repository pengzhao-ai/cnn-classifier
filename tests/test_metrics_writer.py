from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from cnn_classifier.runs.artifacts import MetricsWriter


def test_metrics_writer_extends_history_header_for_new_keys(tmp_path):
    writer = MetricsWriter(SimpleNamespace(directory=tmp_path))
    writer.append({"epoch": 1.0, "train_loss": 1.0}, step=1)
    writer.append({"epoch": 2.0, "train_loss": 0.9, "grad_norm": 2.5}, step=2)

    with (tmp_path / "history.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames

    assert header == ["step", "epoch", "train_loss", "grad_norm"]
    assert rows[0]["grad_norm"] == ""
    assert rows[1]["grad_norm"] == "2.5"

    metrics = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "grad_norm" not in metrics[0]
    assert metrics[1]["grad_norm"] == 2.5
