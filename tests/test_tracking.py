import json

from cnn_classifier.config.schema import RunConfig, TrackingConfig
from cnn_classifier.runs.artifacts import MetricsWriter
from cnn_classifier.runs.context import RunContext
from cnn_classifier.tracking import build_trackers


def test_metrics_writer_writes_jsonl_and_csv(tmp_path, minimal_config_dict):
    config = RunConfig.model_validate(minimal_config_dict)
    run = RunContext.create(config, tmp_path / "runs")
    writer = MetricsWriter(run)
    writer.append({"train_loss": 1.0, "val_loss": 0.9}, step=1)
    writer.append({"train_loss": 0.5, "val_loss": 0.4}, step=2)
    lines = (run.directory / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1
    csv_lines = (run.directory / "history.csv").read_text().strip().splitlines()
    assert csv_lines[0] == "step,train_loss,val_loss"
    assert len(csv_lines) == 3


def test_build_trackers_respects_disabled_backends(tmp_path, minimal_config_dict):
    config = RunConfig.model_validate(minimal_config_dict)
    run = RunContext.create(config, tmp_path / "runs")
    trackers = build_trackers(TrackingConfig(console=False, csv=True, tensorboard=False), run)
    trackers.log_metrics({"val_accuracy": 0.8}, step=1)
    trackers.close()
    assert (run.directory / "metrics.jsonl").is_file()
