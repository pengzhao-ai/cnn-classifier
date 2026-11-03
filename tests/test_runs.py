import json
from pathlib import Path

from cnn_classifier.config.schema import RunConfig
from cnn_classifier.runs.context import RunContext


def test_run_context_creates_artifacts(tmp_path, minimal_config_dict):
    config = RunConfig.model_validate(minimal_config_dict)
    run = RunContext.create(config, tmp_path / "runs")
    assert run.directory.is_dir()
    assert run.config_hash
    assert (run.directory / "config.resolved.yaml").is_file()
    assert (run.directory / "environment.json").is_file()
    assert (run.directory / "git.json").is_file()
    assert (run.directory / "status.json").is_file()
    for name in ("data", "checkpoints", "predictions", "figures"):
        assert (run.directory / name).is_dir()
    status = json.loads((run.directory / "status.json").read_text())
    assert status["status"] == "running"


def test_run_context_writes_json_atomically(tmp_path, minimal_config_dict):
    config = RunConfig.model_validate(minimal_config_dict)
    run = RunContext.create(config, tmp_path / "runs")
    path = run.write_json("summary.json", {"accuracy": 0.5})
    assert json.loads(Path(path).read_text()) == {"accuracy": 0.5}
    run.set_status("completed")
    assert json.loads((run.directory / "status.json").read_text())["status"] == "completed"
