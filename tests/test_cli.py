from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from cnn_classifier.cli import app

runner = CliRunner()
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tests" / "synthetic.yaml"


@pytest.fixture
def synthetic_config_path() -> Path:
    return CONFIG


def _run_directory(root: Path) -> Path:
    directories = [path for path in root.iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0]


def test_cli_help_and_doctor():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "init",
        "doctor",
        "inspect-data",
        "debug",
        "train",
        "evaluate",
        "predict",
        "report",
        "compare",
    ):
        assert command in result.stdout
    doctor = runner.invoke(app, ["doctor"])
    assert doctor.exit_code == 0
    assert "cnn-classifier" in doctor.stdout
    assert "device" in doctor.stdout.lower()


def test_train_dry_run_prints_contract(tmp_path):
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        ["train", "-c", str(CONFIG), "--dry-run", "--runs-dir", str(runs)],
    )
    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "model input" in output
    assert "class" in output
    assert "parameter" in output
    assert "planned epochs" in output
    assert not runs.exists()


def test_inspect_data_reports_contract():
    result = runner.invoke(app, ["inspect-data", "-c", str(CONFIG)])
    assert result.exit_code == 0, result.output
    assert "Class counts" in result.stdout
    assert "Model input" in result.stdout
    assert "duplicates" in result.stdout.lower()


def test_train_evaluate_predict_and_report(tmp_path):
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "train",
            "-c",
            str(CONFIG),
            "--set",
            "training.epochs=1",
            "--runs-dir",
            str(runs),
        ],
    )
    assert result.exit_code == 0, result.output
    run = _run_directory(runs)
    for relative in (
        "config.resolved.yaml",
        "data/model_contract.json",
        "data/fingerprint.json",
        "data/splits.json",
        "model_summary.json",
        "summary.json",
        "evaluation.json",
        "report.html",
        "report.md",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "predictions/test_predictions.npz",
    ):
        assert (run / relative).is_file(), relative
    fingerprint = json.loads((run / "data" / "fingerprint.json").read_text())
    assert fingerprint["fingerprint"]
    summary = json.loads((run / "summary.json").read_text())
    assert summary["test_accuracy"] is not None
    assert summary["params"] > 0

    evaluated = runner.invoke(
        app,
        [
            "evaluate",
            "--checkpoint",
            str(run / "checkpoints" / "best.pt"),
            "--split",
            "test",
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    evaluation = json.loads((run / "evaluation.json").read_text())
    assert evaluation["split"] == "test"

    reported = runner.invoke(app, ["report", "--run", str(run)])
    assert reported.exit_code == 0, reported.output
    assert (run / "report.html").is_file()
    assert (run / "report.md").is_file()

    image_path = tmp_path / "sample.png"
    image = np.random.default_rng(0).integers(0, 255, (16, 16), dtype=np.uint8)
    Image.fromarray(image).save(image_path)
    output_path = tmp_path / "predictions.csv"
    predicted = runner.invoke(
        app,
        [
            "predict",
            "-c",
            str(CONFIG),
            "--set",
            "training.epochs=1",
            "--checkpoint",
            str(run / "checkpoints" / "best.pt"),
            "--input",
            str(image_path),
            "--output",
            str(output_path),
        ],
    )
    assert predicted.exit_code == 0, predicted.output
    assert output_path.is_file()
    assert "predicted" in output_path.read_text().lower()

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"path\n{image_path}\n", encoding="utf-8")
    manifest_output = tmp_path / "manifest_predictions.csv"
    predicted_manifest = runner.invoke(
        app,
        [
            "predict",
            "-c",
            str(CONFIG),
            "--set",
            "training.epochs=1",
            "--checkpoint",
            str(run / "checkpoints" / "best.pt"),
            "--input",
            str(manifest),
            "--output",
            str(manifest_output),
        ],
    )
    assert predicted_manifest.exit_code == 0, predicted_manifest.output
    assert manifest_output.is_file()


def test_compare_requires_matching_fingerprint(tmp_path):
    runs = tmp_path / "runs"
    trained = runner.invoke(
        app,
        [
            "train",
            "-c",
            str(CONFIG),
            "--set",
            "training.epochs=1",
            "--runs-dir",
            str(runs),
        ],
    )
    assert trained.exit_code == 0, trained.output
    run = _run_directory(runs)
    copy = tmp_path / "copy"
    shutil.copytree(run, copy)
    output = tmp_path / "comparison"
    compared = runner.invoke(
        app,
        ["compare", str(run), str(copy), "--out", str(output)],
    )
    assert compared.exit_code == 0, compared.output
    assert (output / "comparison.md").is_file()
    assert (output / "comparison.json").is_file()
    assert (output / "comparison.png").is_file()


def test_train_resume_extends_history(tmp_path):
    runs = tmp_path / "runs"
    first = runner.invoke(
        app,
        [
            "train",
            "-c",
            str(CONFIG),
            "--set",
            "training.epochs=1",
            "--runs-dir",
            str(runs),
        ],
    )
    assert first.exit_code == 0, first.output
    run = _run_directory(runs)
    resumed = runner.invoke(
        app,
        [
            "train",
            "-c",
            str(CONFIG),
            "--set",
            "training.epochs=2",
            "--resume",
            str(run / "checkpoints" / "last.pt"),
            "--allow-config-change",
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    summary = json.loads((run / "summary.json").read_text())
    assert summary["epochs_run"] >= 2
    history = json.loads((run / "summary.json").read_text())["history"]
    assert len(history) >= 2


def test_init_creates_project(tmp_path):
    project = tmp_path / "project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0, result.output
    assert (project / "config.yaml").is_file()
    assert (project / "README.md").is_file()
    assert (project / "data").is_dir()
