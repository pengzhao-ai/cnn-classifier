from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cnn_classifier.config.schema import RunConfig
from cnn_classifier.core.errors import RunError
from cnn_classifier.core.types import NormalizationStats
from cnn_classifier.data.adapters import build_dataset
from cnn_classifier.data.loaders import DataLoaderBundle, build_loaders
from cnn_classifier.models import build_model
from cnn_classifier.runs.context import RunContext
from cnn_classifier.training.trainer import Trainer


def make_config(epochs: int) -> RunConfig:
    return RunConfig.model_validate(
        {
            "schema_version": "1.0",
            "experiment": {"name": "trainer-test", "seed": 3},
            "data": {
                "adapter": "synthetic",
                "class_names": ["a", "b"],
                "num_classes": 2,
                "channels": "gray",
                "image_size": [8, 8],
                "adapter_params": {"num_samples": 8, "num_classes": 2, "image_size": [8, 8]},
            },
            "model": {"name": "simple_cnn", "params": {"channels": [4], "dropout": 0.0}},
            "training": {"epochs": epochs, "batch_size": 2},
        }
    )


def make_trainer(tmp_path, epochs: int, exploding_test: bool = False):
    config = make_config(epochs)
    bundle = build_dataset(config.data)
    loaders = build_loaders(bundle, config.training, config.runtime, torch.device("cpu"))
    if exploding_test:
        loaders = DataLoaderBundle(train=loaders.train, val=loaders.val, test=ExplodingLoader())
    model = build_model(
        config.model,
        num_classes=len(bundle.class_names),
        in_channels=bundle.in_channels,
        image_size=bundle.image_size,
    )
    run = RunContext.create(config, tmp_path / "runs")
    trainer = Trainer(
        model,
        loaders,
        config,
        run,
        torch.device("cpu"),
        model_contract=bundle.model_contract,
        class_names=bundle.class_names,
        normalization=bundle.normalization,
    )
    return trainer, config, model, loaders, run


class ExplodingLoader:
    def __iter__(self):
        raise AssertionError("test loader must never be iterated during fit")

    def __len__(self):
        return 1


def test_fit_writes_history_and_never_touches_test_loader(tmp_path):
    trainer, _, _, _, run = make_trainer(tmp_path, epochs=2, exploding_test=True)
    result = trainer.fit()
    assert len(result.history) == 2
    assert result.best_checkpoint.exists()
    assert (run.directory / "summary.json").is_file()
    assert (run.directory / "model_summary.json").is_file()
    assert (run.directory / "data" / "model_contract.json").is_file()
    assert (run.directory / "metrics.jsonl").is_file()
    assert (run.directory / "history.csv").is_file()


def test_resume_continues_from_last_checkpoint(tmp_path):
    trainer, config, _, _, run = make_trainer(tmp_path, epochs=4)
    config.training.epochs = 2
    trainer.fit()
    checkpoint = run.directory / "checkpoints" / "last.pt"
    config.training.epochs = 4
    trainer.resume(checkpoint)
    result = trainer.fit()
    assert result.history[0]["epoch"] == 3
    assert result.history[-1]["epoch"] == 4


def test_resume_preserves_best_checkpoint_and_summary(tmp_path):
    trainer, config, _, _, run = make_trainer(tmp_path, epochs=1)
    trainer._validate = lambda: {"val_loss": 0.5, "val_accuracy": 0.9}
    first = trainer.fit()
    assert first.best_metric == pytest.approx(0.9)
    best_before = (run.directory / "checkpoints" / "best.pt").read_bytes()

    config.training.epochs = 2
    trainer.resume(
        run.directory / "checkpoints" / "last.pt",
        history=[{"epoch": 1, "val_accuracy": 0.9}],
    )
    trainer._validate = lambda: {"val_loss": 1.0, "val_accuracy": 0.1}
    result = trainer.fit()

    assert result.best_metric == pytest.approx(0.9)
    assert (run.directory / "checkpoints" / "best.pt").read_bytes() == best_before
    summary = json.loads((run.directory / "summary.json").read_text())
    assert summary["best_metric"] == pytest.approx(0.9)


def test_resume_rejects_changed_class_map(tmp_path):
    trainer, _, _, _, run = make_trainer(tmp_path, epochs=1)
    trainer.fit()
    checkpoint = run.directory / "checkpoints" / "last.pt"
    trainer.class_names = ("b", "a")

    with pytest.raises(RunError, match="class"):
        trainer.resume(checkpoint)


def test_resume_rejects_changed_normalization(tmp_path):
    trainer, _, _, _, run = make_trainer(tmp_path, epochs=1)
    trainer.fit()
    checkpoint = run.directory / "checkpoints" / "last.pt"
    trainer.normalization = NormalizationStats(mean=(0.1,), std=(0.2,), source="supplied")

    with pytest.raises(RunError, match="normalization"):
        trainer.resume(checkpoint)


def test_history_logs_learning_rate_used_for_the_epoch(tmp_path):
    trainer, _, _, _, _ = make_trainer(tmp_path, epochs=1)
    result = trainer.fit()
    assert result.history[0]["lr"] == pytest.approx(1.0e-3)


def test_summary_contains_device_and_best_metric(tmp_path):
    trainer, _, _, _, run = make_trainer(tmp_path, epochs=1)
    trainer.fit()
    summary = json.loads((run.directory / "summary.json").read_text())
    assert summary["device"] == "cpu"
    assert "best_metric" in summary
    assert summary["config_hash"] == run.config_hash


def test_training_import_does_not_import_plotting_stack():
    code = (
        "import sys; import cnn_classifier.training.trainer; "
        "assert 'matplotlib' not in sys.modules; "
        "assert 'jinja2' not in sys.modules"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
