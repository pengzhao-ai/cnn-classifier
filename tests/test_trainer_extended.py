from __future__ import annotations

import csv
import json
import math

import pytest
import torch

from cnn_classifier.config.schema import RunConfig
from cnn_classifier.data.adapters import build_dataset
from cnn_classifier.data.loaders import build_loaders
from cnn_classifier.models import build_model
from cnn_classifier.runs.context import RunContext
from cnn_classifier.training.callbacks import _NoOpCallback
from cnn_classifier.training.trainer import Trainer


def _trainer(tmp_path, training: dict, *, transfer: dict | None = None):
    payload = {
        "schema_version": "1.0",
        "experiment": {"name": "trainer-options", "seed": 5},
        "data": {
            "adapter": "synthetic",
            "class_names": ["a", "b"],
            "num_classes": 2,
            "channels": "gray",
            "image_size": [8, 8],
            "adapter_params": {"num_samples": 8, "num_classes": 2, "image_size": [8, 8]},
        },
        "model": {
            "name": "simple_cnn",
            "params": {"channels": [4], "dropout": 0.0},
            "transfer": transfer or {},
        },
        "training": {"epochs": 1, "batch_size": 2, **training},
        "evaluation": {"bootstrap": {"enabled": False}, "latency": {"enabled": False}},
        "tracking": {"console": False, "csv": True, "tensorboard": False},
    }
    config = RunConfig.model_validate(payload)
    bundle = build_dataset(config.data)
    loaders = build_loaders(bundle, config.training, config.runtime, torch.device("cpu"))
    model = build_model(
        config.model,
        num_classes=len(bundle.class_names),
        in_channels=bundle.in_channels,
        image_size=bundle.image_size,
    )
    run = RunContext.create(config, tmp_path / "runs")
    trainer = Trainer(model, loaders, config, run, torch.device("cpu"), model_contract=None)
    return trainer, run


def test_trainer_supports_channels_last_and_mixup_and_clipping(tmp_path):
    trainer, run = _trainer(
        tmp_path,
        {
            "mixup_alpha": 0.2,
            "cutmix_alpha": 0.2,
            "grad_clip_norm": 1.0,
            "scheduler": {"name": "cosine", "step_unit": "batch"},
        },
    )
    trainer.model = trainer.model.to(memory_format=torch.channels_last)
    result = trainer.fit()
    assert result.history
    assert (run.directory / "data" / "model_contract.json").is_file()


def test_trainer_supports_plateau_scheduler_and_ema(tmp_path):
    trainer, _ = _trainer(
        tmp_path,
        {
            "scheduler": {"name": "plateau"},
            "ema": {"enabled": True, "decay": 0.5},
        },
    )
    result = trainer.fit()
    assert result.history
    assert trainer.state.eval_model is not None


def test_trainer_supports_param_groups(tmp_path):
    trainer, _ = _trainer(
        tmp_path,
        {},
        transfer={"param_groups": [{"pattern": "^features\\.0", "lr_multiplier": 2.0}]},
    )
    assert len(trainer.optimizer.param_groups) >= 2
    trainer.fit()


def test_trainer_early_stopping_breaks_loop(tmp_path):
    trainer, _ = _trainer(
        tmp_path,
        {
            "epochs": 4,
            "early_stopping": {
                "enabled": True,
                "monitor": "val_loss",
                "mode": "min",
                "patience": 0,
            },
        },
    )
    losses = iter([0.5, 0.6, 0.7, 0.8])
    trainer._validate = lambda: {"val_loss": next(losses), "val_accuracy": 0.5}
    result = trainer.fit()
    assert len(result.history) < 4


def test_trainer_records_grad_norm_in_artifacts(tmp_path):
    trainer, run = _trainer(tmp_path, {})
    result = trainer.fit()

    assert result.history
    grad_norm = result.history[0]["grad_norm"]
    assert math.isfinite(grad_norm)
    assert grad_norm > 0.0

    summary = json.loads((run.directory / "summary.json").read_text(encoding="utf-8"))
    assert "grad_norm" in summary["history"][0]

    with (run.directory / "history.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert "grad_norm" in header

    metrics = [
        json.loads(line)
        for line in (run.directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "grad_norm" in metrics[0]


def test_trainer_stops_before_step_when_batch_hook_requests_stop(tmp_path):
    class StopBeforeStep(_NoOpCallback):
        def __init__(self):
            self.saw_gradients = False

        def on_batch_end(self, state):
            self.saw_gradients = any(
                parameter.grad is not None for parameter in state.model.parameters()
            )
            state.stop_requested = True

    trainer, _ = _trainer(tmp_path, {})
    callback = StopBeforeStep()
    trainer.callbacks = [callback]
    trainer._callbacks_are_default = False
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    trainer.fit()

    assert callback.saw_gradients is True
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_trainer_tolerates_legacy_callback_without_batch_hook(tmp_path):
    class LegacyCallback:
        def on_train_start(self, state):
            return None

        def on_epoch_end(self, state):
            return None

        def on_train_end(self, state):
            return None

    trainer, _ = _trainer(tmp_path, {})
    trainer.callbacks = [LegacyCallback()]
    trainer._callbacks_are_default = False

    result = trainer.fit()

    assert result.history


@pytest.mark.parametrize("grad_clip_norm", [None, 1.0])
def test_trainer_unscales_gradients_once_per_window(tmp_path, grad_clip_norm):
    class FakeScaler:
        def __init__(self):
            self.unscale_calls = 0
            self.step_calls = 0
            self.update_calls = 0

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            self.unscale_calls += 1

        def step(self, optimizer):
            self.step_calls += 1
            optimizer.step()

        def update(self):
            self.update_calls += 1

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            return None

    training = {} if grad_clip_norm is None else {"grad_clip_norm": grad_clip_norm}
    trainer, _ = _trainer(tmp_path, training)
    expected_windows = len(trainer.loaders.train)
    scaler = FakeScaler()
    trainer.scaler = scaler
    trainer.state.scaler = scaler

    trainer.fit()

    assert scaler.unscale_calls == expected_windows
    assert scaler.step_calls == expected_windows
    assert scaler.update_calls == expected_windows
