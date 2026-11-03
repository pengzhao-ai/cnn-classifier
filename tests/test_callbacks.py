from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from cnn_classifier.config.schema import RunConfig, UnfreezeEvent
from cnn_classifier.core.types import NormalizationStats
from cnn_classifier.runs.context import RunContext
from cnn_classifier.training.callbacks import (
    EarlyStopping,
    EMACallback,
    GradientMonitor,
    ModelCheckpoint,
    NaNDetector,
    TrackerCallback,
    UnfreezeCallback,
)
from cnn_classifier.training.state import TrainingState


def make_state(tmp_path, metrics=None, epoch=0):
    config = RunConfig.model_validate(
        {
            "schema_version": "1.0",
            "experiment": {"name": "test"},
            "data": {
                "adapter": "synthetic",
                "class_names": ["a", "b"],
                "num_classes": 2,
                "adapter_params": {"num_samples": 4, "num_classes": 2, "image_size": [8, 8]},
            },
            "model": {"name": "simple_cnn", "params": {"channels": [4]}},
            "training": {"epochs": 2, "batch_size": 2},
        }
    )
    run = RunContext.create(config, tmp_path / "runs")
    model = nn.Linear(2, 2)
    return TrainingState(
        model=model,
        eval_model=None,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
        scaler=None,
        epoch=epoch,
        global_step=0,
        history=[],
        metrics=dict(metrics or {}),
        best_metric=0.0,
        best_epoch=0,
        stop_requested=False,
        run=run,
        config=config,
        device=torch.device("cpu"),
        class_names=("a", "b"),
        normalization=NormalizationStats((0.0,), (1.0,), "supplied"),
        model_contract={"input_shape": [1, 8, 8]},
    )


def test_early_stopping_requests_stop_after_patience(tmp_path):
    state = make_state(tmp_path, metrics={"val_accuracy": 0.5})
    callback = EarlyStopping("val_accuracy", mode="max", patience=1, min_delta=0.0)
    callback.on_train_start(state)
    callback.on_epoch_end(state)
    state.metrics = {"val_accuracy": 0.4}
    callback.on_epoch_end(state)
    assert state.stop_requested


def test_model_checkpoint_writes_last_and_best(tmp_path):
    state = make_state(tmp_path, metrics={"val_accuracy": 0.5})
    callback = ModelCheckpoint("val_accuracy", mode="max", save_last=True)
    callback.on_train_start(state)
    callback.on_epoch_end(state)
    assert (state.run.directory / "checkpoints" / "last.pt").is_file()
    assert (state.run.directory / "checkpoints" / "best.pt").is_file()


def test_ema_callback_sets_eval_model(tmp_path):
    state = make_state(tmp_path)
    callback = EMACallback(decay=0.9)
    callback.on_train_start(state)
    initial = state.model.weight.detach().clone()
    state.model.weight.data.fill_(2.0)
    callback.on_epoch_end(state)
    assert state.eval_model is not None
    expected = 0.9 * initial + 0.1 * 2.0
    assert torch.allclose(state.eval_model.weight, expected)


def test_nan_detector_stops_on_nonfinite_metric(tmp_path):
    state = make_state(tmp_path, metrics={"val_loss": math.nan})
    NaNDetector().on_epoch_end(state)
    assert state.stop_requested


def test_unfreeze_callback_applies_schedule(tmp_path):
    state = make_state(tmp_path, epoch=2)
    state.model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    for parameter in state.model.parameters():
        parameter.requires_grad = False
    callback = UnfreezeCallback([UnfreezeEvent(epoch=2, modules=("0",))])
    callback.on_epoch_end(state)
    assert state.model[0].weight.requires_grad
    assert not state.model[1].weight.requires_grad


def test_tracker_callback_logs_metrics(tmp_path):
    state = make_state(tmp_path, metrics={"val_accuracy": 0.7})

    class FakeTracker:
        def __init__(self):
            self.seen = []

        def log_metrics(self, metrics, step):
            self.seen.append((metrics, step))

        def close(self):
            return None

    tracker = FakeTracker()
    TrackerCallback(tracker).on_epoch_end(state)
    assert tracker.seen[0][0]["val_accuracy"] == pytest.approx(0.7)


def test_gradient_monitor_reports_layer_norms(tmp_path):
    state = make_state(tmp_path)
    loss = state.model(torch.ones(1, 2)).sum()
    loss.backward()
    monitor = GradientMonitor()
    monitor.on_epoch_end(state)
    assert "weight" in monitor.last_report
    assert monitor.last_report["weight"] > 0


def test_early_stopping_restores_best_and_wait_from_history():
    stopping = EarlyStopping("val_accuracy", mode="max", patience=2)
    stopping.restore_from_history(
        [
            {"val_accuracy": 0.5},
            {"val_accuracy": 0.6},
            {"val_accuracy": 0.55},
        ],
        fallback=0.6,
    )
    assert stopping.best == pytest.approx(0.6)
    assert stopping.wait == 1
