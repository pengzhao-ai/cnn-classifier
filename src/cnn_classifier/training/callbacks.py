"""Training callbacks."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from cnn_classifier.config.schema import UnfreezeEvent
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.models.transfer import unfreeze_modules
from cnn_classifier.training.checkpoint import save_checkpoint
from cnn_classifier.training.state import TrainingState


class Callback(Protocol):
    def on_train_start(self, state: TrainingState) -> None: ...

    def on_epoch_end(self, state: TrainingState) -> None: ...

    def on_train_end(self, state: TrainingState) -> None: ...


class _NoOpCallback:
    def on_train_start(self, state: TrainingState) -> None:
        return None

    def on_epoch_end(self, state: TrainingState) -> None:
        return None

    def on_train_end(self, state: TrainingState) -> None:
        return None


def _is_better(value: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "max":
        return value > best + min_delta
    return value < best - min_delta


def _initial_best(mode: str) -> float:
    return -math.inf if mode == "max" else math.inf


class EarlyStopping(_NoOpCallback):
    def __init__(self, monitor: str, mode: str = "max", patience: int = 10, min_delta: float = 0.0):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = _initial_best(mode)
        self.wait = 0

    def on_train_start(self, state: TrainingState) -> None:
        self.best = _initial_best(self.mode)
        self.wait = 0

    def restore_from_history(
        self,
        history: Sequence[dict[str, float]],
        fallback: float,
    ) -> None:
        """Restore best value and patience progress after a resume."""
        best = _initial_best(self.mode)
        seen = 0
        last_improvement = -1
        for row in history:
            if self.monitor not in row:
                continue
            value = float(row[self.monitor])
            seen += 1
            if _is_better(value, best, self.mode, self.min_delta):
                best = value
                last_improvement = seen - 1
        if last_improvement >= 0:
            self.best = best
            self.wait = seen - 1 - last_improvement
        else:
            self.best = fallback
            self.wait = 0

    def on_epoch_end(self, state: TrainingState) -> None:
        if self.monitor not in state.metrics:
            raise ConfigError(f"early_stopping monitor {self.monitor!r} is not a logged metric")
        value = float(state.metrics[self.monitor])
        if _is_better(value, self.best, self.mode, self.min_delta):
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                state.stop_requested = True


class ModelCheckpoint(_NoOpCallback):
    def __init__(self, monitor: str, mode: str = "max", save_last: bool = True):
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.best = _initial_best(mode)

    def on_train_start(self, state: TrainingState) -> None:
        self.best = _initial_best(self.mode)

    def restore_best(self, best: float) -> None:
        """Restore the best monitored value after a resume."""
        self.best = best

    def _save(self, state: TrainingState, name: str) -> None:
        save_checkpoint(
            Path(state.run.directory) / "checkpoints" / name,
            model=state.eval_model or state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            scaler=state.scaler,
            epoch=state.epoch,
            global_step=state.global_step,
            best_metric=state.best_metric,
            best_epoch=state.best_epoch,
            config_hash=state.run.config_hash,
            class_names=state.class_names,
            normalization=state.normalization,
            model_contract=state.model_contract,
        )

    def on_epoch_end(self, state: TrainingState) -> None:
        if self.monitor not in state.metrics:
            raise ConfigError(f"checkpoint monitor {self.monitor!r} is not a logged metric")
        value = float(state.metrics[self.monitor])
        improved = _is_better(value, self.best, self.mode, 0.0)
        if improved:
            self.best = value
            state.best_metric = value
            state.best_epoch = state.epoch
        if self.save_last:
            self._save(state, "last.pt")
        if improved:
            self._save(state, "best.pt")


class GradientMonitor(_NoOpCallback):
    def __init__(self) -> None:
        self.last_report: dict[str, float] = {}

    def on_epoch_end(self, state: TrainingState) -> None:
        report = {}
        for name, parameter in state.model.named_parameters():
            if parameter.grad is not None:
                report[name] = float(parameter.grad.detach().norm().item())
        self.last_report = report
        if report:
            state.metrics["grad_norm"] = float(
                sum(value * value for value in report.values()) ** 0.5
            )


class NaNDetector(_NoOpCallback):
    def __init__(self) -> None:
        self.failed = False

    def on_train_start(self, state: TrainingState) -> None:
        self.failed = False

    def on_epoch_end(self, state: TrainingState) -> None:
        values = list(state.metrics.values())
        gradients = [
            parameter.grad for parameter in state.model.parameters() if parameter.grad is not None
        ]
        nonfinite = any(not math.isfinite(float(value)) for value in values) or any(
            not bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        if nonfinite:
            self.failed = True
            state.stop_requested = True


class EMACallback(_NoOpCallback):
    def __init__(self, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ConfigError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.ema: nn.Module | None = None

    def on_train_start(self, state: TrainingState) -> None:
        self.ema = copy.deepcopy(state.model)
        for parameter in self.ema.parameters():
            parameter.requires_grad = False

    def on_epoch_end(self, state: TrainingState) -> None:
        if self.ema is None:
            self.on_train_start(state)
        assert self.ema is not None
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                self.ema.parameters(), state.model.parameters(), strict=True
            ):
                ema_parameter.mul_(self.decay).add_(parameter, alpha=1.0 - self.decay)
            for ema_buffer, buffer in zip(self.ema.buffers(), state.model.buffers(), strict=True):
                if torch.is_floating_point(ema_buffer):
                    ema_buffer.mul_(self.decay).add_(buffer, alpha=1.0 - self.decay)
                else:
                    ema_buffer.copy_(buffer)
        state.eval_model = self.ema


class UnfreezeCallback(_NoOpCallback):
    def __init__(self, schedule: Sequence[UnfreezeEvent]) -> None:
        self.schedule = tuple(schedule)

    def on_epoch_end(self, state: TrainingState) -> None:
        for event in self.schedule:
            if event.epoch == state.epoch:
                unfreeze_modules(state.model, event.modules)


class TrackerCallback(_NoOpCallback):
    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker

    def on_epoch_end(self, state: TrainingState) -> None:
        self.tracker.log_metrics(state.metrics, state.epoch)

    def on_train_end(self, state: TrainingState) -> None:
        self.tracker.close()
