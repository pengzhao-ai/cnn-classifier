"""Resumable training loop with best-checkpoint selection and test isolation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from cnn_classifier.config.schema import RunConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.core.seeding import make_generator
from cnn_classifier.data.loaders import DataLoaderBundle
from cnn_classifier.models.summary import model_summary
from cnn_classifier.models.transfer import build_param_groups
from cnn_classifier.runs.context import RunContext
from cnn_classifier.tracking import build_trackers
from cnn_classifier.training.amp import AmpPolicy, autocast_context, resolve_amp
from cnn_classifier.training.callbacks import (
    Callback,
    EarlyStopping,
    EMACallback,
    GradientMonitor,
    ModelCheckpoint,
    NaNDetector,
    TrackerCallback,
    UnfreezeCallback,
)
from cnn_classifier.training.checkpoint import load_checkpoint, validate_checkpoint_identity
from cnn_classifier.training.losses import build_loss, mixup_cutmix_batch
from cnn_classifier.training.optim import build_optimizer
from cnn_classifier.training.schedulers import build_scheduler
from cnn_classifier.training.state import TrainingState


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    best_checkpoint: Path
    history: list[dict[str, float]]
    best_metric: float
    best_epoch: int


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return cast(Any, asdict)(value)
    return value


class Trainer:
    """Train, validate, checkpoint, and resume one classification model."""

    def __init__(
        self,
        model: nn.Module,
        loaders: DataLoaderBundle,
        config: RunConfig,
        run: RunContext,
        device: torch.device,
        callbacks: Sequence[Callback] | None = None,
        *,
        model_contract: Any = None,
        class_names: tuple[str, ...] = (),
        normalization: Any = None,
    ) -> None:
        self.model = model.to(device)
        self.loaders = loaders
        self.config = config
        self.run = run
        self.device = device
        self.model_contract = model_contract
        self.class_names = class_names or (
            config.data.class_names if isinstance(config.data.class_names, tuple) else ()
        )
        self.normalization = normalization
        if config.runtime.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if config.runtime.compile:
            self.model = cast(nn.Module, torch.compile(self.model))
        self.amp_policy: AmpPolicy = resolve_amp(device, config.training.amp)
        param_groups = None
        if config.model.transfer.param_groups:
            param_groups = build_param_groups(
                self.model,
                config.model.transfer,
                base_lr=config.training.optimizer.lr,
                base_weight_decay=config.training.optimizer.weight_decay,
            )
        self.optimizer = build_optimizer(self.model, config.training, param_groups)
        self.scheduler = build_scheduler(
            self.optimizer, config.training.scheduler, config.training.epochs
        )
        self.scaler = self.amp_policy.scaler
        self.loss_fn = build_loss(config.training.loss)
        self._generator = make_generator(config.experiment.seed)
        self._start_epoch = 0
        self.state = TrainingState(
            model=self.model,
            eval_model=None,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=0,
            global_step=0,
            history=[],
            metrics={},
            best_metric=0.0,
            best_epoch=0,
            stop_requested=False,
            run=run,
            config=config,
            device=device,
            class_names=self.class_names,
            normalization=self.normalization,
            model_contract=self.model_contract,
        )
        self.callbacks = list(callbacks) if callbacks is not None else self._default_callbacks()
        self._callbacks_are_default = callbacks is None

    def _default_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = [
            GradientMonitor(),
            NaNDetector(),
            UnfreezeCallback(self.config.model.transfer.unfreeze_schedule),
            ModelCheckpoint(
                self.config.training.checkpoint.monitor,
                mode=self.config.training.checkpoint.mode,
                save_last=self.config.training.checkpoint.save_last,
            ),
            TrackerCallback(build_trackers(self.config.tracking, self.run)),
        ]
        if self.config.training.early_stopping.enabled:
            callbacks.append(
                EarlyStopping(
                    self.config.training.early_stopping.monitor,
                    mode=self.config.training.early_stopping.mode,
                    patience=self.config.training.early_stopping.patience,
                    min_delta=self.config.training.early_stopping.min_delta,
                )
            )
        if self.config.training.ema.enabled:
            callbacks.append(EMACallback(self.config.training.ema.decay))
        return callbacks

    def _input_shape(self) -> tuple[int, int, int]:
        first = self.loaders.train.dataset[0][0]
        return (int(first.shape[0]), int(first.shape[1]), int(first.shape[2]))

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        self.optimizer.zero_grad(set_to_none=True)
        batches = len(self.loaders.train)
        for batch_index, batch in enumerate(self.loaders.train):
            images, labels, _ = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            mixed = None
            if self.config.training.mixup_alpha > 0 or self.config.training.cutmix_alpha > 0:
                mixed = mixup_cutmix_batch(
                    images,
                    labels,
                    self.config.training.mixup_alpha,
                    self.config.training.cutmix_alpha,
                    self._generator,
                )
            with autocast_context(self.amp_policy):
                outputs = self.model(mixed.images if mixed is not None else images)
                if mixed is None:
                    loss = self.loss_fn(outputs, labels)
                else:
                    loss = mixed.lam * self.loss_fn(outputs, mixed.labels_a) + (
                        1.0 - mixed.lam
                    ) * self.loss_fn(outputs, mixed.labels_b)
            scaled_loss = loss / self.config.training.grad_accum_steps
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((outputs.argmax(1) == labels).sum().item())
            total += labels.size(0)
            self.state.global_step += 1
            should_step = (
                batch_index + 1
            ) % self.config.training.grad_accum_steps == 0 or batch_index + 1 == batches
            if should_step:
                if self.config.training.grad_clip_norm is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip_norm
                    )
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.config.training.scheduler.step_unit == "batch":
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        raise ConfigError("plateau scheduler requires epoch stepping")
                    self.scheduler.step()
        if total == 0:
            raise ConfigError("training loader produced no samples")
        return {
            "train_loss": total_loss / total,
            "train_accuracy": total_correct / total,
        }

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        model = self.state.eval_model or self.model
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch in self.loaders.val:
            images, labels, _ = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            with autocast_context(self.amp_policy):
                outputs = model(images)
                loss = self.loss_fn(outputs, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((outputs.argmax(1) == labels).sum().item())
            total += labels.size(0)
        if total == 0:
            raise ConfigError("validation loader produced no samples")
        return {"val_loss": total_loss / total, "val_accuracy": total_correct / total}

    def validate(self) -> dict[str, float]:
        """Evaluate on the validation split without touching test."""
        return self._validate()

    def resume(
        self,
        checkpoint_path: Path,
        *,
        allow_config_change: bool = False,
        history: Sequence[dict[str, float]] | None = None,
    ) -> None:
        """Restore model, optimizer, scheduler, AMP, and RNG state."""
        payload = load_checkpoint(
            checkpoint_path,
            map_location=self.device,
            expected_config_hash=self.run.config_hash,
            allow_config_change=allow_config_change,
        )
        validate_checkpoint_identity(
            payload,
            class_names=self.class_names,
            normalization=self.normalization,
            model_contract=self.model_contract,
        )
        self.model.load_state_dict(payload["model"])
        if payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler") is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        if self.scaler is not None and payload.get("scaler") is not None:
            self.scaler.load_state_dict(payload["scaler"])
        self._start_epoch = int(payload["epoch"])
        self.state.epoch = self._start_epoch
        self.state.global_step = int(payload["global_step"])
        self.state.best_metric = float(payload["best_metric"])
        self.state.best_epoch = int(payload["best_epoch"])
        self.state.history = [dict(row) for row in (history or [])]
        self.state.stop_requested = False
        if self._callbacks_are_default:
            self.callbacks = self._default_callbacks()
        rng = payload.get("rng", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])

    def _restore_callback_state(self) -> None:
        """Reapply resumed best/patience state after callbacks initialize."""
        if self._start_epoch <= 0:
            return
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback.restore_best(self.state.best_metric)
            elif isinstance(callback, EarlyStopping):
                callback.restore_from_history(self.state.history, self.state.best_metric)

    def _write_start_artifacts(self) -> None:
        shape = self._input_shape()
        self.run.write_json("model_summary.json", model_summary(self.model, shape))
        contract = _jsonable(self.model_contract)
        if contract is None:
            contract = {
                "input_shape": list(shape),
                "dtype": "float32",
                "class_names": list(self.class_names),
                "normalization": _jsonable(self.normalization),
            }
        self.run.write_json("data/model_contract.json", contract)

    def fit(self) -> TrainingResult:
        """Run the configured number of epochs on train/val only."""
        self._write_start_artifacts()
        for callback in self.callbacks:
            callback.on_train_start(self.state)
        self._restore_callback_state()
        for epoch in range(self._start_epoch + 1, self.config.training.epochs + 1):
            self.state.epoch = epoch
            metrics = self._train_epoch()
            metrics.update(self._validate())
            metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
            if self.config.training.scheduler.step_unit == "epoch":
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics["val_loss"])
                else:
                    self.scheduler.step()
            metrics["epoch"] = float(epoch)
            self.state.metrics = metrics
            self.state.history.append(dict(metrics))
            for callback in self.callbacks:
                callback.on_epoch_end(self.state)
            if self.state.stop_requested:
                break
        for callback in self.callbacks:
            callback.on_train_end(self.state)
        best_path = self.run.directory / "checkpoints" / "best.pt"
        if not best_path.exists():
            best_path = self.run.directory / "checkpoints" / "last.pt"
        summary = {
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "epochs_run": len(self.state.history),
            "device": str(self.device),
            "config_hash": self.run.config_hash,
            "history": self.state.history,
        }
        self.run.write_json("summary.json", summary)
        self.run.close("completed")
        return TrainingResult(
            run_dir=self.run.directory,
            best_checkpoint=best_path,
            history=self.state.history,
            best_metric=self.state.best_metric,
            best_epoch=self.state.best_epoch,
        )
