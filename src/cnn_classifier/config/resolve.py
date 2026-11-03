"""Path resolution and semantic validation for configuration."""

from __future__ import annotations

from pathlib import Path

from cnn_classifier.config.schema import DataConfig, RunConfig
from cnn_classifier.core.errors import ConfigError


def _absolute(path: Path | None, base_dir: Path) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (base_dir / path).resolve()


def resolve_config(config: RunConfig, base_dir: Path) -> RunConfig:
    """Resolve data paths relative to the experiment file's directory."""
    data = config.data.model_copy(
        update={
            "root": _absolute(config.data.root, base_dir),
            "manifest": _absolute(config.data.manifest, base_dir),
            "image_root": _absolute(config.data.image_root, base_dir),
        }
    )
    return config.model_copy(update={"data": data})


def _check_channel_lengths(data: DataConfig) -> None:
    norm = data.normalization
    if norm.mode != "supplied":
        return
    if norm.mean is None or norm.std is None:
        raise ConfigError("supplied normalization requires mean and std")
    if len(norm.mean) != len(norm.std):
        raise ConfigError("normalization mean and std must have the same length")
    expected = {"rgb": 3, "gray": 1}.get(data.channels)
    if expected is not None and len(norm.mean) != expected:
        raise ConfigError(
            f"normalization has {len(norm.mean)} channels but data.channels={data.channels!r}"
        )


def validate_config_semantics(config: RunConfig) -> None:
    """Reject cross-field combinations that would fail midway through a run."""
    training = config.training
    data = config.data
    scheduler = training.scheduler

    if training.epochs <= 0:
        raise ConfigError("training.epochs must be positive")
    if training.batch_size <= 0:
        raise ConfigError("training.batch_size must be positive")
    if training.grad_accum_steps <= 0:
        raise ConfigError("training.grad_accum_steps must be positive")
    if scheduler.warmup_epochs >= training.epochs:
        raise ConfigError("scheduler.warmup_epochs must be smaller than training.epochs")
    if not 0.0 <= scheduler.min_lr_ratio < 1.0:
        raise ConfigError("scheduler.min_lr_ratio must be in [0, 1)")
    if training.optimizer.lr <= 0:
        raise ConfigError("optimizer.lr must be positive")
    if training.optimizer.weight_decay < 0:
        raise ConfigError("optimizer.weight_decay must be non-negative")
    if not 0.0 <= training.loss.label_smoothing < 1.0:
        raise ConfigError("loss.label_smoothing must be in [0, 1)")
    if not 0.0 <= training.mixup_alpha <= 1.0:
        raise ConfigError("training.mixup_alpha must be in [0, 1]")
    if not 0.0 <= training.cutmix_alpha <= 1.0:
        raise ConfigError("training.cutmix_alpha must be in [0, 1]")
    if training.ema.enabled and not 0.0 < training.ema.decay < 1.0:
        raise ConfigError("training.ema.decay must be in (0, 1)")
    if training.early_stopping.patience < 0:
        raise ConfigError("early_stopping.patience must be non-negative")
    if training.amp == "fp16" and config.runtime.device in {"cpu", "mps"}:
        raise ConfigError("fp16 AMP is only supported on CUDA devices")

    if data.image_size[0] <= 0 or data.image_size[1] <= 0:
        raise ConfigError("data.image_size dimensions must be positive")
    if data.num_classes is not None and data.num_classes <= 0:
        raise ConfigError("data.num_classes must be positive")
    if data.split.mode in {"random", "random_stratified", "group"} and not (
        0.0 < data.split.val_fraction < 1.0
    ):
        raise ConfigError("data.split.val_fraction must be in (0, 1)")
    if (
        isinstance(data.class_names, tuple)
        and data.num_classes is not None
        and len(data.class_names) != data.num_classes
    ):
        raise ConfigError("data.class_names length must match data.num_classes")
    if data.value_range.mode == "fixed":
        if data.value_range.min is None or data.value_range.max is None:
            raise ConfigError("fixed value_range requires min and max")
        if data.value_range.min >= data.value_range.max:
            raise ConfigError("value_range.min must be smaller than value_range.max")
    _check_channel_lengths(data)

    bootstrap = config.evaluation.bootstrap
    if bootstrap.n_resamples <= 0:
        raise ConfigError("bootstrap.n_resamples must be positive")
    if not 0.0 < bootstrap.confidence < 1.0:
        raise ConfigError("bootstrap.confidence must be in (0, 1)")
    if config.evaluation.calibration.bins <= 0:
        raise ConfigError("calibration.bins must be positive")
    if config.evaluation.latency.warmup < 0:
        raise ConfigError("latency.warmup must be non-negative")
    if config.evaluation.latency.iterations <= 0:
        raise ConfigError("latency.iterations must be positive")
