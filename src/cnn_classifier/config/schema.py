"""Typed configuration models for cnn-classifier.

Known keys are validated strictly; adapter and model parameters remain flexible
and are validated by their registered builders. Cross-field physical checks live
in :mod:`cnn_classifier.config.resolve` so they raise :class:`ConfigError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class ConfigModel(BaseModel):
    """Base for strict configuration models."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ExperimentConfig(ConfigModel):
    name: str
    tags: tuple[str, ...] = ()
    notes: str = ""
    seed: int = 42


class SplitConfig(ConfigModel):
    mode: Literal["random", "random_stratified", "group", "explicit", "train_only"] = (
        "random_stratified"
    )
    val_fraction: float = 0.1
    seed: int = 42
    group_column: str | None = None


class NormalizationConfig(ConfigModel):
    mode: Literal["compute", "supplied", "none"] = "compute"
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    cache: bool = True


class ValueRangeConfig(ConfigModel):
    mode: Literal["auto", "fixed"] = "auto"
    min: float | None = None
    max: float | None = None


class TransformConfig(ConfigModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    name: str

    def params(self) -> dict[str, Any]:
        """Return the transform-specific parameters."""
        return dict(self.model_extra or {})


def _empty_transforms() -> dict[Literal["train", "eval"], list[TransformConfig]]:
    return {"train": [], "eval": []}


class DataConfig(ConfigModel):
    adapter: str
    root: Path | None = None
    manifest: Path | None = None
    image_root: Path | None = None
    class_names: tuple[str, ...] | Literal["auto"] = "auto"
    num_classes: int | None = None
    channels: Literal["rgb", "gray", "as_is"] = "rgb"
    image_size: tuple[int, int] = (32, 32)
    split: SplitConfig = Field(default_factory=SplitConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    transforms: dict[Literal["train", "eval"], list[TransformConfig]] = Field(
        default_factory=_empty_transforms
    )
    value_range: ValueRangeConfig = Field(default_factory=ValueRangeConfig)
    adapter_params: dict[str, Any] = Field(default_factory=dict)


class ParamGroupConfig(ConfigModel):
    pattern: str
    lr_multiplier: float = 1.0
    weight_decay_multiplier: float = 1.0


class UnfreezeEvent(ConfigModel):
    epoch: int
    modules: tuple[str, ...]


class TransferConfig(ConfigModel):
    weights: str | None = None
    freeze: Literal["none", "backbone", "all_but_head"] = "none"
    train_bn: bool = False
    unfreeze_schedule: tuple[UnfreezeEvent, ...] = ()
    param_groups: tuple[ParamGroupConfig, ...] = ()


class ModelConfig(ConfigModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    transfer: TransferConfig = Field(default_factory=TransferConfig)


class OptimizerConfig(ConfigModel):
    name: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9


class SchedulerConfig(ConfigModel):
    name: Literal["constant", "cosine", "cosine_warmup", "step", "plateau"] = "cosine"
    warmup_epochs: int = 0
    min_lr_ratio: float = 0.0
    step_size: int = 10
    gamma: float = 0.1
    step_unit: Literal["epoch", "batch"] = "epoch"


class LossConfig(ConfigModel):
    name: Literal["cross_entropy"] = "cross_entropy"
    label_smoothing: float = 0.0


class EMAConfig(ConfigModel):
    enabled: bool = False
    decay: float = 0.999


class EarlyStoppingConfig(ConfigModel):
    enabled: bool = True
    monitor: str = "val_accuracy"
    mode: Literal["min", "max"] = "max"
    patience: int = 10
    min_delta: float = 0.0


class CheckpointConfig(ConfigModel):
    monitor: str = "val_accuracy"
    mode: Literal["min", "max"] = "max"
    save_last: bool = True


class TrainingConfig(ConfigModel):
    epochs: int = 50
    batch_size: int = 128
    grad_accum_steps: int = 1
    selection_metric: str = "val_accuracy"
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    grad_clip_norm: float | None = None
    amp: Literal["auto", "off", "fp16", "bf16"] = "auto"
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    ema: EMAConfig = Field(default_factory=EMAConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


class BootstrapConfig(ConfigModel):
    enabled: bool = True
    n_resamples: int = 1000
    confidence: float = 0.95
    stratify: bool = True


class CalibrationConfig(ConfigModel):
    bins: int = 15


class LatencyConfig(ConfigModel):
    enabled: bool = True
    warmup: int = 20
    iterations: int = 100


class ErrorAnalysisConfig(ConfigModel):
    max_examples: int = 32


class TTAConfig(ConfigModel):
    enabled: bool = False
    transforms: tuple[Literal["identity", "hflip"], ...] = ("identity", "hflip")
    reduction: Literal["mean_probabilities"] = "mean_probabilities"


class EvaluationConfig(ConfigModel):
    metrics: tuple[str, ...] = (
        "accuracy",
        "top_k",
        "macro_f1",
        "per_class",
        "confusion_matrix",
        "ece",
        "nll",
    )
    top_k: int = 3
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    error_analysis: ErrorAnalysisConfig = Field(default_factory=ErrorAnalysisConfig)
    tta: TTAConfig = Field(default_factory=TTAConfig)


class TrackingConfig(ConfigModel):
    console: bool = True
    csv: bool = True
    tensorboard: bool = True


class ReportConfig(ConfigModel):
    enabled: bool = True
    formats: tuple[Literal["html", "markdown"], ...] = ("html", "markdown")
    figures: tuple[str, ...] = (
        "training_history",
        "confusion_matrix",
        "per_class",
        "calibration",
        "errors",
        "model_summary",
    )


class RuntimeConfig(ConfigModel):
    device: str = "auto"
    deterministic: bool = True
    num_workers: int | Literal["auto"] = "auto"
    persistent_workers: bool = True
    pin_memory: bool | Literal["auto"] = "auto"
    channels_last: bool = False
    compile: bool = False


class RunConfig(ConfigModel):
    schema_version: Literal["1.0"] = "1.0"
    defaults: tuple[str, ...] = ()
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
