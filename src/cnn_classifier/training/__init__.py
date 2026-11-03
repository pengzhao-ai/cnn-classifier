"""Training components: optimization, scheduling, losses, AMP, and callbacks."""

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
from cnn_classifier.training.losses import MixedBatch, build_loss, mixup_cutmix_batch
from cnn_classifier.training.optim import build_optimizer
from cnn_classifier.training.schedulers import build_scheduler
from cnn_classifier.training.state import TrainingState

__all__ = [
    "AmpPolicy",
    "Callback",
    "EMACallback",
    "EarlyStopping",
    "GradientMonitor",
    "MixedBatch",
    "ModelCheckpoint",
    "NaNDetector",
    "TrackerCallback",
    "TrainingState",
    "UnfreezeCallback",
    "autocast_context",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
    "mixup_cutmix_batch",
    "resolve_amp",
]
