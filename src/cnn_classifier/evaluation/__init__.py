"""Evaluation metrics, calibration, bootstrap, and comparisons."""

from cnn_classifier.evaluation.bootstrap import (
    ConfidenceInterval,
    bootstrap_ci,
    paired_bootstrap_difference,
)
from cnn_classifier.evaluation.calibration import (
    CalibrationBin,
    CalibrationReport,
    expected_calibration_error,
)
from cnn_classifier.evaluation.metrics import (
    ClassMetrics,
    MetricsReport,
    accuracy,
    classification_metrics,
)

__all__ = [
    "CalibrationBin",
    "CalibrationReport",
    "ClassMetrics",
    "ConfidenceInterval",
    "MetricsReport",
    "accuracy",
    "bootstrap_ci",
    "classification_metrics",
    "expected_calibration_error",
    "paired_bootstrap_difference",
]
