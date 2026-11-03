"""Evaluation metrics, calibration, bootstrap, comparisons, and inference."""

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
from cnn_classifier.evaluation.error_analysis import (
    ErrorExample,
    confusion_pairs,
    most_confident_errors,
)
from cnn_classifier.evaluation.evaluator import (
    EvaluationResult,
    evaluate_model,
    save_predictions,
)
from cnn_classifier.evaluation.latency import LatencyReport, measure_latency
from cnn_classifier.evaluation.metrics import (
    ClassMetrics,
    MetricsReport,
    accuracy,
    classification_metrics,
)
from cnn_classifier.evaluation.predict import Predictions, predict_paths

__all__ = [
    "CalibrationBin",
    "CalibrationReport",
    "ClassMetrics",
    "ConfidenceInterval",
    "ErrorExample",
    "EvaluationResult",
    "LatencyReport",
    "MetricsReport",
    "Predictions",
    "accuracy",
    "bootstrap_ci",
    "classification_metrics",
    "confusion_pairs",
    "evaluate_model",
    "expected_calibration_error",
    "measure_latency",
    "most_confident_errors",
    "paired_bootstrap_difference",
    "predict_paths",
    "save_predictions",
]
