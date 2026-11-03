"""Diagnostics: overfit checks, shape tracing, gradients, activations, determinism."""

from cnn_classifier.diagnostics.activations import ActivationStat, activation_report
from cnn_classifier.diagnostics.architecture import ArchitectureStage, architecture_stages
from cnn_classifier.diagnostics.determinism import determinism_check
from cnn_classifier.diagnostics.gradients import GradientStat, gradient_report
from cnn_classifier.diagnostics.sanity import OverfitResult, overfit_batch
from cnn_classifier.diagnostics.tracing import LayerShape, trace_shapes

__all__ = [
    "ActivationStat",
    "ArchitectureStage",
    "GradientStat",
    "LayerShape",
    "OverfitResult",
    "activation_report",
    "architecture_stages",
    "determinism_check",
    "gradient_report",
    "overfit_batch",
    "trace_shapes",
]
