"""Model builders and registry."""

from cnn_classifier.models.base import (
    MODELS,
    ModelBuildContext,
    ModelBuilder,
    build_model,
    register_model,
)
from cnn_classifier.models.residual_cnn import (
    ResidualCNN,
    ResidualCNNBuilder,
    ResidualCNNParams,
    StageConfig,
    StemConfig,
)
from cnn_classifier.models.simple_cnn import SimpleCNN, SimpleCNNBuilder, SimpleCNNParams
from cnn_classifier.models.summary import model_summary

__all__ = [
    "MODELS",
    "ModelBuildContext",
    "ModelBuilder",
    "ResidualCNN",
    "ResidualCNNBuilder",
    "ResidualCNNParams",
    "SimpleCNN",
    "SimpleCNNBuilder",
    "SimpleCNNParams",
    "StageConfig",
    "StemConfig",
    "build_model",
    "model_summary",
    "register_model",
]
