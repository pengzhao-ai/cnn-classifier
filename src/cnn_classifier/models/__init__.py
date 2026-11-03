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
from cnn_classifier.models.torchvision import TorchvisionBuilder, TorchvisionParams
from cnn_classifier.models.transfer import (
    apply_stem_adaptation,
    apply_transfer,
    build_param_groups,
    replace_classifier,
    unfreeze_modules,
)

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
    "TorchvisionBuilder",
    "TorchvisionParams",
    "apply_stem_adaptation",
    "apply_transfer",
    "build_model",
    "build_param_groups",
    "model_summary",
    "register_model",
    "replace_classifier",
    "unfreeze_modules",
]
