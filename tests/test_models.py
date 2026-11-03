import pytest
import torch

from cnn_classifier.config.schema import ModelConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.models import build_model


def test_simple_cnn_output_shape():
    model = build_model(
        ModelConfig(name="simple_cnn", params={"channels": [4, 8]}),
        num_classes=3,
        in_channels=3,
        image_size=(8, 8),
    )
    assert model(torch.zeros(2, 3, 8, 8)).shape == (2, 3)


def test_reference_residual_config_has_expected_parameter_count():
    params = {
        "stem": {"channels": 64, "kernel_size": 3, "stride": 1, "pool": False},
        "stages": [
            {"channels": 64, "conv": False, "blocks": 1, "downsample": "maxpool"},
            {"channels": 128, "conv": True, "blocks": 1, "downsample": "maxpool"},
            {"channels": 256, "conv": True, "blocks": 1, "downsample": "maxpool"},
            {"channels": 512, "conv": True, "blocks": 1, "downsample": "none"},
        ],
        "dropout": 0.3,
    }
    model = build_model(
        ModelConfig(name="residual_cnn", params=params),
        num_classes=10,
        in_channels=3,
        image_size=(32, 32),
    )
    assert sum(p.numel() for p in model.parameters()) == 7_827_786


def test_residual_cnn_without_residual_or_norm_runs():
    params = {
        "stem": {"channels": 4, "kernel_size": 3, "stride": 1, "pool": False},
        "stages": [
            {"channels": 8, "conv": True, "blocks": 1, "downsample": "maxpool"},
            {"channels": 8, "conv": False, "blocks": 1, "downsample": "none"},
        ],
        "dropout": 0.0,
        "residual": False,
        "norm": "none",
    }
    model = build_model(
        ModelConfig(name="residual_cnn", params=params),
        num_classes=2,
        in_channels=1,
        image_size=(8, 8),
    )
    assert model(torch.zeros(2, 1, 8, 8)).shape == (2, 2)


def test_unknown_model_is_rejected():
    with pytest.raises(ConfigError, match="unknown model"):
        build_model(
            ModelConfig(name="not_real"),
            num_classes=2,
            in_channels=3,
            image_size=(8, 8),
        )
