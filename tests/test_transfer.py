import pytest

from cnn_classifier.config.schema import ModelConfig, TransferConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.models import build_model
from cnn_classifier.models.transfer import build_param_groups


def test_frozen_adapt_3x3_is_rejected():
    with pytest.raises(ConfigError, match="adapt_3x3"):
        build_model(
            ModelConfig(
                name="torchvision",
                params={"model": "resnet18", "stem_adaptation": "adapt_3x3"},
                transfer={"weights": None, "freeze": "backbone"},
            ),
            num_classes=10,
            in_channels=3,
            image_size=(32, 32),
        )


def test_unmatched_param_group_raises():
    model = build_model(
        ModelConfig(
            name="torchvision",
            params={"model": "resnet18", "stem_adaptation": "keep_7x7_stride1"},
        ),
        num_classes=10,
        in_channels=3,
        image_size=(32, 32),
    )
    transfer = TransferConfig(
        param_groups=[{"pattern": "^does_not_exist\\.", "lr_multiplier": 10.0}]
    )
    with pytest.raises(ConfigError, match="match"):
        build_param_groups(model, transfer, base_lr=1e-3, base_weight_decay=5e-4)


def test_param_group_applies_lr_multiplier():
    model = build_model(
        ModelConfig(
            name="torchvision",
            params={"model": "resnet18", "stem_adaptation": "keep_7x7_stride1"},
        ),
        num_classes=10,
        in_channels=3,
        image_size=(32, 32),
    )
    transfer = TransferConfig(param_groups=[{"pattern": "^fc\\.", "lr_multiplier": 10.0}])
    groups = build_param_groups(model, transfer, base_lr=1e-3, base_weight_decay=5e-4)
    fc_group = next(group for group in groups if group["lr"] == pytest.approx(1e-2))
    assert fc_group["params"]
