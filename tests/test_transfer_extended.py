from __future__ import annotations

import pytest
import torch
from torch import nn

from cnn_classifier.config.schema import ModelConfig, TransferConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.models import build_model
from cnn_classifier.models.transfer import (
    apply_stem_adaptation,
    apply_transfer,
    build_param_groups,
    replace_classifier,
    unfreeze_modules,
)


class ClassifierModel(nn.Module):
    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.features = nn.Linear(4, 4)
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _resnet() -> nn.Module:
    return build_model(
        ModelConfig(
            name="torchvision",
            params={"model": "resnet18", "stem_adaptation": "none"},
        ),
        num_classes=4,
        in_channels=3,
        image_size=(32, 32),
    )


def test_replace_fc_classifier_with_and_without_dropout():
    model = _resnet()
    replaced = replace_classifier(model, 7, dropout=0.2)
    assert isinstance(replaced.fc, nn.Sequential)
    assert isinstance(replaced.fc[-1], nn.Linear)
    assert replaced.fc[-1].out_features == 7

    direct = replace_classifier(_resnet(), 5)
    assert isinstance(direct.fc, nn.Linear)
    assert direct.fc.out_features == 5


def test_replace_linear_and_sequential_classifier_heads():
    linear = replace_classifier(ClassifierModel(nn.Linear(4, 3)), 6)
    assert isinstance(linear.classifier, nn.Linear)
    assert linear.classifier.out_features == 6

    sequential = replace_classifier(
        ClassifierModel(nn.Sequential(nn.Dropout(0.1), nn.Linear(4, 3))), 2
    )
    assert isinstance(sequential.classifier, nn.Sequential)
    assert sequential.classifier[-1].out_features == 2


def test_replace_classifier_rejects_unsupported_heads():
    with pytest.raises(ConfigError, match="unsupported"):
        replace_classifier(nn.Sequential(nn.Linear(2, 2)), 1)
    with pytest.raises(ConfigError, match="Linear"):
        replace_classifier(ClassifierModel(nn.Sequential(nn.ReLU())), 1)


def test_stem_adaptation_modes_and_errors():
    original = _resnet()
    assert apply_stem_adaptation(original, "none") is original

    kept = apply_stem_adaptation(_resnet(), "keep_7x7_stride1")
    assert isinstance(kept.conv1, nn.Conv2d)
    assert kept.conv1.stride == (1, 1)
    assert isinstance(kept.maxpool, nn.Identity)

    adapted = apply_stem_adaptation(_resnet(), "adapt_3x3")
    assert isinstance(adapted.conv1, nn.Conv2d)
    assert adapted.conv1.kernel_size == (3, 3)
    with pytest.raises(ConfigError, match="input channels"):
        apply_stem_adaptation(_resnet(), "keep_7x7_stride1", in_channels=1)
    with pytest.raises(ConfigError, match="unknown stem"):
        apply_stem_adaptation(_resnet(), "bogus")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="conv1"):
        apply_stem_adaptation(nn.Linear(2, 2), "keep_7x7_stride1")


def test_apply_transfer_freezes_backbone_and_optionally_bn():
    model = _resnet()
    apply_transfer(model, TransferConfig(freeze="backbone", train_bn=False), stem_adaptation="none")
    conv = next(model.conv1.parameters())
    assert conv.requires_grad is False
    assert model.fc.weight.requires_grad is True
    bn = next(module for module in model.modules() if isinstance(module, nn.BatchNorm2d))
    assert all(parameter.requires_grad is False for parameter in bn.parameters())

    train_bn = _resnet()
    apply_transfer(
        train_bn, TransferConfig(freeze="all_but_head", train_bn=True), stem_adaptation="none"
    )
    bn = next(module for module in train_bn.modules() if isinstance(module, nn.BatchNorm2d))
    assert all(parameter.requires_grad is True for parameter in bn.parameters())


def test_apply_transfer_rejects_frozen_adapt_3x3():
    with pytest.raises(ConfigError, match="adapt_3x3"):
        apply_transfer(_resnet(), TransferConfig(freeze="backbone"), stem_adaptation="adapt_3x3")


def test_build_param_groups_requires_trainable_parameters():
    model = _resnet()
    for parameter in model.parameters():
        parameter.requires_grad = False
    with pytest.raises(ConfigError, match="no trainable"):
        build_param_groups(model, TransferConfig(), base_lr=1e-3, base_weight_decay=0.0)


def test_unfreeze_modules_validates_names():
    model = _resnet()
    apply_transfer(model, TransferConfig(freeze="backbone"), stem_adaptation="none")
    unfreeze_modules(model, ["layer1"])
    assert next(model.layer1.parameters()).requires_grad is True
    with pytest.raises(ConfigError, match="unknown module"):
        unfreeze_modules(model, ["missing"])
