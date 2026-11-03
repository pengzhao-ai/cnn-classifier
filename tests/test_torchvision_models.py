import torch
from torch import nn

from cnn_classifier.config.schema import ModelConfig
from cnn_classifier.models import build_model
from cnn_classifier.models.torchvision import apply_stem_adaptation
from cnn_classifier.models.transfer import unfreeze_modules


def test_keep_7x7_stride1_preserves_kernel_and_removes_maxpool():
    model = build_model(
        ModelConfig(
            name="torchvision",
            params={"model": "resnet18", "stem_adaptation": "keep_7x7_stride1"},
        ),
        num_classes=10,
        in_channels=3,
        image_size=(32, 32),
    )
    assert model.conv1.kernel_size == (7, 7)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)
    assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 10)


def test_adapt_3x3_centers_pretrained_kernel():
    model = build_model(
        ModelConfig(
            name="torchvision",
            params={"model": "resnet18", "stem_adaptation": "none"},
            transfer={"weights": None},
        ),
        num_classes=10,
        in_channels=3,
        image_size=(32, 32),
    )
    original = model.conv1.weight.data.clone()
    apply_stem_adaptation(model, "adapt_3x3")
    assert model.conv1.kernel_size == (3, 3)
    assert torch.equal(model.conv1.weight.data, original[:, :, 2:5, 2:5])


def test_linear_probe_freezes_backbone_and_keeps_head_trainable():
    model = build_model(
        ModelConfig(
            name="torchvision",
            params={"model": "resnet18", "stem_adaptation": "keep_7x7_stride1"},
            transfer={"weights": None, "freeze": "backbone"},
        ),
        num_classes=10,
        in_channels=3,
        image_size=(32, 32),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 5_130


def test_unfreeze_modules_restores_requested_layers():
    model = build_model(
        ModelConfig(
            name="torchvision",
            params={"model": "resnet18", "stem_adaptation": "keep_7x7_stride1"},
            transfer={"weights": None, "freeze": "backbone"},
        ),
        num_classes=10,
        in_channels=3,
        image_size=(32, 32),
    )
    unfreeze_modules(model, ["layer4"])
    assert any(p.requires_grad for p in model.layer4.parameters())
