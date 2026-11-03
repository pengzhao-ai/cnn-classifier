import pytest

from cnn_classifier.config import load_config
from cnn_classifier.models import build_model


def test_cifar_scratch_config_builds_reference_model():
    cfg = load_config("configs/cifar10/scratch_baseline.yaml")
    model = build_model(cfg.model, num_classes=10, in_channels=3, image_size=(32, 32))
    assert sum(p.numel() for p in model.parameters()) == 7_827_786
    assert cfg.data.split.mode == "random"
    assert cfg.training.epochs == 50


def test_transfer_configs_use_expected_freeze_modes():
    scratch = load_config("configs/cifar10/resnet18_scratch_cifar.yaml")
    probe = load_config("configs/cifar10/resnet18_linear_probe_cifar.yaml")
    fine = load_config("configs/cifar10/resnet18_finetune_cifar.yaml")

    assert scratch.model.transfer.weights is None
    assert probe.model.transfer.weights is not None
    assert fine.model.transfer.weights is not None
    assert probe.model.transfer.freeze == "backbone"
    assert fine.model.transfer.freeze == "none"

    for cfg in (scratch, probe, fine):
        assert cfg.model.name == "torchvision"
        assert cfg.model.params["model"] == "resnet18"
        assert cfg.model.params["stem_adaptation"] == "keep_7x7_stride1"
        assert cfg.data.image_size == (32, 32)
        assert cfg.training.epochs == 30

    assert scratch.data == probe.data == fine.data


def test_linear_probe_has_expected_trainable_parameters():
    probe = load_config("configs/cifar10/resnet18_linear_probe_cifar.yaml")
    model_config = probe.model.model_copy(
        update={"transfer": probe.model.transfer.model_copy(update={"weights": None})}
    )
    model = build_model(model_config, num_classes=10, in_channels=3, image_size=(32, 32))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 5_130


def test_finetune_uses_differential_learning_rates():
    fine = load_config("configs/cifar10/resnet18_finetune_cifar.yaml")
    assert fine.training.optimizer.lr == pytest.approx(1.0e-4)
    assert len(fine.model.transfer.param_groups) == 1
    group = fine.model.transfer.param_groups[0]
    assert group.pattern == r"^fc\."
    assert fine.training.optimizer.lr * group.lr_multiplier == pytest.approx(1.0e-3)
