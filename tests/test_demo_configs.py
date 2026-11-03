from cnn_classifier.config import load_config
from cnn_classifier.models import build_model


def test_cifar_scratch_config_builds_reference_model():
    cfg = load_config("configs/cifar10/scratch_baseline.yaml")
    model = build_model(cfg.model, num_classes=10, in_channels=3, image_size=(32, 32))
    assert sum(p.numel() for p in model.parameters()) == 7_827_786
    assert cfg.data.split.mode == "random"
    assert cfg.training.epochs == 50
