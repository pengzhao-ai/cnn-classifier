import numpy as np
import pytest
import torch
from PIL import Image

from cnn_classifier.config.schema import DataConfig, TransformConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.core.types import NormalizationStats
from cnn_classifier.data.transforms import build_transforms


def test_train_transform_flips_and_returns_tensor():
    config = DataConfig(
        adapter="synthetic",
        transforms={
            "train": [
                TransformConfig(name="RandomHorizontalFlip", p=1.0),
                TransformConfig(name="ToTensor"),
            ],
            "eval": [TransformConfig(name="ToTensor")],
        },
    )
    transform = build_transforms(config, None, train=True)
    image = Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8))
    output = transform(image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 4, 5)


def test_normalize_uses_supplied_stats():
    config = DataConfig(
        adapter="synthetic",
        transforms={
            "train": [TransformConfig(name="Normalize", mean="auto", std="auto")],
            "eval": [],
        },
    )
    stats = NormalizationStats(mean=(0.5,), std=(0.5,), source="supplied")
    transform = build_transforms(config, stats, train=True)
    output = transform(torch.ones(1, 2, 2))
    assert float(output.mean()) == pytest.approx(1.0)


def test_unknown_transform_is_rejected():
    config = DataConfig(
        adapter="synthetic",
        transforms={"train": [TransformConfig(name="NotATransform")], "eval": []},
    )
    with pytest.raises(ConfigError, match="NotATransform"):
        build_transforms(config, None, train=True)
