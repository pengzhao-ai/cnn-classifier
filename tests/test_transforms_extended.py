from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from cnn_classifier.config.schema import DataConfig, TransformConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.core.types import NormalizationStats
from cnn_classifier.data.transforms import build_transforms, transform_params


def _config(train: list[TransformConfig]) -> DataConfig:
    return DataConfig(
        adapter="synthetic",
        transforms={"train": train, "eval": []},
    )


def _image() -> Image.Image:
    return Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8))


@pytest.mark.parametrize(
    ("name", "params", "channels"),
    [
        ("Resize", {"size": [8, 8]}, 3),
        ("RandomCrop", {"size": [8, 8]}, 3),
        ("RandomResizedCrop", {"size": [8, 8]}, 3),
        ("RandomHorizontalFlip", {"p": 1.0}, 3),
        ("RandomVerticalFlip", {"p": 1.0}, 3),
        ("RandomRotation", {"degrees": 10}, 3),
        ("CenterCrop", {"size": [8, 8]}, 3),
        ("ColorJitter", {"brightness": 0.1}, 3),
        ("GaussianBlur", {"kernel_size": 3}, 3),
        ("RandAugment", {"num_ops": 1, "magnitude": 5}, 3),
        ("AutoAugment", {"policy": "cifar10"}, 3),
        ("Grayscale", {"num_output_channels": 1}, 1),
        ("RandomErasing", {"p": 1.0}, 3),
    ],
)
def test_builtin_transforms_return_tensors(name, params, channels):
    transform = build_transforms(
        _config([TransformConfig(name=name, **params)]),
        None,
        train=True,
    )
    output = transform(_image())
    assert isinstance(output, torch.Tensor)
    assert output.shape == (channels, 8, 8)


def test_normalize_auto_and_explicit_stats():
    stats = NormalizationStats(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), source="supplied")
    auto = build_transforms(
        _config([TransformConfig(name="Normalize", mean="auto", std="auto")]),
        stats,
        train=True,
    )
    auto_output = auto(torch.full((3, 2, 2), 0.5))
    assert float(auto_output.mean()) == pytest.approx(0.0)

    explicit = build_transforms(
        _config([TransformConfig(name="Normalize", mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
        None,
        train=True,
    )
    explicit_output = explicit(torch.ones(3, 2, 2))
    assert float(explicit_output.mean()) == pytest.approx(1.0)


def test_normalize_requires_stats_or_valid_pairs():
    with pytest.raises(ConfigError, match="normalization statistics"):
        build_transforms(
            _config([TransformConfig(name="Normalize", mean="auto", std="auto")]), None, train=True
        )
    with pytest.raises(ConfigError, match="lists"):
        build_transforms(
            _config([TransformConfig(name="Normalize", mean=0.5, std=[0.5, 0.5, 0.5])]),
            None,
            train=True,
        )


def test_unknown_auto_augment_policy_is_rejected():
    with pytest.raises(ConfigError, match="AutoAugment policy"):
        build_transforms(
            _config([TransformConfig(name="AutoAugment", policy="unknown")]), None, train=True
        )


def test_transform_params_returns_extra_fields():
    spec = TransformConfig(name="RandomCrop", size=[4, 4], padding=2)
    assert transform_params(spec) == {"size": [4, 4], "padding": 2}
