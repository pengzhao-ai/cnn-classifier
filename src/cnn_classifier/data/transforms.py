"""Declarative torchvision v2 transform construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from cnn_classifier.config.schema import DataConfig, TransformConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.core.registry import Registry
from cnn_classifier.core.types import NormalizationStats

Transform = Callable[[Any], torch.Tensor]
TransformFactory = Callable[[TransformConfig, NormalizationStats | None], Callable[[Any], Any]]
TRANSFORMS: Registry[TransformFactory] = Registry("transform")


class EnsureTensor:
    """Convert PIL/NumPy input to a float32 tensor, leaving tensors untouched."""

    def __call__(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(torch.float32)
        if isinstance(value, (Image.Image, np.ndarray)):
            image = v2.functional.to_image(value)
            return v2.functional.to_dtype(image, dtype=torch.float32, scale=True)
        raise ConfigError(f"unsupported image type: {type(value).__name__}")


def _normalize_pair(
    spec: TransformConfig,
    normalization: NormalizationStats | None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    params = spec.params()
    mean = params.get("mean", "auto")
    std = params.get("std", "auto")
    if mean == "auto" or std == "auto":
        if normalization is None:
            raise ConfigError(
                f"transform {spec.name!r} needs normalization statistics; "
                "set data.normalization.mode=supplied or add a compute pipeline"
            )
        return normalization.mean, normalization.std
    if not isinstance(mean, (list, tuple)) or not isinstance(std, (list, tuple)):
        raise ConfigError("Normalize mean/std must be lists/tuples or 'auto'")
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


def _resize(spec: TransformConfig, _: NormalizationStats | None) -> Callable[[Any], Any]:
    params = spec.params()
    return v2.Resize(params.get("size"), antialias=True)


def _random_crop(spec: TransformConfig, _: NormalizationStats | None) -> Callable[[Any], Any]:
    params = spec.params()
    return v2.RandomCrop(params.get("size"), padding=params.get("padding", 0))


def _random_resized_crop(
    spec: TransformConfig, _: NormalizationStats | None
) -> Callable[[Any], Any]:
    params = spec.params()
    return v2.RandomResizedCrop(
        params.get("size"),
        scale=tuple(params.get("scale", (0.08, 1.0))),
        ratio=tuple(params.get("ratio", (0.75, 1.3333333333333333))),
        antialias=True,
    )


def _color_jitter(spec: TransformConfig, _: NormalizationStats | None) -> Callable[[Any], Any]:
    params = spec.params()
    return v2.ColorJitter(
        brightness=params.get("brightness", 0.0),
        contrast=params.get("contrast", 0.0),
        saturation=params.get("saturation", 0.0),
        hue=params.get("hue", 0.0),
    )


def _gaussian_blur(spec: TransformConfig, _: NormalizationStats | None) -> Callable[[Any], Any]:
    params = spec.params()
    return v2.GaussianBlur(
        kernel_size=params.get("kernel_size", 3), sigma=params.get("sigma", (0.1, 2.0))
    )


def _rand_augment(spec: TransformConfig, _: NormalizationStats | None) -> Callable[[Any], Any]:
    params = spec.params()
    return v2.RandAugment(
        num_ops=params.get("num_ops", 2),
        magnitude=params.get("magnitude", 9),
        num_magnitude_bins=params.get("num_magnitude_bins", 31),
    )


def _auto_augment(spec: TransformConfig, _: NormalizationStats | None) -> Callable[[Any], Any]:
    params = spec.params()
    policy_name = str(params.get("policy", "imagenet")).lower()
    policies = {
        "imagenet": v2.AutoAugmentPolicy.IMAGENET,
        "cifar10": v2.AutoAugmentPolicy.CIFAR10,
        "svhn": v2.AutoAugmentPolicy.SVHN,
    }
    if policy_name not in policies:
        raise ConfigError(f"unknown AutoAugment policy {policy_name!r}")
    return v2.AutoAugment(policy=policies[policy_name])


def _random_erasing(spec: TransformConfig, _: NormalizationStats | None) -> Callable[[Any], Any]:
    params = spec.params()
    return v2.RandomErasing(
        p=params.get("p", 0.5),
        scale=tuple(params.get("scale", (0.02, 0.33))),
        ratio=tuple(params.get("ratio", (0.3, 3.3))),
    )


def _normalize(
    spec: TransformConfig, normalization: NormalizationStats | None
) -> Callable[[Any], Any]:
    mean, std = _normalize_pair(spec, normalization)
    return v2.Normalize(mean=mean, std=std)


_BUILTINS: dict[str, TransformFactory] = {
    "resize": _resize,
    "randomcrop": _random_crop,
    "randomresizedcrop": _random_resized_crop,
    "colorjitter": _color_jitter,
    "gaussianblur": _gaussian_blur,
    "randaugment": _rand_augment,
    "autoaugment": _auto_augment,
    "random erasing": _random_erasing,
    "random_erasing": _random_erasing,
    "randomerasing": _random_erasing,
    "normalize": _normalize,
}


def register_transform(name: str) -> Callable[[TransformFactory], TransformFactory]:
    """Register a custom transform factory."""
    return TRANSFORMS.register(name)


def _factory_for(name: str) -> TransformFactory:
    lowered = name.lower()
    if lowered == "totensor":
        return lambda _spec, _norm: EnsureTensor()
    if lowered == "randomhorizontalflip":
        return lambda spec, _norm: v2.RandomHorizontalFlip(p=spec.params().get("p", 0.5))
    if lowered == "randomverticalflip":
        return lambda spec, _norm: v2.RandomVerticalFlip(p=spec.params().get("p", 0.5))
    if lowered == "randomrotation":
        return lambda spec, _norm: v2.RandomRotation(spec.params().get("degrees", 10))
    if lowered == "centercrop":
        return lambda spec, _norm: v2.CenterCrop(spec.params().get("size"))
    if lowered == "grayscale":
        return lambda spec, _norm: v2.Grayscale(
            num_output_channels=spec.params().get("num_output_channels", 1)
        )
    if lowered in _BUILTINS:
        return _BUILTINS[lowered]
    try:
        return TRANSFORMS.get(name)
    except Exception as exc:
        raise ConfigError(f"unknown transform {name!r}") from exc


def build_transforms(
    config: DataConfig,
    normalization: NormalizationStats | None,
    train: bool,
) -> Transform:
    """Build the train or evaluation transform pipeline."""
    specs = config.transforms["train" if train else "eval"]
    operations: list[Callable[[Any], Any]] = [EnsureTensor()]
    for spec in specs:
        if spec.name.lower() == "totensor":
            continue
        operations.append(_factory_for(spec.name)(spec, normalization))
    return v2.Compose(operations)


def transform_params(spec: TransformConfig) -> Mapping[str, Any]:
    """Return transform parameters; useful for logging and tests."""
    return spec.params()
