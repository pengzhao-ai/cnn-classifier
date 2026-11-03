"""Deterministic synthetic dataset adapter for tests and smoke runs."""

from __future__ import annotations

import torch

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.core.hashing import stable_hash
from cnn_classifier.core.types import DatasetBundle
from cnn_classifier.data.adapters._common import (
    RawSample,
    assign_split,
    build_bundle,
)
from cnn_classifier.data.adapters.registry import register_dataset


@register_dataset("synthetic")
class SyntheticAdapter:
    """Generate a tiny deterministic tensor dataset."""

    @staticmethod
    def build(config: DataConfig) -> DatasetBundle:
        params = config.adapter_params
        num_samples = int(params.get("num_samples", 64))
        num_classes = int(params.get("num_classes", config.num_classes or 2))
        if num_samples <= 0 or num_classes <= 0:
            raise ConfigError("synthetic adapter requires positive sample and class counts")
        image_size = tuple(params.get("image_size", config.image_size))
        channels = 3 if config.channels == "rgb" else 1
        generator = torch.Generator().manual_seed(config.split.seed)
        images = torch.rand(
            num_samples, channels, image_size[0], image_size[1], generator=generator
        )
        if isinstance(config.class_names, tuple):
            class_names = config.class_names
            if len(class_names) != num_classes:
                raise ConfigError("data.class_names length must match synthetic num_classes")
        else:
            class_names = tuple(f"class_{index}" for index in range(num_classes))
        labels = [class_names[index % num_classes] for index in range(num_samples)]
        if config.num_classes is not None and config.num_classes != num_classes:
            raise ConfigError("synthetic num_classes must match data.num_classes")
        samples = [
            RawSample(source=images[index], label=labels[index], split="train")
            for index in range(num_samples)
        ]
        split, mode = assign_split(samples, config)
        fingerprint = stable_hash(
            {
                "adapter": "synthetic",
                "num_samples": num_samples,
                "num_classes": num_classes,
                "image_size": list(image_size),
                "seed": config.split.seed,
            }
        )
        return build_bundle(
            samples,
            config,
            class_names=class_names,
            split=split,
            mode=mode,
            fingerprint=fingerprint,
        )
