"""Deterministic synthetic dataset adapter for tests and smoke runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
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
from cnn_classifier.data.splits import SplitIndices


def _three_way_split(
    labels: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> SplitIndices:
    if not 0.0 <= test_fraction < 1.0:
        raise ConfigError("synthetic test_fraction must be in [0, 1)")
    if test_fraction + val_fraction >= 1.0:
        raise ConfigError("synthetic test_fraction + val_fraction must be smaller than 1")
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in sorted(by_class):
        indices = np.array(by_class[label])
        shuffled = indices[rng.permutation(len(indices))].tolist()
        test_count = round(len(shuffled) * test_fraction)
        val_count = round(len(shuffled) * val_fraction)
        if len(shuffled) >= 3:
            test_count = max(1, test_count)
            val_count = max(1, val_count)
        test.extend(shuffled[:test_count])
        val.extend(shuffled[test_count : test_count + val_count])
        train.extend(shuffled[test_count + val_count :])
    return SplitIndices(tuple(sorted(train)), tuple(sorted(val)), tuple(sorted(test)))


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
        test_fraction = float(params.get("test_fraction", 0.0))
        if test_fraction > 0:
            if config.split.mode not in {"random", "random_stratified"}:
                raise ConfigError(
                    "synthetic test_fraction requires a random or random_stratified split"
                )
            split = _three_way_split(
                labels, config.split.val_fraction, test_fraction, config.split.seed
            )
            mode = "random_stratified"
        else:
            split, mode = assign_split(samples, config)
        fingerprint = stable_hash(
            {
                "adapter": "synthetic",
                "num_samples": num_samples,
                "num_classes": num_classes,
                "image_size": list(image_size),
                "seed": config.split.seed,
                "test_fraction": test_fraction,
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
