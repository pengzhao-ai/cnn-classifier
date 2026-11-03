from __future__ import annotations

import copy
from typing import Any

import pytest

from cnn_classifier.config.resolve import validate_config_semantics
from cnn_classifier.config.schema import RunConfig
from cnn_classifier.core.errors import ConfigError


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"training": {"epochs": 0}}, "epochs"),
        ({"training": {"batch_size": 0}}, "batch_size"),
        ({"training": {"grad_accum_steps": 0}}, "grad_accum"),
        ({"training": {"scheduler": {"min_lr_ratio": 1.0}}}, "min_lr_ratio"),
        ({"training": {"optimizer": {"lr": 0.0}}}, "lr"),
        ({"training": {"optimizer": {"weight_decay": -1.0}}}, "weight_decay"),
        ({"training": {"loss": {"label_smoothing": 1.0}}}, "label_smoothing"),
        ({"training": {"mixup_alpha": 1.1}}, "mixup_alpha"),
        ({"training": {"cutmix_alpha": -0.1}}, "cutmix_alpha"),
        ({"training": {"ema": {"enabled": True, "decay": 1.0}}}, "ema"),
        ({"training": {"early_stopping": {"patience": -1}}}, "patience"),
        ({"runtime": {"device": "cpu"}, "training": {"amp": "fp16"}}, "fp16"),
        ({"data": {"image_size": [0, 8]}}, "image_size"),
        ({"data": {"num_classes": 0}}, "num_classes"),
        ({"data": {"split": {"mode": "random", "val_fraction": 0.0}}}, "val_fraction"),
        ({"data": {"num_classes": 3}}, "class_names"),
        ({"data": {"value_range": {"mode": "fixed", "min": 0.0}}}, "value_range"),
        (
            {"data": {"value_range": {"mode": "fixed", "min": 2.0, "max": 1.0}}},
            "value_range",
        ),
        (
            {
                "data": {
                    "channels": "rgb",
                    "normalization": {"mode": "supplied", "mean": [0.5], "std": [0.5]},
                }
            },
            "channels",
        ),
        ({"evaluation": {"bootstrap": {"n_resamples": 0}}}, "n_resamples"),
        ({"evaluation": {"bootstrap": {"confidence": 1.0}}}, "confidence"),
        ({"evaluation": {"calibration": {"bins": 0}}}, "bins"),
        ({"evaluation": {"latency": {"warmup": -1}}}, "warmup"),
        ({"evaluation": {"latency": {"iterations": 0}}}, "iterations"),
    ],
)
def test_invalid_semantics_are_rejected(minimal_config_dict, patch, match):
    payload = _merge(minimal_config_dict, patch)
    with pytest.raises(ConfigError, match=match):
        validate_config_semantics(RunConfig.model_validate(payload))


def test_valid_config_passes_semantics(minimal_config_dict):
    validate_config_semantics(RunConfig.model_validate(minimal_config_dict))
