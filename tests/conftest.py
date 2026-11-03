from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def minimal_config_dict() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment": {"name": "test"},
        "data": {
            "adapter": "synthetic",
            "class_names": ["a", "b"],
            "num_classes": 2,
            "adapter_params": {"num_samples": 8, "num_classes": 2, "image_size": [8, 8]},
        },
        "model": {"name": "simple_cnn", "params": {"channels": [4, 8]}},
        "training": {"epochs": 2, "batch_size": 2},
    }


@pytest.fixture
def write_config(tmp_path: Path):
    def _write(payload: dict[str, Any], name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    return _write
