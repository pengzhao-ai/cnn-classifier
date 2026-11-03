from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.data.adapters import build_dataset
from cnn_classifier.data.contract import build_model_contract, write_model_contract


@pytest.fixture
def bundle(tmp_path):
    for split in ("train", "val"):
        for cls, value in (("cat", 10), ("dog", 200)):
            directory = tmp_path / split / cls
            directory.mkdir(parents=True)
            Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(directory / "x.png")
    return build_dataset(
        DataConfig(
            adapter="image_folder",
            root=tmp_path,
            class_names="auto",
            num_classes=2,
            image_size=(8, 8),
            channels="rgb",
        )
    )


def test_model_contract_records_tensor_shape_and_label_map(bundle):
    contract = build_model_contract(bundle, model_name="residual_cnn")
    assert contract.input_shape == (3, 8, 8)
    assert contract.dtype == "float32"
    assert contract.label_map == {"cat": 0, "dog": 1}
    assert contract.stem_adaptation is None


def test_write_model_contract_is_deterministic(bundle, tmp_path):
    contract = build_model_contract(
        bundle, model_name="residual_cnn", stem_adaptation="keep_7x7_stride1"
    )
    path = write_model_contract(contract, tmp_path / "data" / "model_contract.json")
    payload = json.loads(path.read_text())
    assert payload["input_shape"] == [3, 8, 8]
    assert payload["label_map"] == {"cat": 0, "dog": 1}
    assert payload["stem_adaptation"] == "keep_7x7_stride1"
    assert payload["normalization"]["mean"]
