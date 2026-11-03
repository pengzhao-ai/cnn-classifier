from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.data.adapters import build_dataset
from cnn_classifier.data.inspection import inspect_dataset


@pytest.fixture
def bundle_with_shared_group(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cnn_classifier.data.adapters._common.assert_no_group_overlap",
        lambda *args, **kwargs: None,
    )
    for index in range(4):
        Image.fromarray(np.full((4, 4, 3), index * 10, dtype=np.uint8)).save(
            tmp_path / f"{index}.png"
        )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,label,split,group\n"
        "0.png,a,train,g1\n"
        "1.png,b,train,g1\n"
        "2.png,a,val,g1\n"
        "3.png,b,val,g2\n",
        encoding="utf-8",
    )
    config = DataConfig(
        adapter="manifest",
        manifest=manifest,
        image_root=tmp_path,
        class_names=("a", "b"),
        num_classes=2,
        image_size=(4, 4),
    )
    return build_dataset(config)


def test_inspection_detects_group_leakage(bundle_with_shared_group):
    report = inspect_dataset(bundle_with_shared_group)
    assert report.group_leakage
    assert any("group" in warning.lower() for warning in report.warnings)


def test_inspection_counts_classes_and_sizes(bundle_with_shared_group):
    report = inspect_dataset(bundle_with_shared_group)
    assert report.class_counts["train"]["a"] == 1
    assert report.class_counts["val"]["b"] == 1
    assert report.sizes["4x4x3"] == 4
