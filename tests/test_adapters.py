import numpy as np
import pytest
from PIL import Image

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.data.adapters import build_dataset


def test_image_folder_sorts_classes_and_returns_triples(tmp_path):
    for split in ("train", "val"):
        for cls, color in (("cat", 10), ("dog", 200)):
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8)).save(d / "x.png")
    cfg = DataConfig(
        adapter="image_folder",
        root=tmp_path,
        class_names="auto",
        num_classes=2,
        image_size=(8, 8),
        channels="rgb",
    )
    bundle = build_dataset(cfg)
    assert bundle.class_names == ("cat", "dog")
    image, label, index = bundle.train[0]
    assert image.shape == (3, 8, 8)
    assert isinstance(label, int)
    assert index == 0


def test_manifest_adapter_reads_split_and_classes(tmp_path):
    for name in ("a.png", "b.png", "c.png", "d.png"):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp_path / name)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,label,split\na.png,cat,train\nb.png,dog,train\nc.png,cat,val\nd.png,dog,test\n",
        encoding="utf-8",
    )
    bundle = build_dataset(
        DataConfig(
            adapter="manifest",
            manifest=manifest,
            image_root=tmp_path,
            class_names=("cat", "dog"),
            num_classes=2,
            image_size=(4, 4),
        )
    )
    assert bundle.class_names == ("cat", "dog")
    assert len(bundle.train) == 2
    assert len(bundle.val) == 1
    assert len(bundle.test) == 1


def test_synthetic_adapter_returns_contract(synthetic_bundle):
    assert synthetic_bundle.class_names == ("a", "b")
    image, label, index = synthetic_bundle.train[0]
    assert image.shape == (1, 8, 8)
    assert index == 0
    assert synthetic_bundle.model_contract.input_shape == (1, 8, 8)
    assert 0 <= label < 2


def test_unknown_adapter_is_rejected():
    with pytest.raises(ConfigError, match="unknown"):
        build_dataset(DataConfig(adapter="not_real", class_names=("a",), num_classes=1))
