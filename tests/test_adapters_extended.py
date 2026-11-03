from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.core.errors import ConfigError, DataError
from cnn_classifier.data.adapters import build_dataset


class FakeVisionDataset:
    classes: ClassVar[list[str]] = ["cat", "dog", "bird"]
    targets: ClassVar[list[int] | None] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

    def __init__(self, root: str, train: bool, download: bool) -> None:
        self.train = train

    def __len__(self) -> int:
        return len(self.targets) if self.train else 3

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        value = index if self.train else index % 3
        image = Image.fromarray(np.full((8, 8, 3), (value % 6) * 40, dtype=np.uint8))
        return image, int(self.targets[value] if self.train else value)


class FakeLabelsDataset(FakeVisionDataset):
    classes: ClassVar[list[str]] = ["a", "b"]
    targets: ClassVar[list[int] | None] = None
    labels: ClassVar[list[int]] = [0, 1, 0, 1]

    def __len__(self) -> int:
        return len(self.labels) if self.train else 2

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        value = index if self.train else index % 2
        image = Image.fromarray(np.full((8, 8, 3), (value % 6) * 40, dtype=np.uint8))
        label = self.labels[value] if self.train else value
        return image, int(label)


class NoTargetsDataset(FakeVisionDataset):
    classes: ClassVar[list[str]] = ["a"]
    targets: ClassVar[list[int] | None] = None
    labels: ClassVar[list[int] | None] = None


class BrokenDataset(FakeVisionDataset):
    def __init__(self, root: str, train: bool, download: bool) -> None:
        raise OSError("broken dataset")


def _torchvision_config(tmp_path, **overrides):
    payload = {
        "adapter": "torchvision",
        "root": tmp_path,
        "num_classes": 3,
        "image_size": (8, 8),
        "channels": "rgb",
        "adapter_params": {"dataset": "FakeVisionDataset", "download": False},
        "transforms": {
            "train": [{"name": "ToTensor"}, {"name": "Normalize", "mean": "auto", "std": "auto"}],
            "eval": [{"name": "ToTensor"}, {"name": "Normalize", "mean": "auto", "std": "auto"}],
        },
    }
    payload.update(overrides)
    return DataConfig.model_validate(payload)


def test_torchvision_adapter_wraps_dataset_and_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(torchvision.datasets, "FakeVisionDataset", FakeVisionDataset, raising=False)
    bundle = build_dataset(_torchvision_config(tmp_path))
    assert bundle.class_names == ("bird", "cat", "dog") or bundle.class_names == (
        "cat",
        "dog",
        "bird",
    )
    assert len(bundle.test) == 3
    assert len(bundle.train) + len(bundle.val) == 10
    image, label, index = bundle.train[0]
    assert image.shape == (3, 8, 8)
    assert isinstance(label, int)
    assert isinstance(index, int)
    assert bundle.model_contract.input_shape == (3, 8, 8)


def test_torchvision_adapter_random_and_labels_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(torchvision.datasets, "FakeLabelsDataset", FakeLabelsDataset, raising=False)
    config = _torchvision_config(
        tmp_path,
        num_classes=2,
        class_names=("a", "b"),
        split={"mode": "random", "val_fraction": 0.25},
        adapter_params={"dataset": "FakeLabelsDataset", "download": False},
    )
    bundle = build_dataset(config)
    assert bundle.class_names == ("a", "b")
    assert len(bundle.train) + len(bundle.val) == 4


def test_torchvision_adapter_rejects_unknown_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(torchvision.datasets, "FakeVisionDataset", FakeVisionDataset, raising=False)
    with pytest.raises(ConfigError, match="unknown torchvision"):
        build_dataset(_torchvision_config(tmp_path, adapter_params={"dataset": "Nope"}))


def test_torchvision_adapter_rejects_class_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(torchvision.datasets, "FakeVisionDataset", FakeVisionDataset, raising=False)
    with pytest.raises(ConfigError, match="class order"):
        build_dataset(_torchvision_config(tmp_path, class_names=("dog", "cat", "bird")))


def test_torchvision_adapter_rejects_unsupported_split(tmp_path, monkeypatch):
    monkeypatch.setattr(torchvision.datasets, "FakeVisionDataset", FakeVisionDataset, raising=False)
    with pytest.raises(ConfigError, match="supports random"):
        build_dataset(_torchvision_config(tmp_path, split={"mode": "group"}))


def test_torchvision_adapter_rejects_missing_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(torchvision.datasets, "NoTargetsDataset", NoTargetsDataset, raising=False)
    with pytest.raises(DataError, match="neither targets nor labels"):
        build_dataset(
            _torchvision_config(
                tmp_path, adapter_params={"dataset": "NoTargetsDataset", "download": False}
            )
        )


def test_torchvision_adapter_wraps_dataset_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(torchvision.datasets, "BrokenDataset", BrokenDataset, raising=False)
    with pytest.raises(DataError, match="cannot load"):
        build_dataset(
            _torchvision_config(
                tmp_path, adapter_params={"dataset": "BrokenDataset", "download": False}
            )
        )


def test_torchvision_adapter_requires_root(tmp_path):
    with pytest.raises(ConfigError, match=r"requires data\.root"):
        build_dataset(DataConfig(adapter="torchvision", root=None, num_classes=2))


def _numpy_config(tmp_path, **overrides):
    payload = {
        "adapter": "numpy",
        "root": tmp_path,
        "num_classes": 3,
        "channels": "gray",
        "image_size": (8, 8),
        "split": {"mode": "random_stratified", "val_fraction": 0.25},
        "transforms": {
            "train": [{"name": "ToTensor"}],
            "eval": [{"name": "ToTensor"}],
        },
        "adapter_params": {"array_path": "data.npz"},
    }
    payload.update(overrides)
    return DataConfig.model_validate(payload)


def test_numpy_adapter_reads_npz_and_scales_images(tmp_path):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (12, 1, 8, 8), dtype=np.uint8)
    labels = np.array([0, 1, 2] * 4)
    np.savez(tmp_path / "data.npz", images=images, labels=labels)
    bundle = build_dataset(_numpy_config(tmp_path))
    assert bundle.class_names == ("class_0", "class_1", "class_2")
    image, label, index = bundle.train[0]
    assert image.shape == (1, 8, 8)
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert isinstance(label, int)
    assert isinstance(index, int)


def test_numpy_adapter_reads_separate_npy_files_and_expands_channels(tmp_path):
    rng = np.random.default_rng(1)
    np.save(tmp_path / "images.npy", rng.random((10, 8, 8)).astype(np.float32))
    np.save(tmp_path / "labels.npy", np.arange(10) % 3)
    bundle = build_dataset(
        _numpy_config(
            tmp_path,
            adapter_params={"array_path": "images.npy", "labels_path": "labels.npy"},
        )
    )
    assert len(bundle.train) + len(bundle.val) == 10
    image, _, _ = bundle.train[0]
    assert image.shape == (1, 8, 8)


def test_numpy_adapter_requires_labels_for_npy(tmp_path):
    np.save(tmp_path / "images.npy", np.zeros((4, 8, 8), dtype=np.float32))
    with pytest.raises(ConfigError, match="labels_path"):
        build_dataset(_numpy_config(tmp_path, adapter_params={"array_path": "images.npy"}))


def test_numpy_adapter_rejects_bad_shapes(tmp_path):
    np.savez(tmp_path / "data.npz", images=np.zeros((4, 8), dtype=np.float32), labels=[0, 1, 0, 1])
    with pytest.raises(DataError, match="must be"):
        build_dataset(_numpy_config(tmp_path))


def test_numpy_adapter_rejects_mismatched_lengths(tmp_path):
    np.savez(tmp_path / "data.npz", images=np.zeros((4, 1, 8, 8), dtype=np.float32), labels=[0])
    with pytest.raises(DataError, match="different lengths"):
        build_dataset(_numpy_config(tmp_path))


def test_numpy_adapter_requires_array_path(tmp_path):
    with pytest.raises(ConfigError, match="array_path"):
        build_dataset(_numpy_config(tmp_path, adapter_params={}))


def test_numpy_adapter_wraps_load_errors(tmp_path):
    with pytest.raises(DataError, match="cannot load NumPy"):
        build_dataset(_numpy_config(tmp_path, adapter_params={"array_path": "missing.npz"}))


def test_numpy_adapter_runs_through_a_model(tmp_path):
    np.savez(
        tmp_path / "data.npz",
        images=np.random.default_rng(2).random((6, 1, 8, 8)).astype(np.float32),
        labels=np.arange(6) % 3,
    )
    bundle = build_dataset(_numpy_config(tmp_path))
    batch = torch.stack([bundle.train[0][0], bundle.train[1][0]])
    assert batch.shape == (2, 1, 8, 8)
    assert batch.dtype == torch.float32
