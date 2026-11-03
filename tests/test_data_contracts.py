import pytest
import torch

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.core.errors import DataError
from cnn_classifier.core.types import ModelInputContract, SplitInfo
from cnn_classifier.data.adapters._common import RawSample, build_bundle
from cnn_classifier.data.manifest import Manifest, fingerprint_records
from cnn_classifier.data.splits import SplitIndices


def test_split_info_rejects_overlap():
    with pytest.raises(DataError, match="overlap"):
        SplitInfo(
            mode="random",
            train_indices=(0, 1),
            val_indices=(1, 2),
            test_indices=None,
            seed=1,
            group_column=None,
            fingerprint="abc",
        )


def test_model_contract_and_bundle_are_frozen():
    contract = ModelInputContract(
        input_shape=(3, 8, 8),
        dtype="float32",
        value_range=(0.0, 1.0),
        normalization=None,
        class_names=("a", "b"),
        label_map={"a": 0, "b": 1},
        channel_policy="rgb",
        stem_adaptation=None,
    )
    with pytest.raises(AttributeError):
        contract.input_shape = (1, 8, 8)


def test_fingerprint_records_is_order_independent(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    csv = tmp_path / "manifest.csv"
    csv.write_text(
        "path,label,split,group\na.png,cat,train,g1\nb.png,dog,val,g2\n",
        encoding="utf-8",
    )
    first = Manifest.from_csv(
        csv,
        tmp_path,
        columns={"path": "path", "label": "label", "split": "split", "group": "group"},
    )
    second = Manifest(records=tuple(reversed(first.records)))
    assert fingerprint_records(first.records) == fingerprint_records(second.records)


def test_build_bundle_rejects_group_leakage():
    samples = [
        RawSample(torch.zeros(1, 8, 8), "a", "train", "g1"),
        RawSample(torch.zeros(1, 8, 8), "b", "train", "g2"),
        RawSample(torch.zeros(1, 8, 8), "a", "val", "g1"),
        RawSample(torch.zeros(1, 8, 8), "b", "val", "g2"),
    ]
    config = DataConfig(
        adapter="synthetic",
        class_names=("a", "b"),
        num_classes=2,
        channels="gray",
        image_size=(8, 8),
        normalization={"mode": "none"},
    )
    split = SplitIndices(train_indices=(0, 1), val_indices=(2, 3))

    with pytest.raises(DataError, match="group"):
        build_bundle(
            samples,
            config,
            class_names=("a", "b"),
            split=split,
            mode="explicit",
            fingerprint="test",
        )
