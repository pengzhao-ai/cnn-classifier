import pytest

from cnn_classifier.core.errors import DataError
from cnn_classifier.core.types import ModelInputContract, SplitInfo
from cnn_classifier.data.manifest import Manifest, fingerprint_records


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
