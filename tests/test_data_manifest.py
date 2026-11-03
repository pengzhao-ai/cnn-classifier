import pytest

from cnn_classifier.core.errors import DataError
from cnn_classifier.data.manifest import Manifest


def test_manifest_parses_paths_labels_and_groups(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"not read by this test")
    csv = tmp_path / "manifest.csv"
    csv.write_text("path,label,split,group\na.png,cat,train,g1\n", encoding="utf-8")
    manifest = Manifest.from_csv(
        csv,
        tmp_path,
        columns={"path": "path", "label": "label", "split": "split", "group": "group"},
    )
    assert manifest.records[0].label == "cat"
    assert manifest.records[0].group == "g1"
    assert manifest.class_names == ("cat",)


def test_manifest_rejects_missing_file(tmp_path):
    csv = tmp_path / "manifest.csv"
    csv.write_text("path,label,split\nmissing.png,cat,train\n", encoding="utf-8")
    with pytest.raises(DataError, match=r"missing\.png"):
        Manifest.from_csv(
            csv,
            tmp_path,
            columns={"path": "path", "label": "label", "split": "split"},
        )


def test_manifest_rejects_bad_split(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    csv = tmp_path / "manifest.csv"
    csv.write_text("path,label,split\na.png,cat,validation\n", encoding="utf-8")
    with pytest.raises(DataError, match="split"):
        Manifest.from_csv(
            csv,
            tmp_path,
            columns={"path": "path", "label": "label", "split": "split"},
        )
