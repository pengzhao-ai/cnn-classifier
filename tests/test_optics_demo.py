from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "demos" / "optics_degradation" / "generate_dataset.py"

EXPECTED_LABELS = {
    "clean",
    "defocus",
    "motion_blur",
    "chromatic_aberration",
    "vignetting",
    "sensor_noise",
    "jpeg",
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("optics_generate_dataset", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generator_is_deterministic_and_group_safe(tmp_path):
    module = _load_generator()

    run_a = module.generate_fixture(tmp_path / "a", seed=1)
    run_b = module.generate_fixture(tmp_path / "b", seed=1)

    assert run_a.manifest_digest == run_b.manifest_digest
    assert run_a.group_leakage == []
    assert set(run_a.labels) == EXPECTED_LABELS


def test_manifest_columns_images_and_group_split(tmp_path):
    module = _load_generator()
    root = tmp_path / "data"
    run = module.generate_fixture(root, seed=3)

    manifest = root / "manifest.csv"
    assert manifest.is_file()
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert set(rows[0]) == {"path", "label", "split", "group", "severity"}
    assert all((root / row["path"]).is_file() for row in rows)
    assert {row["label"] for row in rows} == EXPECTED_LABELS
    assert all(0.0 <= float(row["severity"]) <= 1.0 for row in rows)
    assert set(run.labels) == EXPECTED_LABELS
    assert run.record_count == len(rows)

    groups_by_split: dict[str, set[str]] = {}
    for row in rows:
        groups_by_split.setdefault(row["split"], set()).add(row["group"])
    splits = list(groups_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            assert not (groups_by_split[left] & groups_by_split[right])


def test_generator_seed_changes_the_manifest(tmp_path):
    module = _load_generator()

    run_a = module.generate_fixture(tmp_path / "a", seed=1)
    run_b = module.generate_fixture(tmp_path / "b", seed=2)

    assert run_a.manifest_digest != run_b.manifest_digest
