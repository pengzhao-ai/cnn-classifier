from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cnn_classifier.config.loader import apply_overrides, load_config
from cnn_classifier.core.errors import ConfigError


def test_missing_config_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_non_mapping_config_is_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_defaults_cycle_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("defaults: [b.yaml]\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("defaults: [a.yaml]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cyclic"):
        load_config(tmp_path / "a.yaml")


def test_defaults_must_be_a_list(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("defaults: base.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="list"):
        load_config(path)


def test_default_entries_must_be_strings(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("defaults: [1]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="strings"):
        load_config(path)


def test_overrides_require_key_value_syntax(minimal_config_dict):
    with pytest.raises(ConfigError, match="key=value"):
        apply_overrides(minimal_config_dict, ["training.epochs"])


def test_invalid_override_value_is_rejected(minimal_config_dict):
    with pytest.raises(ConfigError, match="invalid override"):
        apply_overrides(minimal_config_dict, ["training.epochs=[unclosed"])


def test_invalid_schema_is_reported_as_config_error(tmp_path, minimal_config_dict):
    payload = dict(minimal_config_dict)
    payload["unknown_section"] = {}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_config(path)


def test_override_replaces_nested_scalar(write_config, minimal_config_dict):
    path = write_config(minimal_config_dict)
    config = load_config(path, ["training.optimizer.lr=0.02"])
    assert config.training.optimizer.lr == pytest.approx(0.02)


def test_absolute_paths_are_preserved(tmp_path, write_config, minimal_config_dict):
    root = tmp_path / "images"
    payload = dict(minimal_config_dict)
    payload["data"] = {**payload["data"], "root": str(root)}
    path = write_config(payload)
    config = load_config(Path(path))
    assert config.data.root == root
