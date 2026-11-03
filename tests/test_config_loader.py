import pytest

from cnn_classifier.config.loader import apply_overrides, load_config
from cnn_classifier.core.errors import ConfigError


def test_load_config_applies_defaults_and_override(write_config, minimal_config_dict):
    path = write_config(minimal_config_dict)
    cfg = load_config(path, overrides=["training.epochs=3"])
    assert cfg.experiment.name == "test"
    assert cfg.training.epochs == 3
    assert cfg.runtime.device == "auto"
    assert cfg.tracking.csv is True


def test_defaults_chain_and_path_resolution(write_config, minimal_config_dict):
    base = dict(minimal_config_dict)
    base["training"] = {"epochs": 10, "batch_size": 4}
    write_config(base, "base.yaml")

    experiment = dict(minimal_config_dict)
    experiment["defaults"] = ["base.yaml"]
    experiment["data"] = {**experiment["data"], "root": "data"}
    path = write_config(experiment, "exp.yaml")

    cfg = load_config(path)
    assert cfg.training.epochs == 2
    assert cfg.data.root == path.parent / "data"


def test_unknown_override_path_fails(minimal_config_dict):
    with pytest.raises(ConfigError, match="unknown override"):
        apply_overrides(minimal_config_dict, ["training.nope=1"])


def test_warmup_must_be_shorter_than_epochs(write_config, minimal_config_dict):
    minimal_config_dict["training"] = {
        "epochs": 2,
        "batch_size": 2,
        "scheduler": {"name": "cosine_warmup", "warmup_epochs": 2},
    }
    with pytest.raises(ConfigError, match="warmup"):
        load_config(write_config(minimal_config_dict))


def test_class_names_must_match_num_classes(write_config, minimal_config_dict):
    minimal_config_dict["data"] = {**minimal_config_dict["data"], "num_classes": 3}
    with pytest.raises(ConfigError, match="class_names"):
        load_config(write_config(minimal_config_dict))


def test_fp16_is_rejected_on_non_cuda_device(write_config, minimal_config_dict):
    minimal_config_dict["runtime"] = {"device": "mps"}
    minimal_config_dict["training"] = {**minimal_config_dict["training"], "amp": "fp16"}
    with pytest.raises(ConfigError, match="fp16"):
        load_config(write_config(minimal_config_dict))


def test_supplied_normalization_requires_stats(write_config, minimal_config_dict):
    minimal_config_dict["data"] = {
        **minimal_config_dict["data"],
        "normalization": {"mode": "supplied"},
    }
    with pytest.raises(ConfigError, match="mean"):
        load_config(write_config(minimal_config_dict))


def test_invalid_val_fraction_is_rejected(write_config, minimal_config_dict):
    minimal_config_dict["data"] = {
        **minimal_config_dict["data"],
        "split": {"mode": "random", "val_fraction": 1.0},
    }
    with pytest.raises(ConfigError, match="val_fraction"):
        load_config(write_config(minimal_config_dict))
