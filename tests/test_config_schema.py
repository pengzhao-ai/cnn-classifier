import pytest
from pydantic import ValidationError

from cnn_classifier.config.schema import RunConfig, TransformConfig


def test_unknown_key_is_rejected(minimal_config_dict):
    minimal_config_dict["trainig"] = {}
    with pytest.raises(ValidationError, match="trainig"):
        RunConfig.model_validate(minimal_config_dict)


def test_transform_config_keeps_params():
    transform = TransformConfig.model_validate({"name": "RandomHorizontalFlip", "p": 0.5})
    assert transform.params() == {"p": 0.5}
