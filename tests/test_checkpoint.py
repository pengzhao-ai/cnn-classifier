import pytest
import torch
from torch import nn

from cnn_classifier.core.errors import RunError
from cnn_classifier.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_and_mismatch(tmp_path):
    model = nn.Linear(2, 2)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=1,
        global_step=2,
        best_metric=0.5,
        best_epoch=1,
        config_hash="abc",
        class_names=("a", "b"),
        normalization=None,
        model_contract={"input_shape": [2, 2, 2]},
    )
    payload = load_checkpoint(path)
    assert payload["epoch"] == 1
    assert payload["class_names"] == ("a", "b")
    with pytest.raises(RunError, match="config"):
        load_checkpoint(path, expected_config_hash="different")


def test_checkpoint_preserves_optimizer_state(tmp_path):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "best.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=1,
        global_step=1,
        best_metric=0.9,
        best_epoch=1,
        config_hash="abc",
        class_names=("a", "b"),
        normalization=None,
        model_contract=None,
    )
    payload = load_checkpoint(path)
    assert "param_groups" in payload["optimizer"]
