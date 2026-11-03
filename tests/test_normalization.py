import json

import pytest
import torch
from torch.utils.data import TensorDataset

from cnn_classifier.data.stats import (
    compute_normalization,
    load_normalization,
    normalization_cache_key,
    save_normalization,
)


def test_compute_normalization_known_values():
    images = torch.tensor([[[[0.0, 2.0]]], [[[0.0, 2.0]]]])
    dataset = TensorDataset(images, torch.zeros(2, dtype=torch.long))
    stats = compute_normalization(dataset, batch_size=1)
    assert stats.mean == (pytest.approx(1.0),)
    assert stats.std == (pytest.approx(1.0),)


def test_constant_channel_uses_unit_std():
    dataset = TensorDataset(torch.zeros(4, 1, 2, 2), torch.zeros(4, dtype=torch.long))
    stats = compute_normalization(dataset, batch_size=2)
    assert stats.mean == (0.0,)
    assert stats.std == (1.0,)


def test_normalization_roundtrip_json(tmp_path):
    dataset = TensorDataset(torch.ones(2, 3, 2, 2), torch.zeros(2, dtype=torch.long))
    stats = compute_normalization(dataset)
    path = tmp_path / "normalization.json"
    save_normalization(stats, path)
    loaded = load_normalization(path)
    assert loaded == stats
    assert json.loads(path.read_text())["source"] == "computed"


def test_cache_key_changes_with_policy():
    a = normalization_cache_key("abc", (8, 8), "rgb", {"mode": "auto"})
    b = normalization_cache_key("abc", (8, 8), "rgb", {"mode": "fixed"})
    assert a != b
