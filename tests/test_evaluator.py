from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cnn_classifier.config.schema import (
    BootstrapConfig,
    EvaluationConfig,
    LatencyConfig,
    TTAConfig,
)
from cnn_classifier.evaluation.evaluator import evaluate_model


class TinyDataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(self) -> None:
        self.images = torch.tensor([[[[0.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]]])
        self.labels = [0, 0, 1, 1]

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index], index


def make_model() -> nn.Module:
    model = nn.Sequential(nn.Flatten(), nn.Linear(1, 2))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[-1.0], [1.0]]))
        model[1].bias.zero_()
    return model


def make_config(**kwargs) -> EvaluationConfig:
    return EvaluationConfig(
        bootstrap=BootstrapConfig(enabled=False),
        latency=LatencyConfig(enabled=False),
        **kwargs,
    )


def test_evaluate_model_produces_metrics_and_predictions():
    loader = DataLoader(TinyDataset(), batch_size=2)
    result = evaluate_model(make_model(), loader, torch.device("cpu"), ["a", "b"], make_config())
    assert result.metrics.accuracy == pytest.approx(1.0)
    assert result.probabilities.shape == (4, 2)
    assert result.y_true.tolist() == [0, 0, 1, 1]
    assert result.base_metrics is not None


def test_tta_stores_base_and_tta_metrics():
    loader = DataLoader(TinyDataset(), batch_size=2)
    config = make_config(tta=TTAConfig(enabled=True))
    result = evaluate_model(make_model(), loader, torch.device("cpu"), ["a", "b"], config)
    assert result.base_metrics is not None
    assert result.tta_metrics is not None
    assert result.metrics.accuracy == pytest.approx(1.0)
