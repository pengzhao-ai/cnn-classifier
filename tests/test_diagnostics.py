import torch
from torch import nn

from cnn_classifier.diagnostics import (
    activation_report,
    determinism_check,
    gradient_report,
    overfit_batch,
    trace_shapes,
)


def test_overfit_batch_reduces_loss(tiny_model, tiny_batch):
    torch.manual_seed(0)
    result = overfit_batch(
        tiny_model,
        tiny_batch,
        nn.CrossEntropyLoss(),
        torch.optim.SGD(tiny_model.parameters(), lr=0.5),
        torch.device("cpu"),
        max_steps=20,
    )
    assert result.final_loss < result.initial_loss


def test_trace_shapes_reports_layers(tiny_model):
    layers = trace_shapes(tiny_model, (3, 8, 8))
    assert layers
    assert layers[0].output_shape[1] == 4


def test_gradient_report_after_backward(tiny_model, tiny_batch):
    images, labels = tiny_batch
    loss = nn.CrossEntropyLoss()(tiny_model(images), labels)
    loss.backward()
    report = gradient_report(tiny_model)
    assert report
    assert all(item.norm >= 0 for item in report)


def test_activation_report_records_means(tiny_model, tiny_batch):
    images, _ = tiny_batch
    report = activation_report(tiny_model, images)
    assert report
    assert all(item.mean >= 0 for item in report if item.mean is not None)


def test_determinism_check_passes_for_seeded_function():
    assert determinism_check(lambda: torch.rand(1).item(), seed=5) is True
