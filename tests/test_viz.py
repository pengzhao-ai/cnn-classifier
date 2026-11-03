import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from cnn_classifier.evaluation.calibration import expected_calibration_error
from cnn_classifier.viz import (
    plot_calibration,
    plot_confusion_matrix,
    plot_error_gallery,
    plot_feature_maps,
    plot_filters,
    plot_gradcam,
    plot_model_card,
    plot_model_comparison,
    plot_per_class_metrics,
    plot_training_history,
)
from cnn_classifier.viz.features import compute_gradcam


def test_training_history_writes_png_and_svg(tmp_path):
    history = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "val_loss": 1.1,
            "train_accuracy": 0.5,
            "val_accuracy": 0.4,
            "lr": 0.1,
        }
    ]
    path = plot_training_history(history, tmp_path / "history.png")
    assert path.exists() and path.stat().st_size > 0
    assert (tmp_path / "history.svg").exists()


def test_confusion_matrix_and_model_card_render(tmp_path):
    cm = [[5, 1], [2, 7]]
    assert plot_confusion_matrix(cm, ["a", "b"], tmp_path / "cm.png").exists()
    summary = {
        "total_params": 10,
        "trainable_params": 10,
        "input_shape": [3, 8, 8],
        "output_shape": [1, 2],
        "modules": [{"name": "model", "type": "Sequential", "params": 10}],
    }
    assert plot_model_card(summary, tmp_path / "card.png").exists()


def test_feature_visualizations_render(tmp_path, tiny_model):
    image = torch.rand(3, 8, 8)
    filters = tiny_model.features[0].weight.detach()
    assert plot_filters(filters, tmp_path / "filters.png").exists()
    features = tiny_model.features[:3](image.unsqueeze(0)).detach()[0]
    assert plot_feature_maps(features, tmp_path / "features.png").exists()
    cam = compute_gradcam(tiny_model, image, target_class=0)
    assert plot_gradcam(image, cam, tmp_path / "gradcam.png").exists()


def test_error_calibration_and_comparison_render(tmp_path):
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    report = expected_calibration_error(probabilities, np.array([0, 1, 1]))
    assert plot_calibration(report, tmp_path / "calibration.png").exists()
    images = torch.rand(3, 3, 8, 8)
    assert plot_error_gallery(images, [0, 1, 1], [1, 1, 0], tmp_path / "errors.png").exists()
    rows = [
        {"run": "a", "accuracy": 0.8, "params": 1000, "latency_ms": 1.0},
        {"run": "b", "accuracy": 0.9, "params": 2000, "latency_ms": 2.0},
    ]
    assert plot_model_comparison(rows, tmp_path / "comparison.png").exists()
    assert plot_per_class_metrics(
        {"a": type("M", (), {"f1": 0.8})(), "b": type("M", (), {"f1": 0.9})()},
        tmp_path / "per_class.png",
    ).exists()
