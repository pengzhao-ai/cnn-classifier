from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.data.transforms import build_transforms
from cnn_classifier.evaluation.predict import predict_paths


def test_predict_paths_returns_probabilities(tmp_path):
    paths = []
    for index, value in enumerate((0, 255)):
        path = tmp_path / f"{index}.png"
        Image.fromarray(np.full((1, 1), value, dtype=np.uint8)).save(path)
        paths.append(path)
    model = nn.Sequential(nn.Flatten(), nn.Linear(1, 2))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[-1.0], [1.0]]))
        model[1].bias.zero_()
    transform = build_transforms(
        DataConfig(adapter="synthetic", channels="gray", image_size=(1, 1)),
        None,
        train=False,
    )
    predictions = predict_paths(model, paths, transform, "cpu", ["zero", "one"])
    assert predictions.probabilities.shape == (2, 2)
    assert predictions.class_names == ("zero", "one")
    assert predictions.predicted_labels.tolist() == [0, 1]
