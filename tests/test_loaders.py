import platform

import torch

from cnn_classifier.data.loaders import build_loaders


def test_loader_worker_policy_on_mps(
    monkeypatch, synthetic_bundle, training_config, runtime_config
):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    loaders = build_loaders(synthetic_bundle, training_config, runtime_config, torch.device("mps"))
    assert loaders.train.num_workers == 0
    assert loaders.train.batch_size == 2


def test_loader_shuffles_train_only(synthetic_bundle, training_config, runtime_config):
    loaders = build_loaders(synthetic_bundle, training_config, runtime_config, torch.device("cpu"))
    batch = next(iter(loaders.train))
    images, labels, indices = batch
    assert images.shape == (2, 1, 8, 8)
    assert labels.shape == (2,)
    assert indices.shape == (2,)
