from __future__ import annotations

import torch

from cnn_classifier.config.schema import RunConfig
from cnn_classifier.data.adapters import build_dataset
from cnn_classifier.data.loaders import build_loaders
from cnn_classifier.models import build_model
from cnn_classifier.runs.context import RunContext
from cnn_classifier.training.trainer import Trainer


def _trainer(tmp_path, training: dict, *, transfer: dict | None = None):
    payload = {
        "schema_version": "1.0",
        "experiment": {"name": "trainer-options", "seed": 5},
        "data": {
            "adapter": "synthetic",
            "class_names": ["a", "b"],
            "num_classes": 2,
            "channels": "gray",
            "image_size": [8, 8],
            "adapter_params": {"num_samples": 8, "num_classes": 2, "image_size": [8, 8]},
        },
        "model": {
            "name": "simple_cnn",
            "params": {"channels": [4], "dropout": 0.0},
            "transfer": transfer or {},
        },
        "training": {"epochs": 1, "batch_size": 2, **training},
        "evaluation": {"bootstrap": {"enabled": False}, "latency": {"enabled": False}},
        "tracking": {"console": False, "csv": True, "tensorboard": False},
    }
    config = RunConfig.model_validate(payload)
    bundle = build_dataset(config.data)
    loaders = build_loaders(bundle, config.training, config.runtime, torch.device("cpu"))
    model = build_model(
        config.model,
        num_classes=len(bundle.class_names),
        in_channels=bundle.in_channels,
        image_size=bundle.image_size,
    )
    run = RunContext.create(config, tmp_path / "runs")
    trainer = Trainer(model, loaders, config, run, torch.device("cpu"), model_contract=None)
    return trainer, run


def test_trainer_supports_channels_last_and_mixup_and_clipping(tmp_path):
    trainer, run = _trainer(
        tmp_path,
        {
            "mixup_alpha": 0.2,
            "cutmix_alpha": 0.2,
            "grad_clip_norm": 1.0,
            "scheduler": {"name": "cosine", "step_unit": "batch"},
        },
    )
    trainer.model = trainer.model.to(memory_format=torch.channels_last)
    result = trainer.fit()
    assert result.history
    assert (run.directory / "data" / "model_contract.json").is_file()


def test_trainer_supports_plateau_scheduler_and_ema(tmp_path):
    trainer, _ = _trainer(
        tmp_path,
        {
            "scheduler": {"name": "plateau"},
            "ema": {"enabled": True, "decay": 0.5},
        },
    )
    result = trainer.fit()
    assert result.history
    assert trainer.state.eval_model is not None


def test_trainer_supports_param_groups(tmp_path):
    trainer, _ = _trainer(
        tmp_path,
        {},
        transfer={"param_groups": [{"pattern": "^features\\.0", "lr_multiplier": 2.0}]},
    )
    assert len(trainer.optimizer.param_groups) >= 2
    trainer.fit()


def test_trainer_early_stopping_breaks_loop(tmp_path):
    trainer, _ = _trainer(
        tmp_path,
        {
            "epochs": 4,
            "early_stopping": {
                "enabled": True,
                "monitor": "val_loss",
                "mode": "min",
                "patience": 0,
            },
        },
    )
    result = trainer.fit()
    assert len(result.history) < 4
