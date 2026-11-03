from __future__ import annotations

import pytest
import torch
from torch import nn

from cnn_classifier.config.schema import SchedulerConfig, TrainingConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.training.amp import autocast_context, resolve_amp
from cnn_classifier.training.optim import build_optimizer
from cnn_classifier.training.schedulers import build_scheduler


def _optimizer() -> torch.optim.Optimizer:
    return build_optimizer(
        nn.Linear(2, 2),
        TrainingConfig(epochs=5, batch_size=2, optimizer={"name": "sgd", "lr": 1.0}),
    )


@pytest.mark.parametrize(
    "config",
    [
        SchedulerConfig(name="constant"),
        SchedulerConfig(name="cosine_warmup", warmup_epochs=0),
        SchedulerConfig(name="step", step_size=2, gamma=0.5),
        SchedulerConfig(name="plateau"),
    ],
)
def test_all_scheduler_variants_build(config):
    scheduler = build_scheduler(_optimizer(), config, epochs=5)
    assert scheduler is not None


def test_unknown_scheduler_is_rejected():
    config = SchedulerConfig.model_construct(name="unknown")
    with pytest.raises(ValueError, match="unknown scheduler"):
        build_scheduler(_optimizer(), config, epochs=5)


def test_amp_policy_variants_and_errors():
    assert resolve_amp(torch.device("cpu"), "off").enabled is False
    assert resolve_amp(torch.device("cpu"), "auto").enabled is False
    bf16 = resolve_amp(torch.device("cpu"), "bf16")
    assert bf16.enabled is True
    assert bf16.dtype == torch.bfloat16
    with pytest.raises(ConfigError, match="MPS"):
        resolve_amp(torch.device("mps"), "bf16")
    with pytest.raises(ConfigError, match="unknown AMP"):
        resolve_amp(torch.device("cpu"), "bogus")  # type: ignore[arg-type]


def test_autocast_context_runs_enabled_and_disabled():
    disabled = resolve_amp(torch.device("cpu"), "off")
    with autocast_context(disabled):
        assert torch.ones(1).dtype == torch.float32
    enabled = resolve_amp(torch.device("cpu"), "bf16")
    with autocast_context(enabled):
        assert torch.ones(1).dtype == torch.float32
