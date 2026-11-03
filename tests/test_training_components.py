import pytest
import torch
from torch import nn

from cnn_classifier.config.schema import LossConfig, SchedulerConfig, TrainingConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.training.amp import resolve_amp
from cnn_classifier.training.losses import build_loss, mixup_cutmix_batch
from cnn_classifier.training.optim import build_optimizer
from cnn_classifier.training.schedulers import build_scheduler


def test_cosine_scheduler_reaches_min_lr():
    model = nn.Linear(3, 2)
    opt = build_optimizer(
        model,
        TrainingConfig(
            epochs=10,
            batch_size=2,
            optimizer={"name": "sgd", "lr": 1.0},
            scheduler={"name": "cosine", "min_lr_ratio": 0.0},
            loss={"name": "cross_entropy"},
        ),
    )
    sched = build_scheduler(opt, SchedulerConfig(name="cosine", min_lr_ratio=0.0), epochs=10)
    for _ in range(10):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-6)


def test_warmup_scheduler_starts_low_and_reaches_cosine():
    model = nn.Linear(3, 2)
    opt = build_optimizer(
        model,
        TrainingConfig(
            epochs=10,
            batch_size=2,
            optimizer={"name": "sgd", "lr": 1.0},
            scheduler={"name": "cosine_warmup", "warmup_epochs": 2},
            loss={"name": "cross_entropy"},
        ),
    )
    sched = build_scheduler(opt, SchedulerConfig(name="cosine_warmup", warmup_epochs=2), epochs=10)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5)
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0)
    for _ in range(9):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] < 0.5


def test_fp16_amp_is_rejected_on_cpu():
    with pytest.raises(ConfigError):
        resolve_amp(torch.device("cpu"), "fp16")


def test_auto_amp_is_off_on_cpu():
    policy = resolve_amp(torch.device("cpu"), "auto")
    assert policy.enabled is False


def test_loss_uses_label_smoothing():
    loss = build_loss(LossConfig(label_smoothing=0.1))
    assert isinstance(loss, nn.CrossEntropyLoss)
    assert loss.label_smoothing == pytest.approx(0.1)


def test_mixup_returns_convex_batch():
    images = torch.zeros(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    mixed = mixup_cutmix_batch(images, labels, 0.2, 0.0, torch.Generator().manual_seed(1))
    assert mixed.images.shape == images.shape
    assert mixed.labels_a.shape == labels.shape
    assert mixed.labels_b.shape == labels.shape
    assert 0.0 <= mixed.lam <= 1.0


def test_cutmix_changes_a_region():
    generator = torch.Generator().manual_seed(4)
    images = torch.rand(4, 3, 16, 16, generator=generator)
    labels = torch.tensor([0, 1, 0, 1])
    mixed = mixup_cutmix_batch(images, labels, 0.0, 1.0, torch.Generator().manual_seed(2))
    assert mixed.images.shape == images.shape
    assert mixed.lam < 1.0
