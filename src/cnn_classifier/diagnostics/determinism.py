"""Determinism check for a seeded callable."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from cnn_classifier.core.seeding import seed_everything


def _equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return bool(torch.equal(torch.as_tensor(left), torch.as_tensor(right)))
    return bool(left == right)


def determinism_check(train_fn: Callable[[], Any], seed: int) -> bool:
    """Run a callable twice with the same seed and compare its result."""
    seed_everything(seed)
    first = train_fn()
    seed_everything(seed)
    second = train_fn()
    return _equal(first, second)
