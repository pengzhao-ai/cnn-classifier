"""Deterministic seeding helpers."""

from __future__ import annotations

import contextlib
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch, and request deterministic kernels.

    MPS does not guarantee bit-identical results across versions or devices even
    with a fixed seed; this function makes runs repeatable within one environment
    as far as the backend allows.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with contextlib.suppress(RuntimeError):
            # Some backends do not expose a deterministic implementation for
            # every op. The run metadata records the requested policy.
            torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded CPU ``torch.Generator``."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
