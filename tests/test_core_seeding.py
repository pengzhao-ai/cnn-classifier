import random

import numpy as np
import pytest
import torch

from cnn_classifier.core.hashing import stable_hash
from cnn_classifier.core.seeding import make_generator, seed_everything


def test_seed_everything_repeats_python_numpy_and_torch():
    seed_everything(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == pytest.approx(second)


def test_make_generator_is_repeatable():
    a = make_generator(7)
    b = make_generator(7)
    assert torch.rand(4, generator=a).tolist() == torch.rand(4, generator=b).tolist()


def test_stable_hash_is_key_order_independent():
    assert stable_hash({"b": 1, "a": 2}) == stable_hash({"a": 2, "b": 1})
    assert stable_hash({"a": 2}) != stable_hash({"a": 3})
