import numpy as np
import pytest

from cnn_classifier.evaluation.bootstrap import (
    bootstrap_ci,
    paired_bootstrap_difference,
)
from cnn_classifier.evaluation.metrics import accuracy


def test_bootstrap_is_deterministic():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    a = bootstrap_ci(y, p, accuracy, n_resamples=100, seed=7)
    b = bootstrap_ci(y, p, accuracy, n_resamples=100, seed=7)
    assert a == b
    assert a.low <= a.estimate <= a.high


def test_paired_bootstrap_detects_identical_predictions():
    y = np.array([0, 1, 0, 1])
    p = np.array([0, 1, 0, 1])
    interval = paired_bootstrap_difference(y, p, p, accuracy, n_resamples=50, seed=3)
    assert interval.estimate == pytest.approx(0.0)
    assert interval.low == pytest.approx(0.0)
    assert interval.high == pytest.approx(0.0)
