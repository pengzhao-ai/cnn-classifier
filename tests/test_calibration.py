import numpy as np
import pytest

from cnn_classifier.evaluation.calibration import expected_calibration_error


def test_perfect_confident_predictions_have_zero_ece():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = np.array([0, 1])
    report = expected_calibration_error(probs, y_true, n_bins=5)
    assert report.ece == pytest.approx(0.0)
    assert report.mce == pytest.approx(0.0)
    assert report.brier == pytest.approx(0.0)
    assert sum(bin.count for bin in report.bins) == 2


def test_overconfident_wrong_predictions_have_positive_ece():
    probs = np.array([[0.95, 0.05], [0.95, 0.05]])
    y_true = np.array([1, 1])
    report = expected_calibration_error(probs, y_true, n_bins=5)
    assert report.ece > 0.5
    assert report.brier > 0.5
