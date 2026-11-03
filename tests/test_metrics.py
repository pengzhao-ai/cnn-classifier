import numpy as np
import pytest

from cnn_classifier.evaluation.metrics import accuracy, classification_metrics


def test_accuracy_and_macro_f1_on_known_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    report = classification_metrics(y_true, y_pred, probs, ["a", "b"], top_k=2)
    assert report.accuracy == pytest.approx(0.75)
    assert report.per_class["a"].recall == pytest.approx(0.5)
    assert report.support == 4
    assert report.top_k_accuracy == pytest.approx(1.0)


def test_confusion_matrix_and_nll_are_reported():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 2, 2])
    probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.2, 0.7], [0.1, 0.1, 0.8]])
    report = classification_metrics(y_true, y_pred, probs, ["a", "b", "c"])
    assert report.confusion_matrix[0] == [1, 0, 0]
    assert report.confusion_matrix[2] == [0, 0, 1]
    assert report.nll is not None and report.nll > 0


def test_accuracy_function():
    assert accuracy(np.array([0, 1]), np.array([0, 0])) == pytest.approx(0.5)
