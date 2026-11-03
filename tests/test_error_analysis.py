import numpy as np

from cnn_classifier.evaluation.error_analysis import confusion_pairs, most_confident_errors


def test_most_confident_errors_are_ranked():
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    y_true = np.array([1, 0, 1])
    errors = most_confident_errors(probabilities, y_true, ["a", "b"], max_examples=2)
    assert len(errors) == 1
    assert errors[0].index == 0
    assert errors[0].predicted == "a"
    assert errors[0].true == "b"


def test_confusion_pairs_excludes_diagonal():
    y_true = np.array([0, 1, 1, 2])
    y_pred = np.array([0, 2, 2, 2])
    pairs = confusion_pairs(y_true, y_pred, ["a", "b", "c"])
    assert pairs[0] == ("b", "c", 2)
    assert all(true != predicted for true, predicted, _ in pairs)
