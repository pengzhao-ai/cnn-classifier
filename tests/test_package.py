import cnn_classifier


def test_import_and_version():
    assert cnn_classifier.__version__ == "0.1.0"
    assert hasattr(cnn_classifier, "__all__")
