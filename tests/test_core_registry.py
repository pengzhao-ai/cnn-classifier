import pytest

from cnn_classifier.core.errors import RegistryError
from cnn_classifier.core.registry import Registry


def test_registry_roundtrip_and_duplicate():
    reg = Registry[int]("thing")

    @reg.register("answer")
    def answer() -> int:
        return 42

    assert reg.get("answer")() == 42
    assert reg.names() == ("answer",)
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("answer")(lambda: 0)


def test_registry_rejects_empty_and_unknown_names():
    reg = Registry[int]("thing")
    with pytest.raises(RegistryError, match="empty"):
        reg.register("")(lambda: 0)
    with pytest.raises(RegistryError, match="not registered"):
        reg.get("missing")
