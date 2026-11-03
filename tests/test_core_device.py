import pytest
import torch

from cnn_classifier.core.device import device_report, resolve_device
from cnn_classifier.core.errors import ConfigError


def test_auto_prefers_mps_when_cuda_is_absent(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_explicit_cpu_is_honored():
    assert resolve_device("cpu").type == "cpu"


def test_explicit_unsupported_device_fails():
    with pytest.raises(ConfigError):
        resolve_device("not-a-device")


def test_device_report_contains_expected_keys():
    report = device_report()
    assert {"device", "cuda_available", "mps_available", "torch_version"} <= set(report)
