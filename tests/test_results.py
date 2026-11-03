from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DOCS_IMAGES = REPO_ROOT / "docs" / "images"

TRANSFER_RUNS = ("scratch", "linear_probe", "finetune")
ABLATION_RUNS = (
    "augmentation",
    "none_augmentation",
    "batchnorm",
    "no_batchnorm",
    "residual",
    "no_residual",
    "cosine",
    "constant",
)


def _summary(*parts: str) -> dict:
    path = RESULTS.joinpath(*parts, "reports", "summary.json")
    assert path.is_file(), path
    return json.loads(path.read_text(encoding="utf-8"))


def test_cifar_results_are_committed():
    summary = _summary("cifar10")
    assert summary["test_accuracy"] == pytest.approx(0.9225, abs=0.002)
    assert summary["device"] == "mps"
    assert summary["test_fingerprint"]
    for name in (
        "training_history.png",
        "confusion_matrix.png",
        "per_class.png",
        "calibration.png",
        "model_card.png",
        "errors.png",
    ):
        assert (RESULTS / "cifar10" / "figures" / name).is_file(), name
    for name in ("report.html", "report.md", "evaluation.json"):
        assert (RESULTS / "cifar10" / "reports" / name).is_file(), name


def test_transfer_results_are_committed_and_fingerprinted():
    fingerprints = set()
    for name in TRANSFER_RUNS:
        summary = _summary("transfer", name)
        assert summary["test_accuracy"] is not None
        assert summary["test_accuracy"] > 0.5
        assert summary["test_fingerprint"]
        fingerprints.add(summary["test_fingerprint"])
    assert len(fingerprints) == 1
    assert _summary("transfer", "linear_probe")["trainable_params"] == 5_130

    comparison = RESULTS / "comparison" / "transfer_comparison.json"
    assert comparison.is_file()
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 3
    for suffix in ("md", "png"):
        assert (RESULTS / "comparison" / f"transfer_comparison.{suffix}").is_file()


def test_ablation_results_are_committed():
    for name in ABLATION_RUNS:
        summary = _summary("ablations", name)
        assert summary["test_accuracy"] is not None
        assert summary["test_accuracy"] > 0.2
        assert (RESULTS / "ablations" / name / "reports" / "report.html").is_file()

    comparison = RESULTS / "comparison" / "ablation_comparison.json"
    assert comparison.is_file()
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 8
    for suffix in ("md", "png"):
        assert (RESULTS / "comparison" / f"ablation_comparison.{suffix}").is_file()


def test_paired_bootstrap_comparisons_are_committed():
    transfer = RESULTS / "comparison" / "transfer_paired.json"
    ablation = RESULTS / "comparison" / "ablation_paired.json"
    assert transfer.is_file()
    assert ablation.is_file()
    transfer_rows = json.loads(transfer.read_text(encoding="utf-8"))["rows"]
    ablation_rows = json.loads(ablation.read_text(encoding="utf-8"))["rows"]
    assert len(transfer_rows) == 3
    assert len(ablation_rows) == 4
    for row in [*transfer_rows, *ablation_rows]:
        assert row["ci_low"] <= row["difference"] <= row["ci_high"]


def test_optics_and_byo_results_are_committed():
    optics = _summary("optics_degradation")
    assert optics["test_accuracy"] is not None
    assert optics["test_accuracy"] > 0.3
    assert (RESULTS / "optics_degradation" / "figures" / "confusion_matrix.png").is_file()

    byo = _summary("bring_your_own_data")
    assert byo["val_accuracy"] is not None
    assert byo["val_accuracy"] > 0.3
    assert (RESULTS / "bring_your_own_data" / "reports" / "report.html").is_file()
    contract = RESULTS / "bring_your_own_data" / "model_contract.json"
    assert contract.is_file()
    contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    assert contract_payload["input_shape"] == [1, 16, 16]


def test_docs_images_are_byte_identical_to_results():
    cifar = RESULTS / "cifar10" / "figures"
    optics = RESULTS / "optics_degradation" / "figures"
    pairs = (
        ("cifar10_training_history.png", cifar / "training_history.png"),
        ("cifar10_confusion_matrix.png", cifar / "confusion_matrix.png"),
        ("cifar10_errors.png", cifar / "errors.png"),
        ("cifar10_model_card.png", cifar / "model_card.png"),
        (
            "transfer_comparison.png",
            RESULTS / "comparison" / "transfer_comparison.png",
        ),
        (
            "ablation_comparison.png",
            RESULTS / "comparison" / "ablation_comparison.png",
        ),
        (
            "transfer_finetune_training_history.png",
            RESULTS / "transfer" / "finetune" / "figures" / "training_history.png",
        ),
        ("optics_confusion_matrix.png", optics / "confusion_matrix.png"),
        ("optics_per_class.png", optics / "per_class.png"),
    )
    for docs_name, result_path in pairs:
        docs_path = DOCS_IMAGES / docs_name
        assert docs_path.is_file(), docs_path
        assert result_path.is_file(), result_path
        assert docs_path.read_bytes() == result_path.read_bytes(), docs_name


def test_no_checkpoints_are_committed_under_results():
    assert not list(RESULTS.rglob("*.pt"))
    assert not list(RESULTS.rglob("*.pth"))


def test_committed_reports_are_portable():
    for name in ("report.md", "report.html"):
        for path in RESULTS.rglob(name):
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "/Users/" not in text, path
            assert "--format html" not in text, path
