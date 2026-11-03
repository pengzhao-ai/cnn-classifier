"""Package completed run artifacts into the committed results tree."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
RESULTS = REPO_ROOT / "results"
DOCS_IMAGES = REPO_ROOT / "docs" / "images"

FIGURE_NAMES = (
    "training_history.png",
    "confusion_matrix.png",
    "per_class.png",
    "calibration.png",
    "model_card.png",
    "errors.png",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_run(name: str, *, root: Path = RUNS, evaluated: str | None = None) -> Path:
    candidates: list[tuple[str, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        config_path = path / "config.resolved.yaml"
        summary_path = path / "summary.json"
        if not config_path.is_file() or not summary_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment", {}).get("name") != name:
            continue
        summary = _load(summary_path)
        if evaluated is not None and summary.get(evaluated) is None:
            continue
        candidates.append((path.name, path))
    if not candidates:
        raise SystemExit(f"no completed run found for {name!r}")
    return sorted(candidates)[-1][1]


def _copy_run(run_dir: Path, destination: Path, *, extra: tuple[str, ...] = ()) -> None:
    reports = destination / "reports"
    figures = destination / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "evaluation.json", "report.html", "report.md"):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, reports / name)
    for name in ("environment.json", "git.json", "model_summary.json"):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, reports / name)
    for name in FIGURE_NAMES:
        source = run_dir / "figures" / name
        if source.is_file():
            shutil.copy2(source, figures / name)
    for relative in extra:
        source = run_dir / relative
        if source.is_file():
            shutil.copy2(source, destination / Path(relative).name)
    print(f"packaged {run_dir.name} -> {destination.relative_to(REPO_ROOT)}")


def _copy_comparison(source_prefix: Path, destination_prefix: Path) -> None:
    for suffix in ("md", "json", "png"):
        source = source_prefix.with_suffix(f".{suffix}")
        if source.is_file():
            shutil.copy2(source, destination_prefix.with_suffix(f".{suffix}"))
    print(f"packaged comparison {source_prefix.name} -> {destination_prefix.name}")


def main() -> None:
    (RESULTS / "comparison").mkdir(parents=True, exist_ok=True)
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)

    cifar = _find_run("cifar10_scratch_baseline", evaluated="test_accuracy")
    if _load(cifar / "summary.json").get("epochs_run") != 50:
        raise SystemExit(f"scratch run is not the full 50-epoch run: {cifar}")
    _copy_run(cifar, RESULTS / "cifar10")

    transfer_names = {
        "scratch": "cifar10_resnet18_scratch",
        "linear_probe": "cifar10_resnet18_linear_probe",
        "finetune": "cifar10_resnet18_finetune",
    }
    for label, name in transfer_names.items():
        run_dir = _find_run(name, evaluated="test_accuracy")
        _copy_run(run_dir, RESULTS / "transfer" / label)

    ablation_names = (
        ("ablation_augmentation", "augmentation"),
        ("ablation_no_augmentation", "none_augmentation"),
        ("ablation_batchnorm", "batchnorm"),
        ("ablation_no_batchnorm", "no_batchnorm"),
        ("ablation_residual", "residual"),
        ("ablation_no_residual", "no_residual"),
        ("ablation_cosine", "cosine"),
        ("ablation_constant", "constant"),
    )
    for name, label in ablation_names:
        run_dir = _find_run(name, evaluated="test_accuracy")
        _copy_run(run_dir, RESULTS / "ablations" / label)

    optics = _find_run("optics_degradation_classifier", evaluated="test_accuracy")
    _copy_run(optics, RESULTS / "optics_degradation")

    byo = _find_run(
        "bring_your_own_data_quickstart",
        root=RUNS / "byo_quickstart",
        evaluated="val_accuracy",
    )
    _copy_run(byo, RESULTS / "bring_your_own_data", extra=("data/model_contract.json",))

    transfer_comparison = RESULTS / "comparison" / "comparison"
    if transfer_comparison.with_suffix(".md").is_file():
        _copy_comparison(
            transfer_comparison,
            RESULTS / "comparison" / "transfer_comparison",
        )
        for suffix in ("md", "json", "png"):
            transfer_comparison.with_suffix(f".{suffix}").unlink(missing_ok=True)

    ablation_comparison = RESULTS / "ablations" / "comparison"
    if ablation_comparison.with_suffix(".md").is_file():
        _copy_comparison(
            ablation_comparison,
            RESULTS / "comparison" / "ablation_comparison",
        )
        for suffix in ("md", "json", "png"):
            ablation_comparison.with_suffix(f".{suffix}").unlink(missing_ok=True)

    shutil.copy2(
        RESULTS / "cifar10" / "figures" / "training_history.png",
        DOCS_IMAGES / "cifar10_training_history.png",
    )
    shutil.copy2(
        RESULTS / "comparison" / "transfer_comparison.png",
        DOCS_IMAGES / "transfer_comparison.png",
    )
    shutil.copy2(
        RESULTS / "comparison" / "ablation_comparison.png",
        DOCS_IMAGES / "ablation_comparison.png",
    )
    shutil.copy2(
        RESULTS / "optics_degradation" / "figures" / "confusion_matrix.png",
        DOCS_IMAGES / "optics_confusion_matrix.png",
    )
    print("docs images copied")


if __name__ == "__main__":
    main()
