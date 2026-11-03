"""Compute paired bootstrap differences from saved test predictions.

Run after the transfer and ablation suites have completed. The run directories
are local (`runs/` is gitignored); the paired summaries are written under
`results/comparison/` for commit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from cnn_classifier.evaluation.bootstrap import paired_bootstrap_difference
from cnn_classifier.evaluation.metrics import accuracy

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
RESULTS = REPO_ROOT / "results" / "comparison"

TRANSFER_BASELINE = "cifar10_resnet18_scratch"
TRANSFER_COMPARISONS = (
    ("linear_probe vs scratch", "cifar10_resnet18_linear_probe", TRANSFER_BASELINE),
    ("finetune vs scratch", "cifar10_resnet18_finetune", TRANSFER_BASELINE),
    ("finetune vs linear_probe", "cifar10_resnet18_finetune", "cifar10_resnet18_linear_probe"),
)
ABLATION_PAIRS = (
    ("augmentation", "ablation_augmentation", "ablation_no_augmentation"),
    ("batchnorm", "ablation_batchnorm", "ablation_no_batchnorm"),
    ("residual", "ablation_residual", "ablation_no_residual"),
    ("cosine vs constant", "ablation_cosine", "ablation_constant"),
)


def _find_run(name: str) -> Path:
    matches: list[tuple[str, Path]] = []
    for run in RUNS.iterdir():
        config_path = run / "config.resolved.yaml"
        summary_path = run / "summary.json"
        if not config_path.is_file() or not summary_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment", {}).get("name") != name:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("test_accuracy") is None:
            continue
        matches.append((run.name, run))
    if not matches:
        raise SystemExit(f"no completed run found for {name!r}")
    return sorted(matches)[-1][1]


def _predictions(run: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(run / "predictions" / "test_predictions.npz")
    return data["y_true"], data["y_pred"]


def _paired(truth: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    interval = paired_bootstrap_difference(
        truth,
        pred_a,
        pred_b,
        accuracy,
        n_resamples=1000,
        confidence=0.95,
        seed=42,
        stratify=True,
    )
    return {
        "difference": interval.estimate,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "confidence": 0.95,
    }


def _write_markdown(title: str, rows: list[dict], path: Path) -> None:
    lines = [
        f"# {title}",
        "",
        "| Comparison | Difference | 95% CI |",
        "|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison']} | {row['difference']:+.4f} | "
            f"{row['ci_low']:+.4f} - {row['ci_high']:+.4f} |"
        )
    lines.append("")
    lines.append(
        "Paired differences use 1,000 stratified bootstrap resamples on the "
        "shared test predictions. A positive difference means the first run "
        "has higher accuracy."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    baseline = _find_run(TRANSFER_BASELINE)
    truth, _ = _predictions(baseline)
    transfer_rows = []
    for label, name, comparator_name in TRANSFER_COMPARISONS:
        run = _find_run(name)
        comparator = _find_run(comparator_name)
        run_truth, run_predictions = _predictions(run)
        comparator_truth, comparator_predictions = _predictions(comparator)
        if not np.array_equal(truth, run_truth) or not np.array_equal(truth, comparator_truth):
            raise SystemExit(f"test predictions are not aligned for {label}")
        transfer_rows.append(
            {
                "comparison": label,
                **_paired(truth, run_predictions, comparator_predictions),
            }
        )
    (RESULTS / "transfer_paired.json").write_text(
        json.dumps({"rows": transfer_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        "Transfer paired bootstrap (accuracy)",
        transfer_rows,
        RESULTS / "transfer_paired.md",
    )

    ablation_rows = []
    for label, on_name, off_name in ABLATION_PAIRS:
        on_run = _find_run(on_name)
        off_run = _find_run(off_name)
        on_truth, on_predictions = _predictions(on_run)
        off_truth, off_predictions = _predictions(off_run)
        if not np.array_equal(on_truth, off_truth):
            raise SystemExit(f"test predictions are not aligned for {label}")
        ablation_rows.append(
            {
                "comparison": f"{label} (on - off)",
                **_paired(on_truth, on_predictions, off_predictions),
            }
        )
    (RESULTS / "ablation_paired.json").write_text(
        json.dumps({"rows": ablation_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        "Ablation paired bootstrap (accuracy)",
        ablation_rows,
        RESULTS / "ablation_paired.md",
    )
    print("wrote paired transfer and ablation comparisons")


if __name__ == "__main__":
    main()
