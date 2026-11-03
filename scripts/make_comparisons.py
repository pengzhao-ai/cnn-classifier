"""Rewrite comparison artifacts with readable labels and richer columns."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cnn_classifier.viz.compare import plot_model_comparison
from cnn_classifier.viz.style import PALETTE, save_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "comparison"
DOCS_IMAGES = REPO_ROOT / "docs" / "images"

TRANSFER = (
    ("scratch", "transfer/scratch"),
    ("linear_probe", "transfer/linear_probe"),
    ("finetune", "transfer/finetune"),
)
ABLATIONS = (
    ("augmentation", "ablations/augmentation"),
    ("none_augmentation", "ablations/none_augmentation"),
    ("batchnorm", "ablations/batchnorm"),
    ("no_batchnorm", "ablations/no_batchnorm"),
    ("residual", "ablations/residual"),
    ("no_residual", "ablations/no_residual"),
    ("cosine", "ablations/cosine"),
    ("constant", "ablations/constant"),
)


def _summary(relative: str) -> dict:
    path = REPO_ROOT / "results" / relative / "reports" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _row(label: str, relative: str) -> dict:
    summary = _summary(relative)
    interval = summary.get("accuracy_ci") or [None, None]
    return {
        "run": label,
        "accuracy": summary.get("test_accuracy"),
        "ci_low": interval[0],
        "ci_high": interval[1],
        "macro_f1": summary.get("macro_f1"),
        "params": summary.get("trainable_params"),
        "total_params": summary.get("params"),
        "train_seconds": summary.get("train_seconds"),
        "latency_ms": summary.get("latency_ms"),
        "peak_memory_mb": summary.get("peak_memory_mb"),
        "test_fingerprint": summary.get("test_fingerprint"),
    }


def _write_markdown(title: str, rows: list[dict], notes: list[str], path: Path) -> None:
    lines = [
        f"# {title}",
        "",
        "| Run | Test accuracy | 95% CI | Macro F1 | Trainable params | "
        "Train time | Latency (ms) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        train_minutes = (
            f"{row['train_seconds'] / 60:.1f} min" if row["train_seconds"] is not None else "n/a"
        )
        latency = f"{row['latency_ms']:.2f}" if row["latency_ms"] is not None else "n/a"
        lines.append(
            f"| {row['run']} | {row['accuracy']:.4f} | "
            f"{row['ci_low']:.4f} - {row['ci_high']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['params']:,} | "
            f"{train_minutes} | {latency} |"
        )
    lines.append("")
    lines.extend(notes)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    transfer_rows = [_row(label, relative) for label, relative in TRANSFER]
    (RESULTS / "transfer_comparison.json").write_text(
        json.dumps({"rows": transfer_rows, "warnings": []}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        "32x32 ResNet-18 transfer benchmark",
        transfer_rows,
        [
            "All three runs share the CIFAR-10 test set; the recorded test "
            "fingerprint is identical across rows.",
            "The scratch control receives ImageNet normalization to match the "
            "pretrained input contract; the reference CIFAR baseline uses CIFAR "
            "normalization. The difference is intentional and labeled.",
            "BatchNorm running statistics continue to update while the linear-probe "
            "parameters are frozen; the probe is not a strict frozen-feature probe.",
            "Per-run bootstrap intervals are not a paired significance test.",
        ],
        RESULTS / "transfer_comparison.md",
    )
    plot_model_comparison(transfer_rows, RESULTS / "transfer_comparison.png")

    ablation_rows = [_row(label, relative) for label, relative in ABLATIONS]
    (RESULTS / "ablation_comparison.json").write_text(
        json.dumps({"rows": ablation_rows, "warnings": []}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        "CIFAR-10 core ablations",
        ablation_rows,
        [
            "Pairs: augmentation on/off, BatchNorm on/off, residual on/off, "
            "cosine/constant schedule.",
            "All eight runs use the same 10,000/2,000 reduced split, seed 42, "
            "8-epoch budget, and model shape.",
            "Per-run bootstrap intervals are not a paired significance test; "
            "overlapping intervals are not evidence of no effect.",
            "This is one seed and a reduced budget, not a universal ranking.",
        ],
        RESULTS / "ablation_comparison.md",
    )
    pair_labels = ("Augmentation", "BatchNorm", "Residual", "Schedule")
    pair_rows = [
        (ablation_rows[index], ablation_rows[index + 1])
        for index in range(0, len(ablation_rows), 2)
    ]
    positions = np.arange(len(pair_rows))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 4.5))
    for offset, (name, index) in ((-width / 2, ("on", 0)), (width / 2, ("off", 1))):
        values = [pair[index]["accuracy"] for pair in pair_rows]
        errors = [max(0.0, pair[index]["accuracy"] - pair[index]["ci_low"]) for pair in pair_rows]
        axis.bar(
            positions + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=name,
            color=PALETTE[0 if index == 0 else 1],
        )
    axis.set_xticks(positions, pair_labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Test accuracy")
    axis.set_title("CIFAR-10 core ablations (reduced budget)")
    axis.legend(title="Knob")
    fig.tight_layout()
    save_figure(fig, RESULTS / "ablation_comparison.png")
    plt.close(fig)

    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
    cifar_figures = REPO_ROOT / "results" / "cifar10" / "figures"
    for name in (
        "training_history.png",
        "confusion_matrix.png",
        "model_card.png",
        "errors.png",
    ):
        shutil.copy2(
            cifar_figures / name,
            DOCS_IMAGES / f"cifar10_{name}",
        )
    shutil.copy2(
        RESULTS / "transfer_comparison.png",
        DOCS_IMAGES / "transfer_comparison.png",
    )
    shutil.copy2(
        RESULTS / "ablation_comparison.png",
        DOCS_IMAGES / "ablation_comparison.png",
    )
    shutil.copy2(
        REPO_ROOT / "results" / "transfer" / "finetune" / "figures" / "training_history.png",
        DOCS_IMAGES / "transfer_finetune_training_history.png",
    )
    optics_figures = REPO_ROOT / "results" / "optics_degradation" / "figures"
    shutil.copy2(
        optics_figures / "confusion_matrix.png",
        DOCS_IMAGES / "optics_confusion_matrix.png",
    )
    shutil.copy2(
        optics_figures / "per_class.png",
        DOCS_IMAGES / "optics_per_class.png",
    )
    print("wrote labeled transfer and ablation comparisons")


if __name__ == "__main__":
    main()
