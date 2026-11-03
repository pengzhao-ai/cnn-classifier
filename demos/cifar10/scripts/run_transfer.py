"""Run the three 32x32 ResNet-18 transfer configurations and compare them.

Training is done first for all three configurations with test evaluation
disabled. Only after every run finishes does the script evaluate the test split
and build the fairness comparison. That ordering keeps the test set out of the
training loop and out of any intermediate model-selection decision.

Usage:
    python demos/cifar10/scripts/run_transfer.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import typer

from cnn_classifier.cli import compare, evaluate, train

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "cifar10"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_COMPARISON_DIR = REPO_ROOT / "results" / "comparison"

RUN_SPECS = (
    ("scratch", CONFIG_DIR / "resnet18_scratch_cifar.yaml"),
    ("linear_probe", CONFIG_DIR / "resnet18_linear_probe_cifar.yaml"),
    ("finetune", CONFIG_DIR / "resnet18_finetune_cifar.yaml"),
)


def _existing_run_dirs(runs_dir: Path) -> set[Path]:
    if not runs_dir.exists():
        return set()
    return {path for path in runs_dir.iterdir() if path.is_dir()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory for run artifacts (default: <repo>/runs).",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help="Directory for the comparison report (default: <repo>/results/comparison).",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir.resolve()
    comparison_dir: Path = args.comparison_dir.resolve()
    seen = _existing_run_dirs(runs_dir)
    run_dirs: list[Path] = []

    try:
        for label, config_path in RUN_SPECS:
            train(
                config_path=config_path,
                set_values=[],
                resume=None,
                dry_run=False,
                runs_dir=runs_dir,
                allow_config_change=False,
                no_evaluate=True,
            )
            new_runs = sorted(_existing_run_dirs(runs_dir) - seen)
            if len(new_runs) != 1:
                raise RuntimeError(
                    f"expected one new run directory for {label}, found {len(new_runs)}"
                )
            run_dir = new_runs[0]
            seen.add(run_dir)
            run_dirs.append(run_dir)
            print(f"{label}: {run_dir}")

        # Test evaluation happens only after all three training runs complete.
        for (label, _), run_dir in zip(RUN_SPECS, run_dirs, strict=True):
            evaluate(
                checkpoint=run_dir / "checkpoints" / "best.pt",
                config_path=None,
                split="test",
                set_values=None,
                run_dir=run_dir,
                allow_config_change=False,
                no_figures=False,
            )
            print(f"{label} report: {run_dir / 'report.html'}")

        compare(run_dirs, output_dir=comparison_dir, no_figure=False)
        print(f"Comparison: {comparison_dir / 'comparison.md'}")
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
