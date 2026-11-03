"""Run the four paired CIFAR-10 ablation experiments and compare them.

Each pair changes exactly one design choice: augmentation, BatchNorm, residual
connections, or the learning-rate schedule. Training for all eight configs runs
first with test evaluation disabled; the test split is evaluated only after the
whole suite finishes, then the runs are compared.

Usage:
    python demos/ablations/run_ablations.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import typer

from cnn_classifier.cli import compare, evaluate, train

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "ablations"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_COMPARISON_DIR = REPO_ROOT / "results" / "ablations"

RUN_SPECS = (
    ("augmentation", CONFIG_DIR / "augmentation.yaml"),
    ("none_augmentation", CONFIG_DIR / "none_augmentation.yaml"),
    ("batchnorm", CONFIG_DIR / "batchnorm.yaml"),
    ("no_batchnorm", CONFIG_DIR / "no_batchnorm.yaml"),
    ("residual", CONFIG_DIR / "residual.yaml"),
    ("no_residual", CONFIG_DIR / "no_residual.yaml"),
    ("cosine", CONFIG_DIR / "cosine.yaml"),
    ("constant", CONFIG_DIR / "constant.yaml"),
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
        help="Directory for the comparison report (default: <repo>/results/ablations).",
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

        # Test evaluation happens only after all eight training runs complete.
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
        print(
            "Per-run 95% bootstrap intervals are in each run's summary.json; "
            "no paired significance claim is made."
        )
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc


if __name__ == "__main__":
    main()
