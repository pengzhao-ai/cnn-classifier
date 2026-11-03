"""Run the CIFAR-10 scratch baseline demo.

Full mode uses the reference configuration for 50 epochs. Quick mode keeps the
same data and model but uses a 10,000/2,000 subset for two epochs with bootstrap
and latency measurement disabled, which is enough for a pipeline smoke check on
a laptop.

Usage:
    python demos/cifar10/scripts/run_scratch.py
    python demos/cifar10/scripts/run_scratch.py --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cnn_classifier.cli import train

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "cifar10" / "scratch_baseline.yaml"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"

QUICK_OVERRIDES = [
    "training.epochs=2",
    "data.adapter_params.max_train_samples=10000",
    "data.adapter_params.max_val_samples=2000",
    "evaluation.bootstrap.enabled=false",
    "evaluation.latency.enabled=false",
]


def _existing_run_dirs(runs_dir: Path) -> set[Path]:
    if not runs_dir.exists():
        return set()
    return {path for path in runs_dir.iterdir() if path.is_dir()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run two epochs with bootstrap and latency measurement disabled.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory for run artifacts (default: <repo>/runs).",
    )
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir.resolve()
    before = _existing_run_dirs(runs_dir)
    # Typer command functions expose OptionInfo objects as defaults when called
    # directly, so every option must be passed explicitly here.
    train(
        config_path=CONFIG_PATH,
        set_values=QUICK_OVERRIDES if args.quick else [],
        resume=None,
        dry_run=False,
        runs_dir=runs_dir,
        allow_config_change=False,
        no_evaluate=False,
    )
    new_runs = sorted(_existing_run_dirs(runs_dir) - before)
    if new_runs:
        run_dir = new_runs[-1]
        print(f"Run directory: {run_dir}")
        print(f"Report: {run_dir / 'report.html'}")


if __name__ == "__main__":
    main()
