"""Train and report the optical-degradation classifier.

Generate the dataset first:

    python demos/optics_degradation/generate_dataset.py --quick

Then run the configured classifier:

    python demos/optics_degradation/scripts/run_optics.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import typer

from cnn_classifier.cli import train

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "optics_degradation" / "degradation_classifier.yaml"
MANIFEST_PATH = REPO_ROOT / "data" / "optics_degradation" / "manifest.csv"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


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
    args = parser.parse_args()

    if not MANIFEST_PATH.is_file():
        raise SystemExit(
            "manifest not found; run "
            "`python demos/optics_degradation/generate_dataset.py --quick` first"
        )

    runs_dir: Path = args.runs_dir.resolve()
    before = _existing_run_dirs(runs_dir)
    try:
        train(
            config_path=CONFIG_PATH,
            set_values=[],
            resume=None,
            dry_run=False,
            runs_dir=runs_dir,
            allow_config_change=False,
            no_evaluate=False,
        )
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc

    new_runs = sorted(_existing_run_dirs(runs_dir) - before)
    if new_runs:
        run_dir = new_runs[-1]
        print(f"Run directory: {run_dir}")
        print(f"Report: {run_dir / 'report.html'}")


if __name__ == "__main__":
    main()
