"""Run the bring-your-own-data quickstart end to end.

This is the CI smoke path: inspect the committed 16-bit fixture, train for two
epochs, evaluate the validation split, and rebuild the report.

Usage:
    python demos/bring_your_own_data/scripts/run_quickstart.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import typer

from cnn_classifier.cli import evaluate, inspect_data, report, train

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "demos" / "bring_your_own_data" / "config.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "byo_quickstart"


@dataclass(frozen=True)
class QuickstartResult:
    run_dir: Path
    report_path: Path


def _existing_run_dirs(runs_dir: Path) -> set[Path]:
    if not runs_dir.exists():
        return set()
    return {path for path in runs_dir.iterdir() if path.is_dir()}


def run_quickstart(config_path: Path, output_root: Path) -> QuickstartResult:
    """Inspect, train, evaluate, and report on the tiny fixture."""
    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    before = _existing_run_dirs(output_root)

    inspect_data(config_path=config_path, set_values=None, max_samples=None)
    train(
        config_path=config_path,
        set_values=[],
        resume=None,
        dry_run=False,
        runs_dir=output_root,
        allow_config_change=False,
        no_evaluate=True,
    )
    new_runs = sorted(_existing_run_dirs(output_root) - before)
    if len(new_runs) != 1:
        raise RuntimeError(f"expected one new run directory, found {len(new_runs)}")
    run_dir = new_runs[0]

    evaluate(
        checkpoint=run_dir / "checkpoints" / "best.pt",
        config_path=None,
        split="val",
        set_values=None,
        run_dir=run_dir,
        allow_config_change=False,
        no_figures=False,
    )
    report(run=run_dir, config_path=None)
    return QuickstartResult(run_dir=run_dir, report_path=run_dir / "report.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    try:
        result = run_quickstart(args.config, args.output_root)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from exc
    print(f"Run directory: {result.run_dir}")
    print(f"Report: {result.report_path}")


if __name__ == "__main__":
    main()
