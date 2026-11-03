"""Command-line entry point for cnn-classifier."""

from __future__ import annotations

import csv
import importlib.util
import json
import platform
import resource
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
import typer
import yaml
from torch import nn
from torch.utils.data import DataLoader

from cnn_classifier._version import __version__
from cnn_classifier.config.loader import load_config
from cnn_classifier.config.resolve import validate_config_semantics
from cnn_classifier.config.schema import RunConfig
from cnn_classifier.core.device import device_report, resolve_device
from cnn_classifier.core.hashing import stable_hash
from cnn_classifier.core.seeding import seed_everything
from cnn_classifier.core.types import DatasetBundle, NormalizationStats
from cnn_classifier.data.adapters import build_dataset
from cnn_classifier.data.inspection import inspect_dataset
from cnn_classifier.data.loaders import DataLoaderBundle, build_loaders
from cnn_classifier.data.transforms import build_transforms
from cnn_classifier.diagnostics.determinism import determinism_check
from cnn_classifier.diagnostics.gradients import gradient_report
from cnn_classifier.diagnostics.sanity import overfit_batch
from cnn_classifier.diagnostics.tracing import trace_shapes
from cnn_classifier.evaluation.compare import compare_runs
from cnn_classifier.evaluation.evaluator import EvaluationResult, evaluate_model, save_predictions
from cnn_classifier.evaluation.predict import predict_paths
from cnn_classifier.models import build_model
from cnn_classifier.models.summary import model_summary
from cnn_classifier.runs.context import RunContext
from cnn_classifier.training.checkpoint import load_checkpoint
from cnn_classifier.training.losses import build_loss
from cnn_classifier.training.trainer import Trainer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Config-driven PyTorch toolkit for CNN image classification.",
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_SET_OPTION = typer.Option(None, "--set", "-s", help="Override a config value: key=value.")


def _version_callback(value: bool) -> None:
    """Print the package version and exit when --version is supplied."""
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the package version and exit.",
    ),
) -> None:
    """Run the cnn-classifier command line."""
    _ = version


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _optional_dependency(name: str) -> str:
    return "installed" if importlib.util.find_spec(name) is not None else "missing"


def _prepare(config: RunConfig) -> tuple[DatasetBundle, DataLoaderBundle, nn.Module, torch.device]:
    device = resolve_device(config.runtime.device)
    seed_everything(config.experiment.seed, deterministic=config.runtime.deterministic)
    bundle = build_dataset(config.data)
    loaders = build_loaders(bundle, config.training, config.runtime, device)
    model = build_model(
        config.model,
        num_classes=len(bundle.class_names),
        in_channels=bundle.in_channels,
        image_size=bundle.image_size,
    )
    return bundle, loaders, model, device


def _split_digest(bundle: DatasetBundle, indices: Sequence[int]) -> str:
    payload = [
        {
            "path": str(bundle.sample_metadata.paths[index]),
            "label": int(bundle.sample_metadata.labels[index]),
        }
        for index in indices
    ]
    return stable_hash(payload)


def _split_groups(bundle: DatasetBundle, indices: Sequence[int]) -> set[str]:
    return {
        str(bundle.sample_metadata.groups[index])
        for index in indices
        if bundle.sample_metadata.groups[index] is not None
    }


def _split_paths(bundle: DatasetBundle, indices: Sequence[int]) -> set[str]:
    return {str(bundle.sample_metadata.paths[index]) for index in indices}


def _fingerprint_payload(bundle: DatasetBundle) -> dict[str, Any]:
    split_info = bundle.split_info
    split_indices: dict[str, tuple[int, ...]] = {
        "train": split_info.train_indices,
        "val": split_info.val_indices,
    }
    if split_info.test_indices is not None:
        split_indices["test"] = split_info.test_indices
    split_rows = {
        name: {"count": len(indices), "fingerprint": _split_digest(bundle, indices)}
        for name, indices in split_indices.items()
    }
    primary = split_rows.get("test", split_rows["val"])["fingerprint"]
    train_groups = _split_groups(bundle, split_info.train_indices)
    val_groups = _split_groups(bundle, split_info.val_indices)
    group_overlap = sorted(train_groups & val_groups)
    train_paths = _split_paths(bundle, split_info.train_indices)
    val_paths = _split_paths(bundle, split_info.val_indices)
    duplicate_overlap = sorted(train_paths & val_paths)
    return {
        "dataset_fingerprint": bundle.fingerprint,
        "fingerprint": primary,
        "test_fingerprint": split_rows.get("test", {}).get("fingerprint"),
        "split_mode": split_info.mode,
        "seed": split_info.seed,
        "splits": split_rows,
        "group_overlap": group_overlap,
        "duplicate_overlap": duplicate_overlap,
    }


def _write_data_artifacts(run: RunContext, bundle: DatasetBundle) -> dict[str, Any]:
    payload = _fingerprint_payload(bundle)
    run.write_json("data/fingerprint.json", payload)
    run.write_json(
        "data/splits.json",
        {
            "mode": bundle.split_info.mode,
            "seed": bundle.split_info.seed,
            "train": list(bundle.split_info.train_indices),
            "val": list(bundle.split_info.val_indices),
            "test": list(bundle.split_info.test_indices or ()),
        },
    )
    run.write_json("data/class_names.json", list(bundle.class_names))
    normalization = asdict(bundle.normalization) if bundle.normalization is not None else None
    run.write_json("data/normalization.json", normalization)
    return payload


def _echo_contract(config: RunConfig, bundle: DatasetBundle) -> None:
    contract = bundle.model_contract
    typer.echo("Raw data -> model intake")
    typer.echo(f"  Adapter: {config.data.adapter}")
    typer.echo(f"  Classes ({len(bundle.class_names)}): {', '.join(bundle.class_names)}")
    typer.echo(
        "  Raw images are decoded, channel-converted, resized when configured, "
        "and scaled to [0, 1]."
    )
    typer.echo(
        f"Model input: float32 {contract.input_shape}, dtype={contract.dtype}, "
        f"value range={contract.value_range}"
    )
    if bundle.normalization is not None:
        typer.echo(
            f"  Normalization: mean={bundle.normalization.mean}, "
            f"std={bundle.normalization.std} ({bundle.normalization.source})"
        )
    else:
        typer.echo("  Normalization: none")
    typer.echo(f"  Channel policy: {contract.channel_policy}")


def _echo_summary(summary: dict[str, Any]) -> None:
    typer.echo(f"Parameters: {summary['total_params']:,} total")
    typer.echo(f"  Trainable: {summary['trainable_params']:,}")
    typer.echo(f"  Output shape: {summary['output_shape']}")


def _spinner(config: RunConfig, bundle: DatasetBundle) -> None:
    typer.echo(f"Device: {resolve_device(config.runtime.device)}")
    typer.echo(f"Planned epochs: {config.training.epochs}")
    test_count = len(bundle.split_info.test_indices or ())
    typer.echo(
        f"Samples: train={len(bundle.split_info.train_indices)}, "
        f"val={len(bundle.split_info.val_indices)}, test={test_count}"
    )


@app.command()
def doctor() -> None:
    """Print package and runtime information."""
    typer.echo(f"cnn-classifier {__version__}")
    typer.echo(f"python {platform.python_version()}")
    typer.echo(f"platform {platform.platform()}")
    typer.echo(f"pytorch {torch.__version__}")
    try:
        import torchvision

        typer.echo(f"torchvision {torchvision.__version__}")
    except ImportError:  # pragma: no cover - torchvision is a required dependency
        typer.echo("torchvision missing")
    try:
        device = resolve_device("auto")
        typer.echo(f"device: {device} ({device.type})")
    except Exception as exc:  # pragma: no cover - defensive hardware reporting
        typer.echo(f"device: unavailable ({exc})")
    report = device_report()
    typer.echo(f"mps available: {report.get('mps_available')}")
    typer.echo(f"cuda available: {report.get('cuda_available')}")
    for name in ("tensorboard", "sklearn", "torchinfo", "matplotlib", "jinja2"):
        typer.echo(f"{name}: {_optional_dependency(name)}")


_INIT_CONFIG = """schema_version: "1.0"

experiment:
  name: my-classifier
  seed: 42

data:
  adapter: image_folder
  root: data
  class_names: auto
  channels: rgb
  image_size: [224, 224]
  split:
    mode: explicit
  normalization:
    mode: compute
  transforms:
    train:
      - name: RandomResizedCrop
        size: [224, 224]
      - name: RandomHorizontalFlip
        p: 0.5
      - name: ToTensor
      - name: Normalize
        mean: auto
        std: auto
    eval:
      - name: Resize
        size: [224, 224]
      - name: ToTensor
      - name: Normalize
        mean: auto
        std: auto

model:
  name: residual_cnn

training:
  epochs: 50
  batch_size: 64
  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    name: cosine
  early_stopping:
    enabled: true
    patience: 10

evaluation:
  bootstrap:
    enabled: true
  latency:
    enabled: true

report:
  enabled: true
  formats: [html, markdown]

runtime:
  device: auto
"""

_INIT_README = """# Image classification project

Put images in `data/train/<class>/`, `data/val/<class>/`, and optionally
`data/test/<class>/`. Each class is a directory name.

```bash
cnncls inspect-data -c config.yaml
cnncls train -c config.yaml
```

The command writes a self-describing run directory under `runs/`, including
the resolved configuration, model contract, data fingerprints, checkpoints,
predictions, figures, and an HTML report.
"""


@app.command()
def init(
    directory: Path = typer.Argument(Path("cnn_project"), help="Project directory to create."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing scaffold."),
) -> None:
    """Create a starter config, README, and data layout."""
    config_path = directory / "config.yaml"
    if config_path.exists() and not force:
        typer.echo(f"Error: {config_path} already exists (use --force)", err=True)
        raise typer.Exit(code=1)
    (directory / "data").mkdir(parents=True, exist_ok=True)
    config_path.write_text(_INIT_CONFIG, encoding="utf-8")
    (directory / "README.md").write_text(_INIT_README, encoding="utf-8")
    typer.echo(f"Created project at {directory}")
    typer.echo(f"  config: {config_path}")
    typer.echo("  next: fill data/train and data/val, then run cnncls inspect-data -c config.yaml")


@app.command("inspect-data")
def inspect_data(
    config_path: Path = typer.Option(..., "-c", "--config", exists=True, dir_okay=False),
    set_values: list[str] | None = _SET_OPTION,
    max_samples: int | None = typer.Option(
        None, "--max-samples", help="Limit inspection work per split."
    ),
) -> None:
    """Report class balance, duplicates, corruption, leakage, and the model contract."""
    try:
        config = load_config(config_path, set_values or [])
        bundle = build_dataset(config.data)
        report = inspect_dataset(bundle, max_samples=max_samples)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Class counts")
    for split_name, counts in report.class_counts.items():
        details = ", ".join(f"{name}={count}" for name, count in counts.items())
        typer.echo(f"  {split_name}: {details}")
    duplicate_groups = sum(1 for count in report.duplicates.values() if count > 1)
    typer.echo(f"Duplicates: {duplicate_groups} content group(s)")
    typer.echo(f"Corrupt samples: {len(report.corrupt)}")
    typer.echo(f"Split overlap: {len(report.split_overlap)}")
    typer.echo(f"Group leakage: {len(report.group_leakage)}")
    for warning in report.warnings:
        typer.echo(f"Warning: {warning}")
    typer.echo("")
    _echo_contract(config, bundle)


@app.command()
def debug(
    config_path: Path = typer.Option(..., "-c", "--config", exists=True, dir_okay=False),
    set_values: list[str] | None = _SET_OPTION,
    overfit: bool = typer.Option(False, "--overfit-batch", help="Overfit a single batch."),
    gradients: bool = typer.Option(False, "--gradients", help="Print per-layer gradient norms."),
    determinism: bool = typer.Option(
        False, "--determinism", help="Check repeated seeded training steps."
    ),
    trace: bool = typer.Option(False, "--trace", help="Print module shape traces."),
    steps: int = typer.Option(100, "--steps", help="Maximum overfit steps."),
) -> None:
    """Run cheap training diagnostics without plotting or writing a run."""
    try:
        config = load_config(config_path, set_values or [])
        bundle, loaders, model, device = _prepare(config)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    batch = next(iter(loaders.train))
    images = batch[0].to(device)
    labels = batch[1].to(device)
    if overfit:
        loss_fn = build_loss(config.training.loss)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        result = overfit_batch(model, (images, labels), loss_fn, optimizer, device, max_steps=steps)
        typer.echo(
            f"Overfit batch: loss {result.initial_loss:.4f} -> {result.final_loss:.4f} "
            f"in {result.steps} step(s)"
        )
    if gradients:
        model.train()
        grad_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        grad_optimizer.zero_grad(set_to_none=True)
        loss = build_loss(config.training.loss)(model(images), labels)
        loss.backward()
        stats = sorted(gradient_report(model), key=lambda item: item.norm, reverse=True)
        typer.echo("Gradient norms")
        for stat in stats[:10]:
            typer.echo(
                f"  {stat.name}: norm={stat.norm:.4f}, zero_fraction={stat.zero_fraction:.3f}"
            )
        grad_optimizer.zero_grad(set_to_none=True)
    if determinism:

        def train_once() -> torch.Tensor:
            seed_everything(config.experiment.seed, deterministic=config.runtime.deterministic)
            candidate = build_model(
                config.model,
                num_classes=len(bundle.class_names),
                in_channels=bundle.in_channels,
                image_size=bundle.image_size,
            ).to(device)
            optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
            optimizer.zero_grad(set_to_none=True)
            value = build_loss(config.training.loss)(candidate(images), labels)
            value.backward()
            optimizer.step()
            return cast(torch.Tensor, value.detach().clone())

        equal = determinism_check(train_once, config.experiment.seed)
        typer.echo(f"Determinism: {'passed' if equal else 'FAILED'}")
    if trace or not (overfit or gradients or determinism):
        typer.echo("Shape trace")
        for layer in trace_shapes(model, bundle.model_contract.input_shape)[:24]:
            typer.echo(
                f"  {layer.name}: {layer.type} {layer.input_shape} -> "
                f"{layer.output_shape} ({layer.params:,} params)"
            )


def _history_from_run(run_dir: Path) -> list[dict[str, float]]:
    summary = _read_json(run_dir / "summary.json")
    history = summary.get("history")
    if not isinstance(history, list):
        return []
    return [
        {str(key): float(value) for key, value in row.items()}
        for row in history
        if isinstance(row, dict)
    ]


def _evaluation_payload(
    config: RunConfig,
    result: EvaluationResult,
    split: str,
) -> dict[str, Any]:
    metrics = result.metrics
    calibration = asdict(metrics.calibration) if metrics.calibration is not None else None
    intervals = {
        name: {
            "low": interval.low,
            "high": interval.high,
            "estimate": interval.estimate,
            "confidence": config.evaluation.bootstrap.confidence,
        }
        for name, interval in result.confidence_intervals.items()
    }
    payload: dict[str, Any] = {
        "split": split,
        "metrics": {
            "accuracy": metrics.accuracy,
            "top_k_accuracy": metrics.top_k_accuracy,
            "macro_f1": metrics.macro_f1,
            "per_class": {name: asdict(value) for name, value in metrics.per_class.items()},
            "confusion_matrix": metrics.confusion_matrix,
            "nll": metrics.nll,
            "calibration": calibration,
            "support": metrics.support,
        },
        "confidence_intervals": intervals,
        "latency": asdict(result.latency) if result.latency is not None else None,
        "tta": {
            "enabled": config.evaluation.tta.enabled,
            "base": None,
            "tta": None,
        },
    }
    if result.base_metrics is not None:
        payload["tta"]["base"] = {"accuracy": result.base_metrics.accuracy}
    if result.tta_metrics is not None:
        payload["tta"]["tta"] = {"accuracy": result.tta_metrics.accuracy}
    return payload


def _peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    if device.type == "mps":
        try:
            return float(torch.mps.current_allocated_memory() / (1024.0 * 1024.0))
        except (AttributeError, RuntimeError):
            return None
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
    return raw / divisor


def _update_summary(
    run: RunContext,
    model: nn.Module,
    result: EvaluationResult,
    split: str,
    fingerprint: dict[str, Any],
    device: torch.device,
) -> None:
    summary = _read_json(run.directory / "summary.json")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    interval = result.confidence_intervals.get("accuracy")
    summary.update(
        {
            "evaluated_split": split,
            "macro_f1": result.metrics.macro_f1,
            "params": total,
            "trainable_params": trainable,
            "latency_ms": result.latency.mean_ms if result.latency is not None else None,
            "peak_memory_mb": _peak_memory_mb(device),
            "dataset_fingerprint": fingerprint.get("fingerprint"),
        }
    )
    if split == "test":
        summary["test_accuracy"] = result.metrics.accuracy
        summary["test_fingerprint"] = fingerprint.get("test_fingerprint")
    else:
        summary["val_accuracy"] = result.metrics.accuracy
    if interval is not None:
        summary["accuracy_ci"] = [interval.low, interval.high]
    run.write_json("summary.json", summary)


def _denormalize(image: torch.Tensor, normalization: NormalizationStats | None) -> torch.Tensor:
    if normalization is None:
        return image.clamp(0.0, 1.0)
    mean = torch.tensor(normalization.mean, dtype=image.dtype).view(-1, 1, 1)
    std = torch.tensor(normalization.std, dtype=image.dtype).view(-1, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def _write_figures(
    run: RunContext,
    bundle: DatasetBundle,
    config: RunConfig,
    model: nn.Module,
    loader: DataLoader[Any],
    result: EvaluationResult,
    history: list[dict[str, float]],
) -> None:
    try:
        from cnn_classifier.viz.architecture import plot_model_card
        from cnn_classifier.viz.evaluation import (
            plot_calibration,
            plot_confusion_matrix,
            plot_error_gallery,
            plot_per_class_metrics,
        )
        from cnn_classifier.viz.training import plot_training_history
    except ImportError:  # pragma: no cover - plotting dependencies are optional
        return
    figures = run.directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if history:
        plot_training_history(history, figures / "training_history.png")
    plot_confusion_matrix(
        result.metrics.confusion_matrix,
        list(bundle.class_names),
        figures / "confusion_matrix.png",
    )
    plot_per_class_metrics(result.metrics.per_class, figures / "per_class.png")
    if result.metrics.calibration is not None:
        plot_calibration(result.metrics.calibration, figures / "calibration.png")
    summary = model_summary(model, bundle.model_contract.input_shape)
    plot_model_card(summary, figures / "model_card.png")

    dataset = loader.dataset
    dataset_indices = getattr(dataset, "indices", ())
    source_to_position = {int(source): position for position, source in enumerate(dataset_indices)}
    limit = min(config.evaluation.error_analysis.max_examples, 16)
    images: list[torch.Tensor] = []
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    for source, truth, prediction in zip(result.indices, result.y_true, result.y_pred, strict=True):
        if int(truth) == int(prediction):
            continue
        position = source_to_position.get(int(source))
        if position is None:
            continue
        image, _, _ = dataset[position]
        images.append(_denormalize(image, bundle.normalization))
        true_labels.append(bundle.class_names[int(truth)])
        predicted_labels.append(bundle.class_names[int(prediction)])
        if len(images) >= limit:
            break
    if images:
        plot_error_gallery(
            torch.stack(images),
            true_labels,
            predicted_labels,
            figures / "errors.png",
        )


def _write_report(run: RunContext, config: RunConfig) -> None:
    if not config.report.enabled:
        return
    from cnn_classifier.viz.report import build_markdown_report, build_report

    if "html" in config.report.formats:
        build_report(run.directory)
    if "markdown" in config.report.formats:
        (run.directory / "report.md").write_text(
            build_markdown_report(run.directory), encoding="utf-8"
        )


def _evaluate_into_run(
    run: RunContext,
    config: RunConfig,
    bundle: DatasetBundle,
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    split: str,
    history: list[dict[str, float]],
    figures: bool,
) -> EvaluationResult:
    result = evaluate_model(
        model,
        loader,
        device,
        bundle.class_names,
        config.evaluation,
    )
    save_predictions(result, run.directory / "predictions" / f"{split}_predictions.npz")
    fingerprint = _fingerprint_payload(bundle)
    run.write_json("evaluation.json", _evaluation_payload(config, result, split))
    _update_summary(run, model, result, split, fingerprint, device)
    if figures:
        _write_figures(run, bundle, config, model, loader, result, history)
    return result


@app.command()
def train(
    config_path: Path = typer.Option(..., "-c", "--config", exists=True, dir_okay=False),
    set_values: list[str] | None = _SET_OPTION,
    resume: Path | None = typer.Option(None, "--resume", help="Checkpoint to resume from."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without training."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Directory for new runs."),
    allow_config_change: bool = typer.Option(
        False, "--allow-config-change", help="Allow resuming with a changed config."
    ),
    no_evaluate: bool = typer.Option(False, "--no-evaluate", help="Skip test evaluation."),
) -> None:
    """Train a model from a YAML configuration."""
    run: RunContext | None = None
    try:
        config = load_config(config_path, set_values or [])
        if dry_run:
            bundle, loaders, model, device = _prepare(config)
            batch = next(iter(loaders.train))
            images = batch[0].to(device)
            labels = batch[1].to(device)
            model.train()
            loss = build_loss(config.training.loss)(model(images), labels)
            loss.backward()
            _echo_contract(config, bundle)
            _spinner(config, bundle)
            _echo_summary(model_summary(model, bundle.model_contract.input_shape))
            typer.echo(
                f"Dry run complete; one batch loss={float(loss.item()):.4f}; nothing written."
            )
            return

        if resume is not None:
            checkpoint_path = resume.resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
            run_directory = (
                checkpoint_path.parent.parent
                if checkpoint_path.parent.name == "checkpoints"
                else checkpoint_path.parent
            )
            previous_history = _history_from_run(run_directory)
            run = RunContext(
                directory=run_directory,
                config=config,
                config_hash=stable_hash(config.model_dump(mode="json")),
            )
            run.write_text(
                "config.resolved.yaml",
                yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            )
            run.set_status("running")
        else:
            previous_history = []
            run = RunContext.create(config, root=runs_dir)

        bundle, loaders, model, device = _prepare(config)
        _write_data_artifacts(run, bundle)
        trainer = Trainer(
            model,
            loaders,
            config,
            run,
            device,
            model_contract=bundle.model_contract,
            class_names=bundle.class_names,
            normalization=bundle.normalization,
        )
        if resume is not None:
            trainer.resume(
                resume,
                allow_config_change=allow_config_change,
                history=previous_history,
            )
        started = time.perf_counter()
        result = trainer.fit()
        train_seconds = time.perf_counter() - started
        summary = _read_json(run.directory / "summary.json")
        summary["train_seconds"] = train_seconds
        run.write_json("summary.json", summary)

        if not no_evaluate and (bundle.test is not None or bundle.val is not None):
            split = "test" if bundle.test is not None else "val"
            loader = loaders.test if split == "test" else loaders.val
            assert loader is not None
            best_path = result.best_checkpoint
            payload = load_checkpoint(
                best_path,
                map_location=device,
                expected_config_hash=run.config_hash,
                allow_config_change=allow_config_change,
            )
            trainer.model.load_state_dict(payload["model"])
            evaluated = _evaluate_into_run(
                run,
                config,
                bundle,
                trainer.model,
                loader,
                device,
                split,
                result.history,
                figures=config.report.enabled,
            )
            typer.echo(
                f"{split} accuracy: {evaluated.metrics.accuracy:.4f}  "
                f"macro F1: {evaluated.metrics.macro_f1:.4f}"
            )
        _write_report(run, config)
        typer.echo(f"Run directory: {run.directory}")
        typer.echo(f"Best checkpoint: {result.best_checkpoint}")
    except Exception as exc:
        if run is not None:
            run.set_status("failed", str(exc))
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _checkpoint_run_dir(checkpoint: Path) -> Path:
    resolved = checkpoint.resolve()
    return resolved.parent.parent if resolved.parent.name == "checkpoints" else resolved.parent


def _config_for_checkpoint(
    config_path: Path | None,
    set_values: list[str] | None,
    checkpoint: Path,
) -> RunConfig:
    if config_path is not None:
        return load_config(config_path, set_values or [])
    resolved = _checkpoint_run_dir(checkpoint) / "config.resolved.yaml"
    if not resolved.is_file():
        raise FileNotFoundError(
            f"no resolved config beside checkpoint; pass -c/--config ({resolved} missing)"
        )
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid resolved config: {resolved}")
    config = RunConfig.model_validate(payload)
    validate_config_semantics(config)
    return config


@app.command("evaluate")
def evaluate(
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, dir_okay=False),
    config_path: Path | None = typer.Option(
        None,
        "-c",
        "--config",
        exists=True,
        dir_okay=False,
        help="Config override; defaults to the run's resolved config.",
    ),
    split: str = typer.Option("test", "--split", help="Split to evaluate: test or val."),
    set_values: list[str] | None = _SET_OPTION,
    run_dir: Path | None = typer.Option(None, "--run", help="Run directory for artifacts."),
    allow_config_change: bool = typer.Option(False, "--allow-config-change"),
    no_figures: bool = typer.Option(False, "--no-figures", help="Skip figure generation."),
) -> None:
    """Evaluate a checkpoint on a held-out split and write artifacts."""
    if split not in {"test", "val"}:
        typer.echo("Error: --split must be 'test' or 'val'", err=True)
        raise typer.Exit(code=2)
    try:
        config = _config_for_checkpoint(config_path, set_values, checkpoint)
        bundle, loaders, model, device = _prepare(config)
        loader = loaders.test if split == "test" else loaders.val
        if loader is None:
            raise RuntimeError(f"the configured dataset has no {split} split")
        payload = load_checkpoint(
            checkpoint,
            map_location=device,
            expected_config_hash=stable_hash(config.model_dump(mode="json")),
            allow_config_change=allow_config_change,
        )
        model.load_state_dict(payload["model"])
        destination = run_dir or _checkpoint_run_dir(checkpoint)
        destination.mkdir(parents=True, exist_ok=True)
        run = RunContext(
            directory=destination,
            config=config,
            config_hash=stable_hash(config.model_dump(mode="json")),
        )
        history = _history_from_run(destination)
        result = _evaluate_into_run(
            run,
            config,
            bundle,
            model,
            loader,
            device,
            split,
            history,
            figures=not no_figures,
        )
        _write_report(run, config)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"{split} accuracy: {result.metrics.accuracy:.4f}  macro F1: {result.metrics.macro_f1:.4f}"
    )
    typer.echo(f"Wrote {destination / 'evaluation.json'}")


@app.command()
def report(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=False),
    config_path: Path | None = typer.Option(
        None, "-c", "--config", exists=True, dir_okay=False, help="Optional config to load."
    ),
) -> None:
    """Build HTML and Markdown reports from an existing run directory."""
    try:
        config = load_config(config_path) if config_path is not None else None
        from cnn_classifier.viz.report import build_markdown_report, build_report

        html = build_report(run)
        (run / "report.md").write_text(build_markdown_report(run), encoding="utf-8")
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _ = config
    typer.echo(f"Wrote {html}")
    typer.echo(f"Wrote {run / 'report.md'}")


def _prediction_paths(
    input_path: Path,
    image_root: Path | None,
) -> list[Path]:
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
    if input_path.suffix.lower() == ".csv":
        paths: list[Path] = []
        with input_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("path")
                if not value:
                    continue
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = (image_root or input_path.parent) / candidate
                paths.append(candidate)
        return paths
    return [input_path]


def _normalization_from_payload(payload: dict[str, Any]) -> NormalizationStats | None:
    raw = payload.get("normalization")
    if not isinstance(raw, dict):
        return None
    mean = raw.get("mean")
    std = raw.get("std")
    if not isinstance(mean, (list, tuple)) or not isinstance(std, (list, tuple)):
        return None
    return NormalizationStats(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
        source=raw.get("source", "none"),
    )


@app.command()
def predict(
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, dir_okay=False),
    input_path: Path = typer.Option(
        ..., "--input", exists=True, help="Image, folder, or manifest CSV."
    ),
    config_path: Path | None = typer.Option(
        None,
        "-c",
        "--config",
        exists=True,
        dir_okay=False,
        help="Config override; defaults to the run's resolved config.",
    ),
    output: Path = typer.Option(Path("predictions.csv"), "--output", help="CSV or JSON output."),
    image_root: Path | None = typer.Option(
        None, "--image-root", help="Base directory for relative manifest paths."
    ),
    top_k: int = typer.Option(3, "--top-k", help="Number of probabilities to report."),
    set_values: list[str] | None = _SET_OPTION,
    allow_config_change: bool = typer.Option(False, "--allow-config-change"),
) -> None:
    """Predict an image, a folder of images, or a manifest."""
    try:
        config = _config_for_checkpoint(config_path, set_values, checkpoint)
        device = resolve_device(config.runtime.device)
        checkpoint_payload = load_checkpoint(
            checkpoint,
            map_location=device,
            expected_config_hash=stable_hash(config.model_dump(mode="json")),
            allow_config_change=allow_config_change,
        )
        class_names = tuple(checkpoint_payload.get("class_names", ()))
        contract = checkpoint_payload.get("model_contract")
        if not class_names or not isinstance(contract, dict):
            raise RuntimeError("checkpoint is missing class names or the model contract")
        input_shape = contract.get("input_shape")
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 3:
            raise RuntimeError("checkpoint has an invalid model input shape")
        model = build_model(
            config.model,
            num_classes=len(class_names),
            in_channels=int(input_shape[0]),
            image_size=(int(input_shape[1]), int(input_shape[2])),
        )
        model.load_state_dict(checkpoint_payload["model"])
        model.to(device)
        normalization = _normalization_from_payload(checkpoint_payload)
        transform = build_transforms(config.data, normalization, train=False)
        paths = _prediction_paths(input_path, image_root)
        if not paths:
            raise RuntimeError(f"no images found at {input_path}")
        predictions = predict_paths(model, paths, transform, device, class_names)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    ranks = predictions.top_k(top_k)
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(predictions.paths):
        probabilities = {
            name: float(predictions.probabilities[index, class_index])
            for class_index, name in enumerate(class_names)
        }
        rows.append(
            {
                "path": str(path),
                "predicted": class_names[int(predictions.predicted_labels[index])],
                "confidence": float(predictions.probabilities[index].max()),
                "probabilities": probabilities,
                "top_k": [
                    {"label": label, "probability": probability}
                    for label, probability in ranks[index]
                ],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output if output.suffix.lower() == ".csv" else output.with_suffix(".csv")
    json_path = output if output.suffix.lower() == ".json" else output.with_suffix(".json")
    fieldnames = ["path", "predicted", "confidence", *class_names]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row["path"],
                    "predicted": row["predicted"],
                    "confidence": f"{row['confidence']:.6f}",
                    **{name: f"{row['probabilities'][name]:.6f}" for name in class_names},
                }
            )
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"Predicted {len(rows)} image(s)")
    for row in rows[:20]:
        typer.echo(f"  {row['path']}: {row['predicted']} ({row['confidence']:.3f})")
    typer.echo(f"Wrote {csv_path}")
    typer.echo(f"Wrote {json_path}")


@app.command()
def compare(
    run_dirs: list[Path] = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("results/comparison"), "--out", help="Output directory."),
    no_figure: bool = typer.Option(False, "--no-figure", help="Skip the comparison figure."),
) -> None:
    """Compare runs that share the same test fingerprint."""
    try:
        report = compare_runs(run_dirs, output_dir)
        if not no_figure:
            from cnn_classifier.viz.compare import plot_model_comparison

            plot_model_comparison(report.pareto_data, output_dir / "comparison.png")
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(report.markdown)
    for warning in report.warnings:
        typer.echo(f"Warning: {warning}")
    typer.echo(f"Wrote {output_dir}")


def main() -> None:
    """Console-script entry point."""
    app()


__all__ = ["app", "main"]
