# cnn-classifier

A config-driven, reproducible PyTorch toolkit for CNN image classification.

`cnn-classifier` turns a new image/label dataset into a trained, evaluated, and
reported classifier through a typed YAML configuration. The same library powers
the command line, so a quick smoke test and a full experiment use one pipeline.
It is designed to be reused across problems, not tied to a single dataset.

## What is in the core

| Area | What you get |
|---|---|
| Data | torchvision, ImageFolder, CSV manifest, NumPy, and synthetic adapters; 8-bit and 16-bit image I/O; group-safe splits; leakage and duplicate inspection |
| Models | configurable simple CNN, configurable residual CNN, torchvision backbone wrapper, 32x32 stem adaptation, freezing, differential parameter groups |
| Training | callbacks, early stopping, checkpoint/resume, AMP, gradient clipping and accumulation, MixUp/CutMix, EMA, warmup/cosine/step/plateau schedules |
| Evaluation | top-k, per-class metrics, confusion matrices, NLL, calibration (ECE/MCE/Brier), bootstrap intervals, TTA, latency, error analysis |
| Diagnostics | one-batch overfit, shape tracing, gradient reports, determinism checks |
| Artifacts | resolved config, environment and git metadata, data fingerprints, metrics, checkpoints, predictions, figures, model contract |
| Reports | self-contained HTML and Markdown, inline figures, no CDN or network dependency |
| Devices | automatic CUDA, Apple MPS, or CPU selection |

## Install

```bash
make sync
source .venv/bin/activate
cnncls doctor
```

`make sync` installs the package non-editable so the `cnncls` entry point works
reliably on macOS, where an editable `.pth` can be hidden after PyTorch loads.
Run `make sync` again after changing library code. Without activating the
environment, use `uv run --no-sync cnncls ...`.

## Quickstart

Train a tiny deterministic smoke model with no downloads:

```bash
cnncls train -c configs/tests/synthetic.yaml --set training.epochs=1
```

The command creates a run directory under `runs/` containing the resolved
configuration, data fingerprints, model contract, checkpoints, test
predictions, figures, `report.html`, and `report.md`.

Inspect the pipeline before training:

```bash
cnncls inspect-data -c configs/tests/synthetic.yaml
cnncls debug -c configs/tests/synthetic.yaml --overfit-batch --steps 20
cnncls train -c configs/tests/synthetic.yaml --dry-run
```

Evaluate, predict, and rebuild reports from artifacts:

```bash
cnncls evaluate --checkpoint runs/<run>/checkpoints/best.pt --split test
cnncls predict --checkpoint runs/<run>/checkpoints/best.pt --input image.png
cnncls report --run runs/<run>
cnncls compare runs/<run-a> runs/<run-b> --out results/comparison
```

`evaluate` and `predict` use the run's resolved config by default; pass
`-c/--config` to make an override explicit. `compare` refuses to compare runs
whose test fingerprints differ.

## Bring your own data

Use the `image_folder` adapter with one directory per class:

```text
data/
├── train/{class_a,class_b,...}/
├── val/{class_a,class_b,...}/
└── test/{class_a,class_b,...}/     # optional
```

`cnncls init my_project` writes a starter config and this layout. The adapter
reports the raw-to-intake contract: images are decoded, converted to the
configured channel policy, scaled to `[0, 1]`, optionally normalized, and
batched as `float32 [B, C, H, W]` tensors.

## Project layout

```text
src/cnn_classifier/
├── config/       typed YAML schema, defaults chain, CLI overrides
├── core/         devices, seeding, hashing, shared contracts
├── data/         adapters, image I/O, splits, transforms, inspection
├── models/       CNN builders, torchvision wrapper, transfer utilities
├── training/     trainer, callbacks, checkpoints, optimizer, schedules, AMP
├── evaluation/   metrics, calibration, bootstrap, latency, comparison
├── diagnostics/  overfit, tracing, gradients, determinism
├── viz/          publication-style figures and HTML/Markdown reports
├── runs/         run directories and artifact writers
└── cli.py        `cnncls` command line
```

## Development

```bash
make test        # library tests
make coverage    # core coverage gate
make lint        # ruff
make typecheck   # mypy
make docs        # MkDocs site
make build       # wheel and sdist
make gate        # all of the above
```

The approved sharp-v1 design and implementation plan live in
`docs/superpowers/`. Milestone A, the reusable core library, is complete;
dataset demos and the published results site follow in Milestone B.

Documentation: <https://pengzhao-ai.github.io/cnn-classifier/>

## License

Apache-2.0. See `LICENSE`.
