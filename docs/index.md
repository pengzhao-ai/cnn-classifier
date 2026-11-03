# cnn-classifier

A config-driven, reproducible PyTorch toolkit for CNN image classification.

`cnn-classifier` turns a new image/label dataset into a trained, evaluated, and
reported classifier through a typed YAML configuration. The reusable core covers
data adapters, configurable CNN and torchvision models, training, evaluation,
diagnostics, and artifact-driven reports.

!!! note "Milestone A complete"
    The core library, CLI, evaluation stack, diagnostics, visualization, and
    report generator are implemented and tested. Dataset demos and the full
    results site are the next milestone.

## Why this exists

Most CNN projects start as a notebook with a hard-coded dataset, model, and
training loop. Reusing that work on the next image problem means rewriting the
same plumbing. This project makes the plumbing explicit:

- **One configuration contract.** Every experiment is a validated YAML file.
  Unknown keys fail; CLI overrides are checked against the same schema.
- **One data contract.** Adapters produce `float32 [C, H, W]` samples, integer
  labels, and a written model contract. Raw files, NumPy arrays, and torchvision
  datasets all feed the same trainer.
- **One artifact contract.** A run directory is self-describing: resolved config,
  environment, git state, fingerprints, checkpoints, metrics, predictions,
  figures, and a report.
- **Test discipline.** The test split is isolated during training and available
  only through an explicit evaluation step.

This is a modern, reproducible, well-diagnosed CNN toolkit. It is not a
state-of-the-art benchmark submission, a model zoo, or a notebook collection.

## Install and verify

```bash
make sync
source .venv/bin/activate
cnncls doctor
```

The package is installed non-editable so the console entry point works reliably
on macOS even after PyTorch loads. Run `make sync` after changing library code.

## Quickstart

The synthetic config exercises the entire pipeline without downloading data:

```bash
cnncls train -c configs/tests/synthetic.yaml --set training.epochs=1
```

Inspect the raw-to-intake contract and debug before a longer run:

```bash
cnncls inspect-data -c configs/tests/synthetic.yaml
cnncls debug -c configs/tests/synthetic.yaml --overfit-batch --steps 20
cnncls train -c configs/tests/synthetic.yaml --dry-run
```

Evaluate and reuse artifacts:

```bash
cnncls evaluate --checkpoint runs/<run>/checkpoints/best.pt --split test
cnncls predict --checkpoint runs/<run>/checkpoints/best.pt --input image.png
cnncls report --run runs/<run>
cnncls compare runs/<run-a> runs/<run-b> --out results/comparison
```

## Command line

| Command | Purpose |
|---|---|
| `cnncls init DIR` | Scaffold a config, README, and data layout |
| `cnncls doctor` | Versions, device selection, optional dependencies |
| `cnncls inspect-data -c CONFIG` | Class balance, duplicates, corruption, leakage, model contract |
| `cnncls debug -c CONFIG` | One-batch overfit, shape trace, gradients, determinism |
| `cnncls train -c CONFIG` | Train, evaluate, write figures and the report |
| `cnncls evaluate --checkpoint PATH` | Evaluate on `test` or `val`, write predictions and metrics |
| `cnncls predict --checkpoint PATH --input PATH` | Image, folder, or manifest inference |
| `cnncls report --run DIR` | Rebuild HTML and Markdown from artifacts |
| `cnncls compare RUN... --out DIR` | Fair comparison with test-fingerprint enforcement |

`train --dry-run` builds the dataset and model, runs one forward/backward pass,
prints the contract and parameter counts, and exits without writing a run.

## Bring your own data

The `image_folder` adapter expects one directory per class:

```text
data/
├── train/{class_a,class_b,...}/
├── val/{class_a,class_b,...}/
└── test/{class_a,class_b,...}/     # optional
```

`cnncls init my_project` writes a starter `config.yaml` for this layout. Other
adapters cover CSV manifests, NumPy arrays, torchvision datasets, and synthetic
fixtures.

## What is covered

- **Augmentation and regularization:** torchvision v2 transforms, MixUp, CutMix,
  label smoothing, dropout, weight decay, early stopping.
- **Normalization and BatchNorm:** computed or supplied statistics, explicit normalization transforms,
  BatchNorm on/off in both CNN families.
- **Learning-rate management:** warmup, cosine, step, plateau, batch-level
  stepping, differential parameter groups.
- **ResNet and transfer learning:** configurable residual stages, torchvision
  backbones, 32x32 stem adaptation, freezing, BatchNorm policy, linear probes.
- **Evaluation:** top-k, macro F1, per-class precision/recall/F1, confusion
  matrices, NLL, ECE/MCE/Brier, bootstrap intervals, TTA, latency, error
  galleries.
- **Reproducibility:** seeded data splits, RNG state in checkpoints, deterministic
  mode, environment and git capture, data fingerprints.

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
make docs        # this site
make build       # wheel and sdist
make gate        # all of the above
```

The approved design and implementation plan are in `docs/superpowers/`.

## License

Apache-2.0. See `LICENSE`.
