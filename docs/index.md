# cnn-classifier

A config-driven, reproducible PyTorch toolkit for CNN image classification.

`cnn-classifier` turns a new image/label dataset into a trained, evaluated, and
reported classifier through a typed YAML configuration. The reusable core covers
data adapters, a configurable residual CNN, torchvision transfer learning,
training, evaluation, diagnostics, and artifact-driven reports.

## Status

Sharp v1 is under active development. The approved design and implementation plan
are in `docs/superpowers/`.

## Quickstart

```bash
uv sync --all-extras
uv run cnncls doctor
```

## What this is not

This is a modern, reproducible, well-diagnosed CNN training toolkit. It is not a
state-of-the-art benchmark submission, a model zoo, or a notebook collection.
