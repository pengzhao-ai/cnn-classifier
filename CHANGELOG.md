# Changelog

All notable changes to `cnn-classifier-toolkit` are documented here.

## [Unreleased]

## [0.1.1] - 2026-11-22

### Fixed

- Gradient norm is captured at the optimizer window and written to
  `history.csv`, `metrics.jsonl`, `summary.json`, and training-history figures.
- The gradient half of `NaNDetector` checks gradients before the optimizer step
  and stops the run instead of applying a nonfinite update.
- `demos/cifar10/scripts/run_scratch.py --quick` applies its sample-limit
  overrides instead of failing at config load.

### Documentation

- Recorded validation of the 12 known-issue reports and closed the false,
  by-design, and already-resolved items.

## [0.1.0] - 2026-11-03

Initial public release.

### Added

- Typed YAML configuration, dataset adapters, image I/O, and split policies.
- Simple and residual CNN builders, torchvision transfer controls, and stem
  adaptation.
- Trainer with callbacks, checkpoints, resume, AMP, schedulers, MixUp/CutMix,
  and EMA.
- Evaluation, calibration, bootstrap intervals, error analysis, latency, TTA,
  prediction, and fair comparison.
- Diagnostics for overfit, tracing, gradients, activations, and determinism.
- Self-contained HTML/Markdown reports and publication-style figures.
- CIFAR-10 scratch baseline, 32×32 transfer benchmark, four paired ablations,
  optical-degradation demo, and bring-your-own-data quickstart.
- MkDocs Material documentation site, packaging metadata, CI, and release
  workflows.

### Notes

- The CIFAR-10 scratch baseline reproduced 92.25% test accuracy on Apple
  Silicon MPS (reference 92.61%).
- Full transfer, ablation, and optical results are committed under `results/`,
  with comparison tables and figures under `results/comparison/`.
