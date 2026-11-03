# Evaluation rigor

## What

Evaluation covers top-k accuracy, macro F1, per-class precision/recall/F1,
confusion matrices, NLL, calibration (ECE/MCE/Brier), bootstrap confidence
intervals, error analysis, optional test-time augmentation, latency, and fair
multi-run comparison with test-fingerprint enforcement.

## Why

A single accuracy number hides class imbalance, overconfidence, and test-set
contamination. Evaluation is a separate artifact-producing step so the test
split is never used for model selection and every reported number has a
provenance.

## Config keys

| Knob | Config key | Notes |
|---|---|---|
| Metrics | `evaluation.metrics` | list of metric names |
| Top-k | `evaluation.top_k` | `k` for top-k accuracy |
| Bootstrap | `evaluation.bootstrap` | enabled, resamples, confidence, stratify |
| Calibration | `evaluation.calibration.bins` | reliability bins |
| Latency | `evaluation.latency` | warmup, iterations |
| Error analysis | `evaluation.error_analysis.max_examples` | gallery size |
| TTA | `evaluation.tta` | identity / hflip, mean probabilities |

## Implementation

`evaluation.metrics` computes the scalar metrics; `calibration` computes
reliability bins; `bootstrap` resamples the predictions; `error_analysis`
collects the highest-confidence errors; `compare` refuses runs with different
test fingerprints. Test is loaded only by the explicit `evaluate` command.

## Evidence

- CIFAR-10 scratch baseline: **92.25%** test accuracy, macro F1 92.26%,
  top-5 98.90%, 95% bootstrap CI 91.76–92.77%, ECE 0.0434, NLL 0.2894.
- The test split is evaluated once, from the best validation checkpoint
  (epoch 46), not used for selection.
- Every run records its test fingerprint, including the torchvision adapter's
  separate test set. The three transfer runs share the fingerprint
  `7aac74439fdc9640`.
- The optical degradation demo reaches 85.71% test accuracy with a 95% CI of
  82.38–88.81% on 420 generated test samples.
- The ablation and transfer comparisons include per-run intervals; the
  comparison report does not claim significance from overlapping intervals.

## Caveat

Bootstrap intervals describe uncertainty for one run; they are not a paired
significance test across runs. Overlapping intervals are not evidence of no
effect. Calibration and latency depend on the device and batch size, and the
reported numbers are measurements, not guarantees.
