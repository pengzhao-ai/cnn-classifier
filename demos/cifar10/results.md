# CIFAR-10 scratch baseline — results

Status: reference baseline reproduced on Apple Silicon MPS. Full run artifacts
are under `runs/20261103T055515Z_cifar10_scratch_baseline_0283f0f0/`
(gitignored); committed result packaging is Task 28 of the sharp-v1 plan.

## Reproduced run

| Field | Value |
|---|---|
| Run ID | `20261103T055515Z_cifar10_scratch_baseline_0283f0f0` |
| Device | Apple MPS (M3 Max) |
| Seed | 42 |
| Epochs | 50 (best epoch 46) |
| Best validation accuracy | **92.78%** |
| Test accuracy | **92.25%** |
| Test macro F1 | 92.26% |
| Test top-5 accuracy | 98.90% |
| Test accuracy 95% CI | 91.76% – 92.77% |
| Test NLL | 0.2894 |
| Calibration ECE | 0.0434 |
| Latency (MPS, batch inference) | 2.16 ms / image (462.9 images/s) |
| Peak MPS memory | 181.5 MB |
| Training time | 1,338 s (22.3 min) |
| Parameters | 7,827,786 (all trainable) |

## Reference comparison

| Split | Reference | Reproduced | Difference |
|---|---:|---:|---:|
| Validation | 92.42% | 92.78% | +0.36 pp |
| Test | 92.61% | 92.25% | −0.36 pp |

The reference comes from the project design document (MPS, seed 42). The
reproduced test accuracy is within 0.4 percentage points and clears the ≥90%
acceptance gate. No tuning was performed against the test split; the best
checkpoint was selected on validation accuracy (epoch 46) and evaluated once.

## Artifacts

Local run directory (not committed):

```text
runs/20261103T055515Z_cifar10_scratch_baseline_0283f0f0/
├── checkpoints/best.pt       # best validation checkpoint, ~90 MB
├── checkpoints/last.pt       # resumable final-epoch checkpoint, ~90 MB
├── evaluation.json           # per-class, confusion matrix, calibration, CI, latency
├── predictions/test_predictions.npz
├── figures/                  # training history, confusion matrix, per-class,
│                             # calibration, model card, error gallery
├── report.html, report.md    # self-contained reports
├── summary.json              # headline metrics and full history
├── data/                     # model contract, splits, fingerprints, normalization
└── environment.json, git.json, config.resolved.yaml
```

Task 28 copies selected summaries, figures, and reports into `results/cifar10/`
for commit. Checkpoints stay local and are not committed.

## Notes

- The test split is evaluated only after training, from `checkpoints/best.pt`.
- `best.pt` is selected by validation accuracy, not test accuracy.
- The quick mode (`--quick`) is a two-epoch pipeline smoke check, not a result.
- The CIFAR adapter's recorded test fingerprint is currently null; comparisons
  fall back to the shared dataset/validation fingerprint. This is noted for the
  Task 22 fairness review.
