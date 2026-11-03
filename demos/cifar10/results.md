# CIFAR-10 scratch baseline — results

Status: reference baseline reproduced on Apple Silicon MPS. Committed
summaries, figures, and reports are in `results/cifar10/`; the full run
directory remains under `runs/` (gitignored).

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

## Transfer benchmark (32×32 ResNet-18)

All three runs share the CIFAR-10 test set and the same test fingerprint.
The full comparison, Pareto chart, and caveats are in
`results/comparison/transfer_comparison.md`.

| Run | Test accuracy | 95% CI | Macro F1 | Trainable params | Train time |
|---|---:|---|---:|---:|---:|
| Scratch | 90.88% | 90.33–91.43% | 90.90% | 11,181,642 | 19.6 min |
| Linear probe | 62.66% | 61.75–63.63% | 62.54% | 5,130 | 8.4 min |
| Fine-tune | **94.98%** | 94.57–95.41% | 94.97% | 11,181,642 | 19.5 min |

The scratch control uses ImageNet normalization to match the pretrained input
contract. BatchNorm running statistics update even while the probe parameters
are frozen, so the linear probe is not a strict frozen-feature probe; the
comparison labels that caveat rather than hiding it.

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

Committed copies of the summary, evaluation, reports, and figures are in
`results/cifar10/`. Checkpoints stay local and are not committed.

## Notes

- The test split is evaluated only after training, from `checkpoints/best.pt`.
- `best.pt` is selected by validation accuracy, not test accuracy.
- The quick mode (`--quick`) is a two-epoch pipeline smoke check, not a result.
- The CIFAR test fingerprint is now recorded and matches the transfer runs.
