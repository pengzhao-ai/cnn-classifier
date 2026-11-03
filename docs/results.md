# Results

This page is the single source for committed benchmark numbers. Demo pages own
the narrative; this page owns the comparison and the environment.

## Environment

| Field | Value |
|---|---|
| Machine | Apple M3 Max, macOS 27.0.1 arm64 |
| Device | MPS |
| Python | 3.11.8 |
| PyTorch | 2.15.0 |
| TorchVision | 0.29.2 |
| Seed | 42 for the reference and transfer anchor; 1 for the optics demo |

The CIFAR-10 scratch baseline uses seed 42. The remaining full-suite timings
are estimates until the benchmark stage records them.

## Benchmark table

| Experiment | Test accuracy | Macro F1 | Params | Training time | Status |
|---|---:|---:|---:|---:|---|
| CIFAR-10 scratch baseline | **92.25%** | 92.26% | 7,827,786 | 1,338 s | measured |
| ResNet-18 scratch, 32×32 | pending | pending | 11.18M | pending | benchmark stage |
| ResNet-18 linear probe | pending | pending | 5,130 trainable | pending | benchmark stage |
| ResNet-18 fine-tune | pending | pending | 11.18M | pending | benchmark stage |
| Four core ablations | pending | pending | 482,730 | pending | benchmark stage |
| Optical degradation | pending | pending | 94,375 | pending | benchmark stage |
| Bring-your-own-data quickstart | contract smoke | — | tiny | ~3 s | complete |

## CIFAR-10 scratch baseline detail

| Field | Value |
|---|---:|
| Best validation accuracy | 92.78% (epoch 46) |
| Test accuracy 95% CI | 91.76–92.77% |
| Top-5 accuracy | 98.90% |
| NLL | 0.2894 |
| ECE | 0.0434 |
| Latency | 2.16 ms/image |
| Peak MPS memory | 181.5 MB |

Reference: 92.42% validation / 92.61% test. The reproduced test result is
−0.36 percentage points from the reference and above the ≥90% gate. No test-set
tuning was performed.

## Artifact policy

- `runs/` is gitignored; checkpoints and raw datasets are not committed.
- `results/<demo>/` holds committed summaries, figures, and reports.
- `docs/` links to committed results; selected figures are copied into
  `docs/images/` with byte-identical checks.
- Timings are measurements, not guarantees.

## Caveats

- The transfer, ablation, and optics numbers are produced in the benchmark
  stage; this page is updated from artifacts, not estimated from partial runs.
- Bootstrap intervals are per-run; they are not a paired significance test.
- MPS kernel behavior can vary across library versions; the environment is
  recorded with every run.
- The project is a modern, reproducible, well-diagnosed toolkit, not a
  state-of-the-art benchmark submission.
