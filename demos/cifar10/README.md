# CIFAR-10 scratch baseline

## Question

How well can a small residual CNN trained from scratch do on CIFAR-10 with a
plain, reproducible recipe — and does the library reproduce the reference
result without special-case code?

This is the reference baseline for the project. It is deliberately a standard
recipe, not a state-of-the-art submission: the point is a trustworthy,
config-driven training pipeline that reports its own behavior honestly.

## Data

CIFAR-10 is downloaded through the `torchvision` adapter on first use:

- 50,000 training images, split 45,000 train / 5,000 validation with a plain
  random split (seed 42);
- 10,000 test images, touched only by the explicit evaluation step after
  training;
- 32×32 RGB images, supplied CIFAR mean/std normalization.

The `data/` directory is gitignored; the first run downloads about 170 MB.

## Configuration

`configs/cifar10/scratch_baseline.yaml` is the complete recipe:

| Component | Value |
|---|---|
| Model | `residual_cnn`, 3→64 stem, four residual stages, widths 64/128/256/512, GAP, dropout 0.3 |
| Parameters | 7,827,786 |
| Augmentation | horizontal flip, random crop (32, padding 4), mild color jitter, random erasing |
| Optimizer | Adam, lr 1e-3, weight decay 5e-4 |
| Schedule | cosine, no warmup, no gradient clipping |
| Budget | 50 epochs, batch 128, early stopping patience 10 |
| Device | `auto`: CUDA → MPS → CPU |

The architecture is a preset expressed through the generic `residual_cnn`
builder, not a CIFAR-specific model class.

## Run

From an activated environment at the repository root:

```bash
python demos/cifar10/scripts/run_scratch.py
```

Quick smoke check (two epochs, bootstrap and latency measurement disabled):

```bash
python demos/cifar10/scripts/run_scratch.py --quick
```

With a two-epoch cosine schedule the final epoch runs at learning rate zero, so
the quick run is a pipeline smoke check, not a meaningful accuracy result. The
script prints the run directory and the report path. Run artifacts stay under
`runs/` (gitignored). The full 50-epoch run is a Task 28 benchmark stage; do not
expect its numbers from the quick mode.

You can also drive the same config through the CLI:

```bash
cnncls train -c configs/cifar10/scratch_baseline.yaml
cnncls train -c configs/cifar10/scratch_baseline.yaml --dry-run
```

## Results

Full-run results, figures, and the HTML report are committed under
`results/cifar10/` after the benchmark stage. `demos/cifar10/results.md` records
the environment, seed, device, and any difference from the reference.

Reference from the design document, to be reproduced rather than assumed:

| Split | Reference accuracy |
|---|---|
| Validation | 92.42% |
| Test | 92.61% |

The acceptance gate is **≥90% test accuracy**. The target is the 92–93% range.
If the reproduced number differs, the result is reported as measured; the test
split is never used for model selection or tuning.

## What to notice

- **One YAML file describes the whole experiment.** Data, split, augmentation,
  model, optimizer, schedule, evaluation, and reporting are all typed and
  validated.
- **The test split is isolated.** `Trainer.fit()` uses only train/validation;
  test is evaluated once, from the best checkpoint, after training.
- **The run directory is self-describing.** It contains the resolved config,
  environment and git state, data fingerprint, model contract, checkpoints,
  metrics, predictions, figures, and an HTML/Markdown report.
- **The report is honest about provenance.** It records the device, seed, and
  data fingerprint so a reviewer can compare runs instead of trusting a number.
