# CIFAR-10 core ablations

## Question

Which of four basic design choices matters under a fixed, reduced CIFAR-10
budget: augmentation, BatchNorm, residual connections, and the learning-rate
schedule?

This is a controlled comparison, not a leaderboard. Each pair changes exactly
one knob and holds the data, model shape, optimizer, seed, and epoch budget
constant. The goal is to show a trustworthy ablation workflow and honest
paired differences, not state-of-the-art accuracy.

## Data and budget

- CIFAR-10 with a deterministic reduced split: **10,000 train / 2,000
  validation**, full 10,000-image test set.
- Seed 42, plain random train/validation split.
- 8 epochs, batch 128, AdamW lr 1e-3, weight decay 5e-4.
- Residual CNN preset: 3→32 stem, widths 32/64/128, one block per stage,
  dropout 0.2.
- Device auto: CUDA → MPS → CPU.

The subset is selected by the torchvision adapter's `max_train_samples` and
`max_val_samples` options, so all eight runs see exactly the same samples.

## Pairs

| Pair | Knob on | Knob off | Configs |
|---|---|---|---|
| Augmentation | random crop + horizontal flip | tensor + normalize only | `augmentation.yaml` / `none_augmentation.yaml` |
| BatchNorm | `norm: batch` | `norm: none` | `batchnorm.yaml` / `no_batchnorm.yaml` |
| Residual connections | `residual: true` | `residual: false` | `residual.yaml` / `no_residual.yaml` |
| LR schedule | cosine | constant | `cosine.yaml` / `constant.yaml` |

## Run

From an activated environment at the repository root:

```bash
python demos/ablations/run_ablations.py
```

The runner trains all eight configs first with test evaluation disabled, then
evaluates the test split for every run, and finally writes:

```text
results/ablations/
├── comparison.md
├── comparison.json
└── comparison.png
```

Run artifacts stay under `runs/` (gitignored). The full eight-run suite has
been completed; committed summaries, figures, and reports are in
`results/ablations/`, with the cross-pair comparison in
`results/comparison/ablation_comparison.md`.

## Results

Full results, per-run reports, and committed summaries are in
`results/ablations/` and described in `demos/ablations/results.md`.

The comparison reports per-run accuracy, macro F1, parameter count, and latency,
plus the per-run 95% bootstrap interval recorded in each `summary.json`. No
paired significance claim is made without a paired test; overlapping intervals
are not evidence of no effect.

## What to notice

- **One knob per pair.** The shared `_base.yaml` makes that structural rather
  than a promise in prose.
- **Same data, same seed, same budget.** The subset limits are part of the data
  fingerprint recorded in every run.
- **Test discipline.** Test evaluation happens only after all training runs
  finish.
- **Compute is recorded.** `summary.json` and `comparison.json` carry training
  time, device, and parameter counts so differences can be read against cost.
