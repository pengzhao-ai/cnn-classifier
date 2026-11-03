# Optimization & LR

## What

The toolkit supports Adam, AdamW, and SGD; constant, cosine, cosine-with-warmup,
step, and plateau schedules; gradient clipping; gradient accumulation; and
device-aware AMP. Differential parameter groups give a backbone and head
different learning rates.

## Why

Optimizer and schedule choices interact with batch size, augmentation, and
model family. Hard-coding them hides the interaction. A config-driven optimizer
and scheduler make the budget explicit and let a fairness comparison hold them
constant.

## Config keys

| Knob | Config key | Notes |
|---|---|---|
| Optimizer | `training.optimizer.name` | `adam`, `adamw`, `sgd` |
| Base LR | `training.optimizer.lr` | positive |
| Weight decay | `training.optimizer.weight_decay` | non-negative |
| Momentum | `training.optimizer.momentum` | SGD |
| Schedule | `training.scheduler.name` | `constant`, `cosine`, `cosine_warmup`, `step`, `plateau` |
| Warmup | `training.scheduler.warmup_epochs` | must be < epochs |
| Minimum LR ratio | `training.scheduler.min_lr_ratio` | `[0, 1)` |
| Grad clipping | `training.grad_clip_norm` | `null` disables |
| Accumulation | `training.grad_accum_steps` | effective batch multiplier |
| AMP | `training.amp` | `auto`, `off`, `fp16`, `bf16` |

Parameter groups use regexes:

```yaml
transfer:
  param_groups:
    - pattern: '^fc\.'
      lr_multiplier: 10.0
```

## Implementation

`training.optim` builds the optimizer and differential groups; `training.
schedulers` builds the schedule; `training.amp` resolves AMP against the actual
device. `auto` enables CUDA fp16 only; MPS and CPU run without autocast in this
release. Gradient accumulation divides the loss by the accumulation steps, and
clipping happens before the optimizer step.

## Evidence

- The CIFAR baseline uses Adam, lr 1e-3, weight decay 5e-4, cosine, no clipping,
  50 epochs, and reaches 92.25% test accuracy.
- The transfer anchor uses AdamW; the fine-tune run uses 1e-4 for the backbone
  and 1e-3 for the head via a regex group.
- On the reduced 8-epoch ablation budget, cosine reached 67.92% test accuracy
  versus 59.44% for a constant schedule. This is one seed and a short budget,
  not a universal schedule result.

## Caveat

A very short cosine schedule reaches zero learning rate on its final epoch; a
two-epoch smoke run is not a meaningful schedule result. Warmup and clipping are
sensitivity knobs, not defaults that are always better. The reported timings
are measurements on one MPS machine.
