# Augmentation & regularization

## What

The training pipeline can apply torchvision v2 transforms, MixUp and CutMix,
label smoothing, dropout, weight decay, and early stopping. All of them are
declared in YAML; the training loop receives tensors and never builds a
transform.

## Why

Augmentation and regularization are the difference between memorizing a small
training set and learning a decision boundary that transfers. They are also
easy to apply inconsistently. Declaring them in one place, and recording the
resolved config in every run, makes experiments comparable.

## Config keys

| Knob | Config key | Notes |
|---|---|---|
| Train transforms | `data.transforms.train` | ordered list of named transforms |
| Eval transforms | `data.transforms.eval` | usually resize, tensor, normalize |
| MixUp | `training.mixup_alpha` | `0.0` disables |
| CutMix | `training.cutmix_alpha` | `0.0` disables |
| Label smoothing | `training.loss.label_smoothing` | `[0, 1)` |
| Dropout | `model.params.dropout` | model-specific |
| Weight decay | `training.optimizer.weight_decay` | optimizer-specific |
| Early stopping | `training.early_stopping` | monitor, mode, patience |

Supported transform names include `RandomHorizontalFlip`, `RandomVerticalFlip`,
`RandomRotation`, `RandomCrop`, `RandomResizedCrop`, `ColorJitter`,
`GaussianBlur`, `RandAugment`, `AutoAugment`, `RandomErasing`, `ToTensor`, and
`Normalize`. Normalization can use `auto` to take the run's statistics.

## Implementation

`data.transforms` builds a callable per sample from the ordered config list.
`training.losses` handles label smoothing and MixUp/CutMix; `training.callbacks`
handles early stopping. The reference CIFAR baseline uses horizontal flip,
random crop with padding 4, mild color jitter, and random erasing.

## Evidence

- The CIFAR-10 scratch baseline reaches 92.25% test accuracy with the reference
  augmentation family (see [Results](../results.md)).
- On the reduced 8-epoch ablation budget, augmentation **off** reached 71.06%
  test accuracy versus 67.92% with the crop/flip pipeline. That is a measured
  outcome on one seed and a small budget, not a general recommendation.
- The optical degradation demo deliberately does **not** use flips or crops:
  motion-blur direction and vignetting position are part of the label.

## Caveat

More augmentation is not automatically better, and it can destroy
orientation-sensitive labels. The ablations report paired differences on one
reduced budget, not universal recommendations. The CIFAR baseline is a
reproduction, not a state-of-the-art result.
