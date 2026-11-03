# Transfer learning

## What

The torchvision wrapper supports a replaced classifier head, three stem
adaptations, freeze modes, BatchNorm policy, and regex parameter groups. The
sharp-v1 transfer benchmark uses three 32×32 ResNet-18 configs: scratch, linear
probe, and fine-tune.

## Why

Transfer learning is only a fair comparison if the architecture, data, and
evaluation are identical and the pretraining is the only changed variable. The
toolkit makes that structural: all three configs use the same split,
augmentation family, epoch budget, early stopping, and test evaluation. The
stem is adapted the same way in all three.

## Config keys

| Knob | Config key | Notes |
|---|---|---|
| Weights | `model.transfer.weights` | `null` or a torchvision weights name |
| Stem | `model.params.stem_adaptation` | `none`, `keep_7x7_stride1`, `adapt_3x3` |
| Freeze | `model.transfer.freeze` | `none`, `backbone`, `all_but_head` |
| BN policy | `model.transfer.train_bn` | train BN parameters while frozen |
| Differential LR | `model.transfer.param_groups` | pattern, multipliers |
| Head dropout | `model.params.dropout` | applied in the replacement head |

## Implementation

`models.torchvision` builds the backbone and replaces the classifier.
`models.transfer.apply_stem_adaptation` changes the stem; `apply_transfer`
freezes parameters and applies the BN policy; `build_param_groups` builds
disjoint groups and rejects patterns that match nothing. A frozen `adapt_3x3`
stem is rejected because the random stem would block the pretrained features.

## Evidence

- The linear-probe config has exactly **5,130** trainable parameters before
  training (the 512→10 head and bias).
- All three configs use `keep_7x7_stride1`, ImageNet normalization, the same
  45k/5k random split, 30 epochs, and validation-based early stopping.
- Measured test accuracy: fine-tune **94.98%**, scratch 90.88%, linear probe
  62.66%. All three share the same test fingerprint and have per-run bootstrap
  intervals in the comparison.
- The reference scratch baseline uses CIFAR mean/std; the transfer anchor uses
  ImageNet mean/std because the pretrained model expects it. The comparison
  labels that difference rather than hiding it.
- BatchNorm running statistics update even while probe parameters are frozen,
  so the linear probe is a frozen-parameter probe rather than a strict
  frozen-feature probe.

## Caveat

The 32×32 protocol is one compute budget, not a claim that linear probes are
always best. The 224×224 tracks and budget-sensitivity sweep are deferred to
v1.1. The reference baseline and the transfer anchor are separate experiments;
they should not be conflated.
