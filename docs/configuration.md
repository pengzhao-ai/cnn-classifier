# Configuration

Every experiment is a typed YAML file. The schema is Pydantic v2 with unknown
keys rejected for known models, a `schema_version`, and CLI overrides checked
against the same schema.

## Precedence

1. Packaged defaults (`src/cnn_classifier/config/presets/default.yaml`).
2. The experiment's `defaults:` chain, in order.
3. The experiment file.
4. CLI `--set` overrides.

Mappings merge recursively; lists and scalars replace. Relative paths in
`data.root`, `data.manifest`, and `data.image_root` resolve against the
experiment file's directory, so a config can live beside its data.

## Minimal example

```yaml
schema_version: "1.0"

experiment:
  name: my_experiment
  seed: 7

data:
  adapter: image_folder
  root: data
  class_names: auto
  num_classes: 3
  channels: rgb
  image_size: [32, 32]
  split:
    mode: random_stratified
    val_fraction: 0.2
    seed: 7
  normalization:
    mode: compute

model:
  name: simple_cnn
  params:
    channels: [32, 64]
    dropout: 0.2

training:
  epochs: 10
  batch_size: 32
  optimizer:
    name: adamw
    lr: 1.0e-3
    weight_decay: 5.0e-4
  scheduler:
    name: cosine

runtime:
  device: auto
```

## Top-level sections

| Section | Key fields | Purpose |
|---|---|---|
| `experiment` | `name`, `seed`, `tags`, `notes` | identity and reproducibility |
| `data` | `adapter`, `root`, `manifest`, `class_names`, `num_classes`, `channels`, `image_size`, `split`, `normalization`, `transforms`, `adapter_params` | dataset contract |
| `model` | `name`, `params`, `transfer` | architecture and transfer controls |
| `training` | `epochs`, `batch_size`, `optimizer`, `scheduler`, `loss`, `early_stopping`, `checkpoint`, `amp`, `grad_clip_norm`, MixUp/CutMix, EMA | optimization budget |
| `evaluation` | `metrics`, `bootstrap`, `calibration`, `latency`, `error_analysis`, `tta` | evaluation protocol |
| `tracking` | `console`, `csv`, `tensorboard` | run tracking |
| `report` | `enabled`, `formats`, `figures` | artifact generation |
| `runtime` | `device`, `deterministic`, `num_workers`, `pin_memory`, `compile` | execution policy |

## Overrides

```bash
cnncls train -c config.yaml --set training.epochs=5 --set training.batch_size=64
```

Overrides use dotted paths and YAML scalar parsing. Unknown paths fail loudly;
they are not silently added.

## Validation

`validate_config_semantics` rejects cross-field combinations that would fail
midway through a run, including:

- warmup epochs ≥ total epochs;
- `min_lr_ratio` outside `[0, 1)`;
- invalid `val_fraction` for split modes that use it;
- class-name length mismatch with `num_classes`;
- supplied normalization with the wrong channel count;
- `fp16` AMP on CPU or MPS;
- invalid EMA decay, bootstrap resampling, or latency settings.

Adapter-specific options are validated by the adapter, for example
`max_train_samples` / `max_val_samples` for the torchvision adapter.

## Device policy

`runtime.device: auto` resolves CUDA → MPS → CPU. An explicit `cuda` or `mps`
request is rejected when that backend is unavailable, so a run never silently
changes hardware. AMP `auto` enables CUDA fp16 only; MPS and CPU run without
autocast in this release.

## Data contracts

The configuration describes the raw data; the adapter produces the model
contract. The two are linked by:

- `data.channels`: `rgb`, `gray`, or `as_is`;
- `data.image_size`;
- `data.normalization`: computed, supplied, or none;
- `data.value_range`: automatic or fixed for scientific images.

`inspect-data` and `train --dry-run` print the resulting contract, and the run
directory stores it in `data/model_contract.json`. See
[Bring your own data](bring_your_own_data.md) for the full table.

## Where to go next

- [Models & architectures](models.md) for `model.params`.
- [Techniques](techniques/augmentation.md) for method-level guidance.
- [API](api.md) for the typed schema reference.
