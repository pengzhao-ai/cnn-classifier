# Bring your own data — quickstart

## Question

What do I actually have to provide, and what does the model actually receive?

This demo is a real, runnable answer. It uses a committed 3-class, 60-image
16-bit grayscale fixture, so it needs no network, GPU, or license approval. It
runs `inspect-data`, `train`, `evaluate`, and `report` in a few seconds.

## What you provide

```text
demos/bring_your_own_data/
├── fixture/
│   ├── train/
│   │   ├── circle/     # 15 images
│   │   ├── square/     # 15 images
│   │   └── stripe/     # 15 images
│   └── val/
│       ├── circle/     # 5 images
│       ├── square/     # 5 images
│       └── stripe/     # 5 images
├── manifest.csv        # path,label,split,group
└── config.yaml
```

The images are 16×16, 16-bit grayscale PNGs with values up to 65,535. The
manifest makes the split and group explicit:

```csv
path,label,split,group
train/circle/circle_00.png,circle,train,circle_00
...
```

Supported inputs through the same adapters:

| Item | Options |
|---|---|
| Bit depth | 8-bit RGB/grayscale; 16-bit grayscale PNG; 16-bit RGB/grayscale TIFF |
| Labels | folder names or manifest `label` |
| Split | folder names or manifest `split` |
| Group | optional manifest `group` for leakage-safe splitting |
| Class order | adapter order or explicit `data.class_names` |
| Value range | automatic or fixed |

## What the model receives

| Contract field | Value in this demo |
|---|---|
| `dtype` | `float32` |
| One sample | `[1, 16, 16]` — channels-first, grayscale |
| One batch | `[B, 1, 16, 16]` |
| Value range | `[0.0, 1.0]` |
| Channel policy | `gray` |
| Label map | `circle: 0`, `square: 1`, `stripe: 2` |
| Normalization | mean/std computed from the training split |
| Batch size | 8 |

The same information is written to `data/model_contract.json`, printed by
`inspect-data` and `train --dry-run`, and embedded in the HTML report.

## Run

From an activated environment at the repository root:

```bash
python demos/bring_your_own_data/scripts/run_quickstart.py
```

Expected runtime is a few seconds on CPU or MPS. The script prints the run
directory and `report.html` path.

Equivalent CLI steps:

```bash
cnncls inspect-data -c demos/bring_your_own_data/config.yaml
cnncls train -c demos/bring_your_own_data/config.yaml
cnncls evaluate --checkpoint runs/<run>/checkpoints/best.pt --split val
cnncls report --run runs/<run>
```

## Registering a custom adapter

The demo uses the built-in `manifest` adapter. A genuinely custom format can be
added in the demo module without editing `src/cnn_classifier`:

```python
from cnn_classifier.core.types import DatasetBundle
from cnn_classifier.data.adapters import register_dataset


@register_dataset("my_format")
class MyFormatAdapter:
    @staticmethod
    def build(config) -> DatasetBundle:
        # Read your files, build datasets, and return the same bundle contract.
        ...
```

Any `data.adapter: my_format` config can then use it, provided the module is
imported before the run. This is the extension path, not a framework rewrite.

## What to notice

- **The contract is explicit.** The report shows the raw input, the label map,
  the tensor shape, and the normalization actually used.
- **16-bit data is not silently truncated.** The reader scales the full
  16-bit range to `[0, 1]` and records that policy.
- **The split is explicit.** The manifest supplies train/validation rows and a
  group column for leakage checks.
- **The same trainer handles a new dataset.** Only the config and data change.
