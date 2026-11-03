# `cnn-classifier` — Design Document

**Status:** Implemented; v0.1.0 released 2026-11-03
**Date:** 2026-11-02
**Repository:** `CNN_Classifier`
**Distribution:** `cnn-classifier-toolkit`
**Import package:** `cnn_classifier`
**CLI:** `cnncls`
**License:** Apache-2.0

> **Post-release note (2026-11-03):** The distribution was released as
> `cnn-classifier-toolkit`. PyPI rejected the originally planned
> `cnn-classifier` because its name-similarity rule normalizes it to the same
> value as the existing `cnnClassifier` project. The import package and CLI are
> unchanged.

---

## 1. Intent

`cnn-classifier` is a reusable, configurable, modular PyTorch codebase for image
classification. It exists for two reasons:

1. **A working toolbox.** A new image-classification project should be mostly a
   dataset adapter plus a YAML configuration. The user should not rewrite the
   training loop, evaluation code, checkpoint handling, or plots for every new
   dataset.
2. **An interview artifact.** A reviewer should be able to see, in ten minutes,
   that the author understands the complete training pipeline: data handling,
   augmentation, architecture design, regularization, optimization, learning-rate
   control, transfer learning, evaluation, failure analysis, reproducibility, and
   the engineering required to make those things reusable.

The codebase is derived conceptually from days 15–21 of `DL_learning`, with the
day 21 notebook as the completed reference behavior, day 29's `Trainer` as an
early sketch, and day 30's degradation-classification capstone as the seed of the
second demo. It is not a notebook dump. The notebook arc is the specification of
what must be possible; the library is the product.

The project follows the operating conventions of `IQ_machine_vision`: `uv` for
Python and dependencies, a `src/` layout, Pydantic-validated configuration,
strict typing, a real test suite, documentation published with MkDocs Material,
committed result artifacts, and a tag-driven release process.

---

## 1.1 Sharp v1 scope

The approved v1 is deliberately narrower than the full design:

**Included**

- core config, data, models, training, evaluation, diagnostics, and reporting;
- CIFAR-10 scratch baseline;
- three 32×32 transfer configs: scratch ResNet-18, linear probe, and fine-tune;
- optical degradation demo;
- bring-your-own-data quickstart;
- four core ablations: augmentation, BatchNorm, residual connections, and LR
  schedule;
- tests, documentation, MkDocs site, PyPI/Pages workflows.

**Deferred to v1.1**

- the two 224×224 transfer tracks and the budget-sensitivity sweep;
- optimizer and label-smoothing ablations;
- ONNX export, sweep runner, profiler, and optional VisualTorch/TorchView
  architecture backends;
- advanced diagnostic plots beyond the core diagnostics.

The core library is built and reviewed before the demos. This keeps the
interfaces stable and avoids debugging the framework through a full benchmark.

---

## 2. Goals

### G1 — Reusable classifier toolbox

- A new dataset can be trained by supplying a YAML config and one of the built-in
  adapters (`torchvision`, `image_folder`, `manifest`) without editing library
  code.
- A genuinely new data format can be supported by implementing a small
  `DatasetAdapter` and registering it.
- A new architecture can be supported by implementing a model builder and
  registering it.
- The training engine is usable as a library, not only through the CLI.
- The visualization and reporting layers consume run artifacts; training does
  not depend on plotting code.

### G2 — Configurable experiments

Every meaningful knob is configuration, not a hard-coded constant:

- data root, split policy, image size, channels, normalization, class names;
- augmentation pipeline, including optional RandAugment, AutoAugment, MixUp,
  CutMix, and Random Erasing;
- model family, depth, widths, normalization, dropout, residual connections,
  pretrained weights, frozen layers, and differential learning rates;
- optimizer, weight decay, momentum, label smoothing, gradient clipping,
  gradient accumulation, AMP, channels-last, and compilation;
- learning-rate scheduler, warmup, cosine/step/plateau/one-cycle, and per-group
  multipliers;
- early stopping, checkpoint selection, EMA, and resume behavior;
- evaluation metrics, calibration, bootstrap confidence intervals, latency
  measurement, and test-time augmentation;
- device policy, worker policy, determinism, and logging backends.

### G3 — Honest benchmarking

- The CIFAR-10 from-scratch result (reference baseline: 92.61% test) and the
  ImageNet-pretrained ResNet-18 result (day 20 reference: 96.06% test) are
  reproduced under a controlled protocol.
- The comparison reports the protocol, compute cost, parameter count, trainable
  parameter count, confidence interval, and inference latency.
- The 224×224 upsample recipe is deferred to v1.1. When added, it is reported as
  a separate compute-heavy track, never mixed into the fairness anchor.
- The test split is evaluated once, after model selection. Training never
  optimizes against the test split.
- Run comparison refuses to compare runs whose test-set fingerprints differ.

### G4 — Diagnostics and publication-quality visualization

- One consistent visual language across all figures: colorblind-safe palette,
  no chartjunk, SVG and PNG output, print-ready DPI.
- Training diagnostics: loss, accuracy, learning rate, gradient norm, best epoch,
  early-stopping marker, train/validation gap.
- Evaluation diagnostics: confusion matrix, per-class precision/recall/F1,
  calibration/reliability diagram, confidence analysis, most-confident errors,
  misclassification gallery.
- Model diagnostics: parameter/MAC summary, layer-shape table, architecture
  diagram, first-layer filters, feature maps, activation statistics, dead-unit
  analysis, and Grad-CAM.
- A self-contained HTML experiment report generated from artifacts, plus a
  multi-run comparison report.
- The visualization code is tested by smoke tests and headless rendering, not
  by pixel-perfect image comparison.

### G5 — Two strong demos and a quickstart

1. **CIFAR-10 classification.** The reference residual CNN, reproduced through the library,
   plus a fair transfer-learning benchmark against ResNet-18 under matched
   protocols.
2. **Optical degradation classification.** A manifest-based dataset generated
   from CIFAR-10 base images, with physically motivated degradations and a
   group-aware split. It uses the same model, training, evaluation, and reporting
   infrastructure as the CIFAR-10 demo.
3. **Bring-your-own-data quickstart.** A tiny local fixture that runs
   inspect/train/evaluate/report in under a minute and explicitly contrasts the
   raw data contract with the tensor/label shape the model receives.

An ablation suite demonstrates the effect of augmentation, BatchNorm, residual
connections, and learning-rate schedule under a reduced budget.

### G6 — Publishable engineering

- Python 3.11–3.13, `uv`, committed `uv.lock`, `hatchling`, `ruff`, `mypy`,
  `pytest`, pre-commit, GitHub Actions.
- Latest stable PyTorch and TorchVision at implementation time; the verified
  current versions on 2026-11-02 are PyTorch 2.15.0 and TorchVision 0.29.2.
  Exact versions are pinned in `uv.lock`; `pyproject.toml` carries compatible
  lower bounds.
- `src/` layout; demos, docs, and tests outside the package.
- MkDocs Material documentation with API reference, concepts, configuration,
  tutorials, benchmark results, and release policy.
- GitHub Pages deployment from `main`; tag-driven PyPI release with trusted
  publishing; changelog, citation metadata, contribution guide, code of conduct,
  third-party notices.
- No `notebooks/` or `examples/` directories in v1. Demos and documentation
  are the canonical runnable examples.

---

## 3. Non-goals (v1)

- Full ImageNet training or evaluation. It is too large for the target hardware,
  CI, and demo runtime, and evaluating on the pretraining dataset is not a
  transfer-learning benchmark.
- Object detection, segmentation, generative models, self-supervised pretraining,
  or video classification.
- Non-CNN model families as first-class built-ins. MLP and VGG are not v1; the
  core is `simple_cnn`, `residual_cnn`, and torchvision CNN wrappers. Other
  architectures can be added through the registry.
- Distributed training, multi-node orchestration, or cloud job submission.
- Hyperparameter optimization as a first-class service. A lightweight sweep
  runner may follow after v1; Ray Tune/Optuna integration is deferred.
- The 224×224 transfer tracks, budget-sensitivity sweep, ONNX export, profiler,
  and optional VisualTorch/TorchView backends in sharp v1. They are v1.1 work.
- Model serving, REST APIs, or mobile deployment. ONNX export is a stretch goal.
- Custom CUDA kernels or low-level performance engineering.
- Reimplementing torchvision architectures or pretrained weights.
- Claiming state-of-the-art accuracy. The target is a correct, reproducible,
  well-diagnosed baseline and a useful toolbox.
- Supporting every image format or scientific array layout. v1 covers 8-bit
  RGB/grayscale, 16-bit grayscale/RGB TIFF/PNG through an explicit reader,
  ImageFolder, manifest CSV, and NumPy arrays; exotic formats use a custom
  adapter.

---

## 4. Anti-goals (failure even if the code runs)

- A repository that is only a thin wrapper around one CIFAR-10 script.
- Hard-coded CIFAR-10 class counts, image sizes, normalization values, or paths
  outside demo configs.
- A training loop that cannot be resumed, tested on a tiny dataset, or used
  without matplotlib installed.
- A config system that validates syntax but silently ignores a misspelled knob.
- A benchmark that changes the split, augmentation, or evaluation protocol
  between models and then compares the final number.
- Evaluating the test set during hyperparameter selection.
- Data leakage between train and validation, especially in the degradation demo
  where one base image produces several samples.
- A "fair" comparison that mixes 32×32 native training with 224×224 upsampling
  without labelling the compute difference.
- Publication figures that depend on a notebook session, a system Graphviz
  install, or a non-deterministic plotting order.
- Tests that require network access, a GPU, or more than a few minutes.
- Reproducibility claims that ignore device, PyTorch version, or nondeterministic
  MPS kernels.

---

## 5. Users and success scenarios

### Primary user: the author

> I have a folder of images and labels. I want to train a classifier, compare two
> architectures, and produce a report without rewriting the pipeline.

Success: copy a preset config, point it at the data, run `cnncls train`, inspect
the run directory and HTML report.

### Secondary user: an interviewer or reviewer

> I have ten minutes. Show me that this person understands model training and can
> engineer it.

Success: the README shows the benchmark, the docs explain the design, the code is
typed and tested, the reports show learning curves and failure analysis, and the
comparison is honest about compute and protocol.

### Tertiary user: a collaborator

> I have a new architecture or dataset format. I want to plug it in.

Success: implement one registry entry against a documented protocol; all existing
training, evaluation, and reporting functionality works.

### Quaternary user: the author in an optics context

> I have optical captures or simulated images and want to classify degradation,
> aberration, or quality regime.

Success: the manifest adapter reads a CSV, the group split prevents leakage, and
the report shows which degradations are confused.

---

## 6. Design principles

1. **Configuration is the interface.** Experiments are data, not code. A config
   is validated, resolved, hashed, and stored with the run.
2. **Registries over conditionals.** Datasets, models, transforms, optimizers,
   schedulers, losses, metrics, and trackers register themselves by name. The
   config refers to names; no growing `if name == ...` chain.
3. **Headless training.** The training package has no dependency on matplotlib,
   Jinja2, or the CLI. Reports are built from artifacts after training.
4. **Artifacts are the contract.** Metrics, predictions, checkpoints, figures,
   and reports are files with documented schemas. A run can be inspected without
   rerunning it.
5. **Honest comparisons.** Same split, same test fingerprint, same evaluation
   code, explicit compute accounting, confidence intervals.
6. **Pure PyTorch.** The training loop is visible and owned by the project. No
   Lightning or hidden trainer. This is both a pedagogical and an interview
   requirement.
7. **Typed and testable.** Pydantic configs and strict mypy at the boundaries;
   small protocols for extension points; deterministic fixtures for tests.
8. **Graceful optional dependencies.** Core training installs without Graphviz,
   TensorBoard, VisualTorch, or scikit-learn. Visualization degrades to a
   documented fallback rather than crashing.
9. **Device honesty.** `auto` chooses CUDA, then MPS, then CPU, but features that
   are not reliable on a device are disabled by default and documented.
10. **One visual language.** Every figure uses the same style module and the same
    colorblind-safe palette.

---

## 7. Architecture

### 7.1 Repository layout

```text
CNN_Classifier/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── NOTICE / THIRD_PARTY.md
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── Makefile
├── mkdocs.yml
├── .pre-commit-config.yaml
├── .github/workflows/{ci,docs,release}.yml
├── configs/
│   ├── base/
│   ├── cifar10/
│   ├── optics_degradation/
│   ├── ablations/
│   └── bring_your_own_data/
├── demos/
│   ├── cifar10/               # scratch baseline + transfer benchmark
│   ├── optics_degradation/
│   ├── ablations/
│   └── bring_your_own_data/   # minimal manifest/ImageFolder quickstart
├── docs/
│   ├── index.md
│   ├── quickstart.md
│   ├── concepts.md
│   ├── architecture.md
│   ├── configuration.md
│   ├── data.md
│   ├── models.md
│   ├── training.md
│   ├── evaluation.md
│   ├── visualization.md
│   ├── reproducibility.md
│   ├── benchmarks.md
│   ├── case_studies/{cifar10.md,optics_degradation.md}
│   ├── bring_your_own_data.md
│   ├── extending.md
│   ├── api.md
│   ├── release.md
│   ├── images/
│   └── superpowers/specs/2026-11-02-cnn-classifier-design.md
├── results/
│   ├── cifar10/
│   ├── optics_degradation/
│   ├── ablations/
│   └── comparison/
├── src/cnn_classifier/
│   ├── __init__.py
│   ├── _version.py
│   ├── cli.py
│   ├── config/
│   ├── core/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── tracking/
│   ├── diagnostics/
│   ├── viz/
│   └── runs/
└── tests/
```

### 7.2 Package modules

```text
src/cnn_classifier/
├── __init__.py          # lazy public exports; cheap import
├── _version.py          # single version source
├── cli.py               # Typer app: train/evaluate/predict/inspect/debug/report/compare/doctor
├── core/
│   ├── registry.py      # generic name -> factory registry
│   ├── device.py        # auto CUDA/MPS/CPU resolution and capability report
│   ├── seeding.py       # Python/NumPy/PyTorch/worker seeds
│   ├── types.py         # DatasetBundle, ClassInfo, ModelInputContract, EpochMetrics, RunSummary, SplitInfo
│   ├── hashing.py       # config/data/model fingerprints
│   └── errors.py        # ConfigError, DataError, RegistryError, RunError
├── config/
│   ├── schema.py        # Pydantic v2 models, extra="forbid", schema_version
│   ├── loader.py        # YAML load, defaults merge, --set overrides
│   ├── resolve.py       # path resolution, registry validation, derived values
│   └── presets/         # package-shipped base configs and templates
├── data/
│   ├── base.py          # DatasetAdapter protocol, DatasetBundle
│   ├── adapters/
│   │   ├── torchvision.py
│   │   ├── image_folder.py
│   │   ├── manifest.py
│   │   ├── numpy.py
│   │   └── synthetic.py
│   ├── manifest.py      # CSV parsing and validation
│   ├── splits.py        # explicit, stratified, group-aware, deterministic
│   ├── transforms.py    # torchvision.transforms.v2 builder and custom registry
│   ├── image_io.py      # 8-bit and 16-bit readers, channel/dtype policy
│   ├── stats.py         # streaming mean/std, cache, fingerprint
│   └── inspection.py    # balance, corruption, duplicates, size distribution, leakage
├── models/
│   ├── base.py          # ModelBuilder protocol, ModelSpec
│   ├── simple_cnn.py
│   ├── residual_cnn.py  # configurable residual CNN
│   ├── torchvision.py   # torchvision wrapper, stem adaptation, freeze modes
│   └── summary.py       # parameter counts, layer table, MACs, receptive field
├── training/
│   ├── trainer.py       # epoch loop, train/validation, resume
│   ├── state.py         # TrainingState
│   ├── callbacks.py     # EarlyStopping, ModelCheckpoint, GradientMonitor, EMA, Unfreeze
│   ├── optim.py         # optimizer registry and param-group construction
│   ├── schedulers.py    # scheduler registry and warmup
│   ├── losses.py        # loss registry, label smoothing, MixUp/CutMix
│   ├── amp.py           # device-aware autocast/scaler policy
│   └── checkpoint.py    # save/load last/best, RNG state
├── evaluation/
│   ├── evaluator.py     # inference over a split, predictions, metrics
│   ├── metrics.py       # accuracy, top-k, per-class, confusion, F1
│   ├── calibration.py   # ECE/MCE, reliability data, Brier, NLL
│   ├── bootstrap.py     # confidence intervals, paired tests
│   ├── error_analysis.py# most-confident errors, confusion pairs
│   ├── latency.py       # warmup, synchronize, throughput, peak memory
│   ├── compare.py       # run loader, fingerprint check, leaderboard, Pareto
│   └── predict.py       # single image/folder/manifest inference
├── tracking/
│   ├── base.py          # Tracker protocol
│   ├── console.py
│   ├── csv.py
│   ├── tensorboard.py
│   └── composite.py
├── diagnostics/
│   ├── sanity.py        # one-batch overfit, shape trace, loss/grad checks
│   ├── gradients.py     # per-layer norms, vanishing/exploding
│   ├── activations.py   # activation stats, dead units
│   └── determinism.py   # repeat-run comparison
├── viz/
│   ├── style.py         # Okabe-Ito, rcParams, save_figure (PNG/SVG)
│   ├── data.py          # class distribution, samples, augmentation gallery
│   ├── training.py      # history dashboard, multi-run overlay
│   ├── evaluation.py    # confusion, per-class, calibration, errors, confidence
│   ├── features.py      # filters, feature maps, activations, Grad-CAM, embeddings
│   ├── architecture.py  # model card, torchinfo table
│   ├── compare.py       # accuracy/params/latency/Pareto
│   ├── report.py        # self-contained HTML and Markdown report
│   └── templates/report.html.j2
└── runs/
    ├── context.py       # run directory, metadata, atomic writes, status
    └── artifacts.py     # metrics.jsonl, summary.json, prediction archives
```

### 7.3 Dependency rules

| Layer | May depend on | Must not depend on |
|---|---|---|
| `core` | stdlib, PyTorch (device/seeding), Pydantic | data, models, training, viz, CLI |
| `config` | stdlib, Pydantic, PyYAML, core | torch model code, training, viz |
| `data` | core, config, torch, torchvision, NumPy, Pillow, tifffile | models, training, viz |
| `models` | core, config, torch, torchvision | data, training, viz |
| `training` | core, config, torch, models via protocols | viz, report, CLI |
| `evaluation` | core, config, torch, data types | training loop, viz, CLI |
| `tracking` | core, config, stdlib, optional TensorBoard | torch model internals |
| `runs` | core, config, stdlib | torch, viz |
| `diagnostics` | core, config, torch, data/model protocols | viz, CLI |
| `viz` | core, config, evaluation types, matplotlib, optional extras | CLI, training loop |
| `cli` | every layer; orchestration only | — |

Two rules are non-negotiable:

1. `training` must not import `viz`, `matplotlib`, or `jinja2`. A run can train on a
   headless machine with no plotting dependencies.
2. A report is generated from files in a run directory, not from live objects. This
   keeps reporting reproducible and lets a reviewer regenerate figures later.

### 7.4 Public API shape

The top-level package exposes a small, stable surface lazily:

```python
import cnn_classifier as cnn

run = cnn.train("configs/cifar10/scratch.yaml", overrides=["training.epochs=5"])
metrics = cnn.evaluate(run.checkpoint, split="test", config="configs/cifar10/scratch.yaml")
figures = cnn.report(run.directory)
```

The CLI is a thin wrapper around the same functions. The exact public names are
finalized in the implementation plan; no internal module is imported eagerly by
`cnn_classifier/__init__.py`.

---

## 8. Configuration

### 8.1 Format and precedence

- YAML files, validated by Pydantic v2 models with `extra="forbid"` and
  `schema_version`.
- Configuration inheritance via `defaults: [path, ...]`; deep merge, later files
  win; lists replace rather than merge.
- CLI overrides use `--set section.key=value`; values are parsed as YAML scalars
  and validated again.
- Precedence: packaged presets < `defaults` chain < experiment file < CLI
  overrides. Environment variables are reserved for machine paths and secrets,
  not experiment semantics.
- Paths are resolved relative to the config file that declares them; the resolved
  absolute config is written to the run directory.
- Unknown keys, unknown registry names, invalid ranges, and incompatible
  combinations fail before any data is loaded.

Hydra is deliberately not used. It is a good tool, but its working-directory
magic, config-group composition, and CLI syntax would make the library harder to
call programmatically and harder to test. The project gets the same benefits from
a small, explicit resolver.

### 8.2 Schema outline

```yaml
schema_version: "1.0"

defaults:
  - ../base/cifar10.yaml

experiment:
  name: cifar10_scratch_baseline
  tags: [cifar10, scratch, baseline]
  notes: "Reference baseline architecture and recipe."

data:
  adapter: cifar10
  root: ../../data
  class_names: auto
  num_classes: 10
  channels: 3
  image_size: [32, 32]
  split:
    mode: random             # reference baseline uses random_split, not stratified
    val_fraction: 0.1
    seed: 42
  normalization:
    mode: supplied           # compute | supplied | none
    mean: [0.4914, 0.4822, 0.4465]
    std: [0.2470, 0.2435, 0.2616]
    cache: true
  transforms:
    train:
      - {name: RandomHorizontalFlip, p: 0.5}
      - {name: RandomCrop, size: 32, padding: 4}
      - {name: ColorJitter, brightness: 0.1, contrast: 0.1}
      - {name: ToTensor}
      - {name: Normalize, mean: auto, std: auto}
      - {name: RandomErasing, p: 0.1}
    eval:
      - {name: ToTensor}
      - {name: Normalize, mean: auto, std: auto}

model:
  name: residual_cnn
  params:
    stem: {channels: 64, kernel_size: 3, stride: 1, pool: false}
    stages:
      - {channels: 64, conv: false, blocks: 1, downsample: maxpool}
      - {channels: 128, conv: true, blocks: 1, downsample: maxpool}
      - {channels: 256, conv: true, blocks: 1, downsample: maxpool}
      - {channels: 512, conv: true, blocks: 1, downsample: none}
    dropout: 0.3
    norm: batch
    residual: true
  transfer:
    weights: null
    freeze: none

training:
  epochs: 50
  batch_size: 128
  grad_accum_steps: 1
  selection_metric: val_accuracy
  optimizer:
    name: adam
    lr: 1.0e-3
    weight_decay: 5.0e-4
  scheduler:
    name: cosine
    warmup_epochs: 0
    min_lr_ratio: 0.0
    step_unit: epoch
  loss:
    name: cross_entropy
    label_smoothing: 0.0
  grad_clip_norm: null
  amp: auto
  mixup_alpha: 0.0
  cutmix_alpha: 0.0
  ema:
    enabled: false
    decay: 0.999
  early_stopping:
    enabled: true
    monitor: val_accuracy
    mode: max
    patience: 10
    min_delta: 0.0
  checkpoint:
    monitor: val_accuracy
    mode: max
    save_last: true

evaluation:
  metrics: [accuracy, top_k, macro_f1, per_class, confusion_matrix, ece, nll]
  top_k: 3
  bootstrap:
    enabled: true
    n_resamples: 1000
    confidence: 0.95
    stratify: true
  calibration:
    bins: 15
  latency:
    enabled: true
    warmup: 20
    iterations: 100
  error_analysis:
    max_examples: 32
  tta:
    enabled: false
    transforms: [identity, hflip]
    reduction: mean_probabilities

tracking:
  console: true
  csv: true
  tensorboard: true

report:
  enabled: true
  formats: [html, markdown]
  figures: [training_history, confusion_matrix, per_class, calibration, errors, model_summary]

runtime:
  device: auto
  deterministic: true
  num_workers: auto
  persistent_workers: true
  pin_memory: auto
  channels_last: false
  compile: false
```

The example above is the reference baseline recipe, not a generic "modern"
recipe. It exists to validate the library against a known result. The library API
remains generic: `residual_cnn` is a configurable builder, and the CIFAR preset
is demo data. A separate config family uses the torchvision ResNet wrapper and a
controlled protocol for ablations and transfer comparisons. The two must not be
conflated.

### 8.3 Regularization placement

Regularization knobs live where they act, to avoid one undifferentiated
"regularization" blob:

- augmentation: `data.transforms.train`;
- dropout and normalization: `model.params`;
- weight decay, momentum: `training.optimizer`;
- label smoothing: `training.loss`;
- MixUp/CutMix: `training`;
- early stopping: `training.early_stopping`;
- gradient clipping: `training.grad_clip_norm`.

### 8.4 Config resolution errors

Examples of rejected configurations:

- `training.optimizer.name: adamww` — unknown registry name;
- `model.params.dropout: 1.5` — outside `[0, 1)`;
- `data.channels: 3` with `data.image_size: [32]` — malformed size;
- `training.scheduler.warmup_epochs >= training.epochs` — warmup never ends;
- `training.amp: fp16` with `runtime.device: mps` — unsupported policy;
- `evaluation.bootstrap.n_resamples: 0` — no resamples;
- a dependency key misspelled, e.g. `trainig` — unknown top-level key.

---

## 9. Data layer

### 9.1 DatasetBundle contract

Every adapter returns:

```python
@dataclass(frozen=True)
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset | None
    class_names: tuple[str, ...]
    in_channels: int
    image_size: tuple[int, int]
    normalization: NormalizationStats | None
    split_info: SplitInfo
    fingerprint: str
    sample_metadata: SampleMetadata | None
    model_contract: ModelInputContract
```

`Dataset` items are `(image_tensor, label, index)` or a typed sample object with
an explicit collate function. The default is a three-tuple so error analysis can
recover the source path without re-reading the dataset. The exact sample type is
decided in the implementation plan; the important contract is that the index and
source path are recoverable.

### 9.2 Built-in adapters

| Adapter | Purpose | Key config |
|---|---|---|
| `torchvision` | CIFAR-10, MNIST, Fashion-MNIST, and other torchvision datasets | `name`, `root`, `split`, `download` |
| `image_folder` | `train/val/test` class directories | `root`, `split_dirs`, `class_map` |
| `manifest` | CSV with `path,label,split,group` (extra columns preserved) | `manifest`, `image_root`, `columns` |
| `numpy` | `.npy`/`.npz` arrays for scientific data | `array_path`, `labels_path`, `layout`, `dtype` |
| `synthetic` | Deterministic tiny shapes for tests | `num_samples`, `num_classes`, `image_size` |

`data.class_names` may be `auto` or an explicit ordered list. `auto` takes the
adapter's class mapping; an explicit list must have exactly `num_classes` entries
and must match the adapter's label indices, otherwise resolution fails. The
resolved list is written to `data/class_names.json` and used in every report.

Custom adapters implement the `DatasetAdapter` protocol and register with
`@register_dataset("name")`. The config's `data.adapter` is validated against the
registry before data loading.

### 9.3 Splits and leakage

Supported policies:

- `explicit`: a `split` column or directory split;
- `random`: deterministic random split, matching the reference notebook's
  split behavior;
- `random_stratified`: deterministic, class-balanced;
- `group`: split by a group column, with all samples from one group assigned to
  one split;
- `train_only`: no validation split; used only for final training, not for model
  selection.

The optics demo must use `group`, where the group is the base CIFAR-10 image.
The split is computed on base images first; degradations are generated afterward.
This prevents the same source image from appearing in train and validation.

The resolved split is written to `data/splits.json` with per-split indices and
group ids. The fingerprint is written to `data/fingerprint.json` and is used by
the comparison command.

### 9.4 Image I/O and bit depth

- 8-bit RGB/grayscale: Pillow via torchvision.
- 16-bit TIFF/PNG: explicit `tifffile`/Pillow reader; never rely on implicit
  uint8 conversion.
- `channels`: `rgb`, `gray`, or `as_is`; `as_is` is allowed only when the model
  declares matching input channels.
- `value_range`: `auto` (full dtype range) or `fixed` (explicit min/max) in
  sharp v1. Percentile scaling is v1.1. The policy is recorded in
  `normalization.json`.
- `resize`: optional; interpolation and antialias policy recorded.
- Corrupt images raise `DataError` naming the path, row, and index.
- The dataset inspection command reports image modes, dtypes, sizes, and
  per-channel min/max before training.

### 9.5 Normalization

- `compute`: streaming per-channel mean and standard deviation over the training
  split, written to `normalization.json` and reused for validation/test.
- `supplied`: explicit values, useful for ImageNet-pretrained models.
- `none`: no normalization; only for debugging or already-normalized arrays.
- Statistics are computed after augmentation-free decoding and resizing, before
  random augmentation. This is the standard behavior and is documented.
- A cache key includes the data fingerprint, image size, channel policy, and
  value-range policy.

### 9.6 Transforms

- Built on `torchvision.transforms.v2`.
- Config is a list of `{name, ...params}` entries; custom transforms register with
  `@register_transform("name")`.
- Train and eval pipelines are separate. Augmentation is never applied to
  validation/test.
- Supported v1 transforms: resize, center crop, random crop, random resized crop,
  horizontal/vertical flip, rotation, color jitter, grayscale, Gaussian blur,
  RandAugment, AutoAugment, normalization, Random Erasing, and ToTensor.
- MixUp/CutMix are batch-level and live in the training engine, not in the
  per-sample transform pipeline.
- A `preview_augmentations` utility produces a labeled gallery for a handful of
  samples.

### 9.7 Inspection

`cnncls inspect-data` reports, without training:

- class counts and imbalance ratio per split;
- duplicate images by content hash;
- corrupt/unreadable files;
- image dimensions, modes, dtypes, and per-channel statistics;
- split overlap by path, content hash, and group;
- group leakage between train and validation;
- the raw-to-intake contract: class map, tensor shape, dtype, range, and
  normalization the model will receive;
- a sample gallery and class-distribution figure (optional).

This is both a debugging utility and a portfolio signal: data quality is part of
model quality.

### 9.8 Raw input versus model intake

The data layer must make this distinction explicit, not implicit:

| Stage | What it contains |
|---|---|
| **You provide** | Image files, folder or manifest labels, split/group columns, optional class-name list, image bit depth and channel layout |
| **Adapter validates** | Paths, labels, class counts, split membership, group leakage, readable images, dtype/mode/size, value range |
| **Adapter resolves** | `class_names`, label-to-index map, `in_channels`, `image_size`, normalization statistics, split fingerprint |
| **Model receives** | `float32` tensor `[C, H, W]` per sample, integer label, batch `[B, C, H, W]`, normalized value range, optional grayscale-to-RGB or stem adaptation |
| **Contract artifact** | `data/model_contract.json` with shape, dtype, range, normalization, label map, channel policy, and input adaptation |

The contract is printed by `inspect-data` and `train --dry-run`, written to every
run directory, and included in the HTML report. A mismatch between the adapter's
output and the model's declared input is a configuration error, not a runtime
shape error halfway through the first epoch.

---

## 10. Model layer

### 10.1 Registry and model spec

Each model registers a name and a parameter model:

```python
@register_model("residual_cnn")
class ResidualCNNBuilder:
    Params = ResidualCNNParams

    @staticmethod
    def build(params: ResidualCNNParams, *, num_classes: int, in_channels: int) -> nn.Module:
        ...
```

The config's `model.name` resolves the builder; `model.params` is validated by the
builder's Pydantic parameter model. Unknown names and unknown params fail before
model construction.

### 10.2 Built-in model families

| Name | Purpose |
|---|---|
| `simple_cnn` | Minimal stacked Conv-BN-ReLU-MaxPool baseline; fast smoke tests and ablations |
| `residual_cnn` | Generic configurable residual CNN: stem, stages, blocks, widths, downsampling, dropout, residual on/off. The reference topology is a demo preset, not a separate builder |
| `torchvision.*` | Any torchvision classification model, with configurable stem adaptation and freeze options |

The `residual_cnn` builder applies an optional stem convolution block, then per
stage an optional stage convolution block (`conv: true/false`), the configured
number of residual blocks, and the configured downsampling. The reference preset
uses a 3→64 stem, no stage convolution in stage 1, one stage convolution in
stages 2–4, one residual block per stage, and max-pooling after stages 1–3. That
is a configuration of the generic builder, not a separate model class.

The `torchvision` wrapper handles:

- model lookup by name (e.g., `torchvision.resnet18`);
- weights (`None`, `DEFAULT`, or a named enum/string);
- `num_classes` replacement;
- configurable stem adaptation (`keep_7x7_stride1`, `adapt_3x3`, or
  `upsample_224`) so a model can run at native 32×32 or at 224×224;
- `freeze: none | backbone | all_but_head`;
- optional unfreeze schedule at configured epochs;
- BatchNorm policy (`train_bn: true/false`);
- differential learning rates via regex param groups;
- input-channel adaptation: grayscale-to-RGB replication or first-conv
  replacement.

#### Stem adaptation policy

`stem_adaptation` is an explicit mode, not a boolean:

- `keep_7x7_stride1` (default for ImageNet weights at 32×32): keep the
  pretrained 7×7 kernel, set stride to 1, and replace the initial max-pool with
  identity. This preserves the pretrained stem and is the fairness anchor for
  the 32×32 transfer benchmark. It is also the same architecture used for a
  from-scratch ResNet-18 control, so pretraining is the only changed variable.
- `adapt_3x3`: replace the 7×7 stem with a 3×3 stride-1 convolution, initialized
  from the center 3×3 crop of the ImageNet kernel when weights are present, and
  remove the max-pool. This matches common small-image ResNet conventions and is
  an optional ablation, but it loses part of the pretrained stem.
- `upsample_224`: the day 20 recipe: standard 7×7 stride-2 stem and max-pool on
  224×224 inputs. This is a separate compute-heavy track, not the fairness
  anchor.

A linear probe must use `keep_7x7_stride1` or `upsample_224`. Freezing a
randomly initialized `adapt_3x3` stem would make the "ImageNet features" pass
through an untrained first layer, so that combination is rejected at config
resolution. If `adapt_3x3` is used with pretrained weights, the stem is trainable
even in an otherwise frozen backbone.

### 10.3 Transfer-learning modes

- **Feature extraction / linear probe:** freeze the backbone, train the new head.
- **Full fine-tune:** train all parameters, with a smaller learning rate for the
  backbone and a larger one for the head.
- **Gradual unfreezing:** start with the head, unfreeze stages at configured
  epochs.
- **From-scratch control:** same architecture with `weights=None`.

The config supports per-parameter-group overrides:

```yaml
model:
  transfer:
    weights: IMAGENET1K_V1
    freeze: backbone
    train_bn: false
    unfreeze_schedule:
      - {epoch: 3, modules: [layer4, fc]}
      - {epoch: 6, modules: [layer3]}
    param_groups:
      - {pattern: "^fc\\.", lr_multiplier: 10.0}
      - {pattern: "^layer[1-3]\\.", lr_multiplier: 0.1}
```

The resolver validates that every regex matches at least one parameter; an
unmatched pattern is a config error, not a silent no-op.

### 10.4 Model summary

`model_summary.json` records:

- total and trainable parameter counts;
- per-module parameter counts;
- output shape for the configured input;
- optional MACs/FLOPs (via `torchinfo` when installed);
- receptive field estimate per stage;
- frozen/unfrozen modules;
- input adaptation and normalization contract.

The architecture figure uses this metadata, so it does not require a live
training run.

---

## 11. Training engine

### 11.1 Trainer responsibilities

- Resolve device and seeds.
- Build DataLoaders with worker policy.
- Build model, optimizer, scheduler, loss, and callbacks.
- Run train/validation epochs.
- Log metrics to configured trackers.
- Save last/best checkpoints.
- Restore best weights after early stopping.
- Write run artifacts.
- Never evaluate the test split.
- Never generate plots.

### 11.2 Training loop

Per epoch:

1. Set model train mode.
2. Iterate batches; move inputs/labels to device.
3. Optional MixUp/CutMix.
4. Forward, loss, backward.
5. Gradient accumulation with loss scaling.
6. Gradient clipping.
7. Optimizer step.
8. Scheduler step if `step_unit: batch`.
9. Accumulate batch metrics.
10. Validation forward pass if configured.
11. Scheduler step if `step_unit: epoch`.
12. Compute metrics; update best; callbacks.
13. Write metrics record.
14. Check early-stopping condition.
15. Checkpoint last, and best if improved.

### 11.3 Callbacks

A small callback protocol:

```python
class Callback:
    def on_train_start(self, state: TrainingState) -> None: ...
    def on_epoch_end(self, state: TrainingState) -> None: ...
    def on_train_end(self, state: TrainingState) -> None: ...
```

Built-ins:

- `ModelCheckpoint` — best and last checkpoints;
- `EarlyStopping` — patience, min delta, mode, restore-best;
- `GradientMonitor` — per-layer gradient norms and vanishing/exploding flags;
- `EMACallback` — optional exponential moving average of weights;
- `UnfreezeCallback` — applies a scheduled unfreeze;
- `TrackerCallback` — forwards metrics to console/CSV/TensorBoard;
- `TimerCallback` — epoch and step timing;
- `NaNDetector` — stops with a diagnostic if loss/gradients become non-finite.

Callbacks receive state and cannot mutate the test split. A callback failure marks
the run failed and writes the traceback.

### 11.4 Optimizers and schedulers

Optimizers: AdamW (default), Adam, SGD with momentum, RMSprop. Schedulers:
constant, cosine, cosine with warmup, step, ReduceLROnPlateau, one-cycle. Scheduler
stepping is explicit (`epoch` or `batch`). Warmup is expressed in epochs or steps,
not as an opaque lambda.

Weight decay is excluded from BatchNorm and bias parameters by default; this is
configurable. The parameter-group construction is logged.

EMA is optional (`training.ema.enabled`, default false). When enabled, the
callback maintains shadow weights with the configured decay; validation and
checkpointing use the EMA weights, while the raw model's metrics are logged
alongside them. The selection metric may be `val_accuracy_ema` or
`val_accuracy`; the choice is recorded in `summary.json`. EMA does not apply
during the first epoch's warmup.

### 11.5 AMP, accumulation, and device policy

- `amp: auto` uses CUDA fp16 autocast + GradScaler; CPU/MPS default to fp32.
- `amp: bf16` is available where the device supports bfloat16; unsupported
  combinations fail at config resolution.
- `amp: fp16` requires CUDA; requesting it on MPS/CPU is a config error.
- Gradient accumulation scales the loss and records the effective batch size.
- `channels_last` and `torch.compile` are opt-in and off by default; they are
  validated against the device.
- MPS caveats are documented: no CUDA APIs, limited memory introspection, some
  kernels nondeterministic, and `torch.compile` support may lag.

### 11.6 Checkpoints and resume

`checkpoints/last.pt` contains model, optimizer, scheduler, scaler, epoch, global
step, best metric, config hash, class names, normalization, and RNG states.
`checkpoints/best.pt` contains the same selection metadata and weights.

`cnncls train --resume runs/<run>/checkpoints/last.pt` restores the full state and
continues. The resumed run writes a new metrics segment and records the resume
point; it does not overwrite the original run's history. A checkpoint whose config
hash does not match the current config requires `--allow-config-change`.

### 11.7 Run directory

```text
runs/<UTC timestamp>_<experiment name>_<short hash>/
├── config.resolved.yaml
├── environment.json          # OS, Python, PyTorch, TorchVision, device, CPU/GPU
├── git.json                  # commit, branch, dirty flag (when available)
├── status.json               # running/completed/failed + timestamps
├── data/
│   ├── splits.json
│   ├── class_names.json
│   ├── normalization.json
│   ├── model_contract.json
│   └── fingerprint.json
├── model_summary.json
├── metrics.jsonl
├── history.csv
├── checkpoints/{last.pt,best.pt}
├── predictions/val_predictions.npz
├── figures/*.png,*.svg
├── report.html
├── summary.json
└── stdout.log
```

Writes are atomic where practical. A failed run leaves a readable diagnostic
rather than a half-written summary.

---

## 12. Evaluation

### 12.1 Metrics

Core:

- top-1 accuracy, top-k accuracy;
- per-class precision, recall, F1, support;
- macro and weighted averages;
- balanced accuracy;
- confusion matrix (counts and row-normalized);
- negative log-likelihood;
- calibration: ECE, MCE, Brier score, reliability bins;
- optional one-vs-rest ROC-AUC and PR-AUC;
- bootstrap confidence intervals for accuracy and macro F1;
- inference latency, throughput, parameter count, and peak memory.

Test-time augmentation is optional and restricted to deterministic,
label-preserving transforms (`identity`, `hflip` by default). Probabilities are
averaged across views; the base and TTA metrics are both reported, and the
comparison command treats them as separate rows. TTA is applied only during
evaluation, never during training or model selection, and it never sees labels.

### 12.2 Model selection and test discipline

- During training, the selection metric is validation-only.
- The test split is available only through `cnncls evaluate --split test` or an
  explicit evaluation step after training.
- The demo scripts run test evaluation once, after the best checkpoint is chosen.
- The run summary records which split was used and when.
- The test fingerprint is stored with the results.

### 12.3 Error analysis

- Most-confident wrong predictions with source paths and probabilities.
- Confusion pairs ranked by count.
- Per-class confidence distributions.
- Misclassification gallery with true label, predicted label, confidence, and
  optional Grad-CAM overlay.
- Optionally, a content-vs-degradation control for the optics demo: verify that
  the classifier is not simply memorizing CIFAR class content. The demo can train
  a content classifier on the same images as a control and compare confusion
  structure. This is a stretch analysis, not a v1 gate.

### 12.4 Fair comparison

`cnncls compare runs/a runs/b ...`:

- loads `summary.json`, `data/fingerprint.json`, and `environment.json`;
- refuses to compare runs with different test fingerprints;
- reports accuracy, macro F1, 95% CI, params, trainable params, train time,
  latency, peak memory, and device;
- computes paired bootstrap differences when predictions are available;
- labels protocol differences (input size, epochs, augmentation, pretrained
  weights, freeze mode) explicitly;
- produces a Markdown table, an HTML report, and a Pareto figure.

The CIFAR-10 transfer benchmark uses this command so the comparison cannot silently
change the test set.

### 12.5 Prediction

`cnncls predict` accepts a single image, a folder, or a manifest; loads a
checkpoint and its normalization contract; writes a CSV/JSON with path, predicted
class, confidence, and per-class probabilities. It does not require the training
split.

---

## 13. Visualization and reporting

### 13.1 Shared style

- Okabe-Ito colorblind-safe palette.
- White background, no top/right spines, light grid behind data.
- DejaVu Sans, 10 pt body, 11 pt titles.
- `viridis` for continuous image data; never rainbow/jet.
- Save PNG at 200–300 dpi and SVG for publication.
- Deterministic figure ordering and filenames.
- A `save_figure` helper creates parent directories and closes figures.

### 13.2 Figure catalog

| Function | Output |
|---|---|
| `plot_training_history` | Loss, accuracy, LR, gradient norm; best epoch; early stop; optional smoothing |
| `plot_multi_run_history` | Overlay of selected runs on validation accuracy/loss |
| `plot_confusion_matrix` | Counts and row-normalized panels |
| `plot_per_class_metrics` | Precision/recall/F1 dot plot with support |
| `plot_calibration` | Reliability diagram, ECE, confidence histogram |
| `plot_confidence_errors` | Most-confident wrong predictions |
| `plot_error_gallery` | Image grid with true/pred/confidence and optional Grad-CAM |
| `plot_dataset_overview` | Class distribution, sample grid, image-size distribution |
| `plot_augmentation_gallery` | One image under multiple augmentations |
| `plot_filters` | First-layer filters, color-normalized |
| `plot_feature_maps` | Layer-wise feature maps with consistent scaling |
| `plot_activation_stats` | Activation distributions and dead-channel fraction |
| `plot_gradcam` | Correct Grad-CAM overlay using hooks |
| `plot_embedding` | Optional t-SNE/PCA of penultimate features |
| `plot_architecture_card` | Parameter table, stage shapes, and optional diagram |
| `plot_model_comparison` | Accuracy vs params, accuracy vs latency, Pareto |
| `plot_learning_rate` | Schedule preview before training |

### 13.3 Architecture visualization

The default architecture artifact is a **model card**: a clean table of stages,
output shapes, parameter counts, and trainable status, rendered as a figure and
JSON. When `torchinfo` is installed, the layer table and MACs are included.

VisualTorch, TorchView, Netron/ONNX, and Graphviz-based diagrams are v1.1
extensions. The v1 card must be useful on its own, without system dependencies or
optional architecture backends.

The optional-import rule is still enforced: a missing optional package produces a
clear skip message, never a `NameError`, and the model card always renders.

### 13.4 Experiment report

`report.html` is self-contained: CSS and images are embedded, no CDN or network is
required. Sections:

1. Experiment identity, config summary, git/environment.
2. Dataset summary and split integrity.
3. Model summary and architecture card.
4. Training curves and learning-rate schedule.
5. Validation metrics.
6. Test metrics (only if explicitly evaluated).
7. Confusion matrix and per-class metrics.
8. Calibration and confidence analysis.
9. Error gallery.
10. Runtime, parameters, latency, and reproducibility notes.

A Markdown summary mirrors the key numbers for the docs site. The report generator
reads artifacts; it never retrains or re-evaluates.

### 13.5 Debugging and sanity checks

The `diagnostics` package provides callable checks and a `cnncls debug` command,
separate from plotting:

- **One-batch overfit:** train a single batch for a small number of steps and
  assert that the loss can approach zero. This catches a broken model, loss,
  optimizer, or label mapping before a full run.
- **Shape trace:** register forward hooks and print each module's input/output
  shape and parameter count.
- **Gradient report:** per-layer gradient norms, zero/NaN counts, and
  vanishing/exploding flags.
- **Activation report:** per-layer mean/std, saturation, and dead-channel
  fraction on a fixed batch.
- **Determinism check:** run two identical short trainings with the same seed and
  compare loss and predictions within tolerance.
- **Checkpoint inspector:** print/validate the contents of `last.pt`/`best.pt`
  and detect config or class-map mismatches.
- **Optional profile:** deferred to v1.1.

These checks run on CPU and synthetic fixtures in the test suite. A failed
sanity check writes the same diagnostic information a report would, but without
requiring matplotlib.

---

## 14. CLI and workflows

```bash
cnncls init my_project                 # scaffold configs and a data README
cnncls doctor                          # versions, devices, optional deps
cnncls inspect-data -c config.yaml
cnncls debug -c config.yaml [--overfit-batch] [--gradients] [--determinism]
cnncls train -c config.yaml [--set key=value] [--resume PATH] [--dry-run]
cnncls evaluate -c config.yaml --checkpoint PATH --split test
cnncls predict -c config.yaml --checkpoint PATH --input PATH --output predictions.csv
cnncls report -c config.yaml --run runs/PATH
cnncls compare runs/a runs/b [--out results/comparison/comparison.html]
```

- `--dry-run` resolves config, validates data, builds the model, runs one
  forward/backward pass on one batch, prints the raw-to-intake contract, model
  input shape/dtype/range, parameter counts, and the resolved device, then exits
  without training.
- `--set` is repeatable and validated after parsing.
- `train` writes the report automatically when `report.enabled`.
- `evaluate` writes predictions and an evaluation report; it never mutates the
  checkpoint.
- `doctor` prints device selection, PyTorch/TorchVision versions, optional
  dependencies, and known device caveats.

A `sweep` command is a v1.1 extension: grid or random overrides, sequential
execution, and a comparison report. It is not required for the first release.

---

## 15. Demo suite

There are two dataset case studies, one ablation study, and one quickstart demo:

1. **CIFAR-10 end-to-end** — the scratch baseline and the three-config transfer
   benchmark. The transfer benchmark is a sub-suite of this case study, not a
   separate dataset.
2. **Optical degradation classification** — a second dataset and a proof that the
   data adapter and reporting layers generalize.
3. **Ablation study** — controlled experiments on a CIFAR subset, not a separate
   application.
4. **Bring-your-own-data quickstart** — a tiny local fixture that demonstrates
   the raw-to-model data contract end-to-end in under a minute.

Each demo is standalone: `demos/<name>/` contains its README, configs, run
scripts, expected runtime, and result links. Demos share the installed library
but do not import each other.

```text
demos/
├── cifar10/               # scratch baseline + transfer benchmark
│   ├── README.md
│   ├── configs/
│   ├── scripts/
│   └── results.md
├── optics_degradation/    # manifest-based optical degradation task
│   ├── README.md
│   ├── configs/
│   ├── scripts/
│   └── results.md
├── ablations/             # reduced-budget controlled comparisons
│   ├── README.md
│   ├── sweep.yaml
│   ├── scripts/
│   └── results.md
└── bring_your_own_data/   # minimal manifest/ImageFolder quickstart
    ├── README.md
    ├── fixture/
    ├── manifest.csv
    ├── config.yaml
    ├── scripts/
    └── results.md
```

### 15.1 CIFAR-10 from scratch

- Config: `configs/cifar10/scratch_baseline.yaml`.
- Model: `residual_cnn` configured with a 3→64 stem and four residual stages;
  stages 2–4 begin with a stage convolution, each stage has one residual block,
  and max-pooling follows stages 1–3; widths 64/128/256/512; GAP; dropout 0.3;
  7,827,786 parameters. The reference topology is a demo preset, not a
  library-specific model class.
- Reference: the source notebook's best validation 92.42%, test 92.61% (MPS,
  seed 42).
- Full run: plain `random` 45k/5k split, supplied CIFAR mean/std, 50 epochs,
  batch 128, Adam, weight decay 5e-4, cosine schedule, no gradient clipping,
  early stopping patience 10.
- The gate is ≥90% test accuracy; the target is to reproduce the 92–93% range.
  If the reproduced number differs, the report states the environment and the
  difference rather than tuning against the test split.
- A second "controlled recipe" config may use AdamW and gradient clipping for
  comparison, but it is labelled separately and is not the reference
  reproduction.
- Deliverables: curves, confusion matrix, per-class report, calibration, error
  gallery, model card, HTML report, summary JSON.
- Quick mode: subset, 2 epochs, used by CI/docs smoke.

### 15.2 CIFAR-10 transfer benchmark

Configs (all 32×32, `keep_7x7_stride1`):

| Config | Weights | Trainable | Budget |
|---|---|---|---|
| `resnet18_scratch_cifar.yaml` | none | all | 30 epochs, AdamW 1e-3, wd 5e-4, cosine, early stop 10 |
| `resnet18_linear_probe_cifar.yaml` | ImageNet | head only | max 30 epochs, AdamW head 1e-3, wd 5e-4, early stop 10 |
| `resnet18_finetune_cifar.yaml` | ImageNet | all | 30 epochs, AdamW backbone 1e-4 / head 1e-3, wd 5e-4, cosine, early stop 10 |

The three configs are the fairness anchor: same plain `random` 45k/5k/10k split
(seed 42), same reference augmentation family, same maximum epoch budget, same
validation-based early stopping, same evaluation. They all use
`keep_7x7_stride1` so the architecture is identical and pretraining is the only
changed variable. They use ImageNet mean/std because the pretrained model expects
it and the scratch control must receive the same input contract; the reference
baseline uses CIFAR mean/std, and the comparison report labels that difference
instead of hiding it.

The linear probe freezes the entire `keep_7x7_stride1` backbone, including the
pretrained stem, so its features are genuinely ImageNet features. The
`adapt_3x3` stem is an optional ablation, not the fairness anchor.

Success for the anchor is not a fixed accuracy number: the report must show
whether pretraining helps, at what compute cost, with paired bootstrap intervals.
The 224×224 tracks and the budget-sensitivity sweep are deferred to v1.1. When
added, they remain separate compute-heavy references and never enter the fairness
anchor.

Deliverables: comparison table, Pareto plot, paired bootstrap, per-model reports,
and a written analysis of accuracy versus compute.

### 15.3 Optical degradation demo

Base: CIFAR-10 images, 32×32, chosen because the dataset is already available,
reproducible, and CI-friendly. The design explicitly notes that the 32×32
resolution makes some optical effects subtle; the demo is a framework and
diagnostics showcase, not a claim about production image-quality classification.

Classes (single dominant degradation per sample):

- clean;
- defocus (Gaussian PSF approximation);
- motion blur (oriented line kernel);
- chromatic aberration (per-channel radial shift);
- vignetting (radial gain);
- sensor noise (Gaussian + Poisson);
- JPEG compression.

Severity is randomized per sample within a configured range. The generator writes
images and a manifest CSV with `path,label,split,group,severity`. Splits are
group-aware by base CIFAR image. The generator is a demo script under
`demos/optics_degradation/`, not core library code; it uses the public image I/O
and manifest helpers.

The model, trainer, evaluator, and report are the same as CIFAR-10. Deliverables:
class gallery, confusion matrix, per-class metrics, confidence errors, and an
analysis of which degradations are confused. A content control (train on CIFAR
labels with the same images) is optional.

### 15.4 Ablation suite

Curated, reduced-budget experiments on a CIFAR subset:

- augmentation: none vs full;
- BatchNorm: on vs off;
- residual connections: on vs off;
- schedule: constant vs cosine.

Optimizer, label-smoothing, dropout, and budget-sensitivity ablations are deferred
to v1.1. Each included ablation is a config; a small shell or Python runner
produces runs and a comparison report. Results are committed as summaries and
figures, with seeds and compute recorded.

### 15.5 Bring-your-own-data quickstart

This is a real, runnable demo, not a docs-only tutorial. It uses a tiny local
fixture generated by a script, so it needs no network, GPU, or license approval.
It runs `inspect-data`, `train` for a couple of epochs, `evaluate`, and `report`
in under a minute, and doubles as the CI smoke test.

The demo must make the **raw-to-intake contract** unmistakable. Its README and
`docs/bring_your_own_data.md` show two side-by-side tables.

**What you provide:**

```text
fixture/
├── train/
│   ├── circle/*.png
│   ├── square/*.png
│   └── stripe/*.png
├── val/
│   └── ...
└── manifest.csv     # optional: path,label,split,group
```

- Image files: 8-bit RGB, 8-bit grayscale, or 16-bit grayscale/RGB.
- Labels: folder names or a manifest `label` column.
- Split: folder names or a manifest `split` column.
- Group: optional manifest column for leakage-safe splitting.
- Class order: adapter order or an explicit `data.class_names` list.
- Value range: automatic, fixed, or percentile, recorded in the config.

**What the model receives:**

- A `DatasetBundle` with `class_names`, `in_channels`, `image_size`,
  `normalization`, and `split_info`.
- One sample as a `torch.float32` tensor of shape `[C, H, W]` and an integer
  label index.
- One batch as `[B, C, H, W]`, after the configured augmentation and
  normalization.
- An explicit model-input contract in `data/model_contract.json`: shape, dtype,
  value range, normalization values, label map, channel policy, and any
  grayscale-to-RGB or stem adaptation.
- The same contract is printed by `inspect-data` and `train --dry-run`, and
  included in the HTML report.

The demo also shows how to register a custom adapter in the demo module without
editing `src/cnn_classifier`. This is the proof that the toolbox is not
CIFAR-specific and that bringing new data is a format-and-config exercise, not a
framework rewrite.

---

## 16. Reproducibility and device policy

### 16.1 Device resolution

`auto` order:

1. CUDA, if `torch.cuda.is_available()`;
2. MPS, if `torch.backends.mps.is_available()` and not disabled by the user;
3. CPU.

Explicit values (`cpu`, `mps`, `cuda`, `cuda:1`) are honored and validated. The
resolved device is recorded in `environment.json` and printed by `doctor`.

### 16.2 Seeding

- Python `random`, NumPy, PyTorch CPU, CUDA, and MPS seeds are set.
- DataLoader workers receive a deterministic `worker_init_fn` and generator.
- Split generation, augmentation, model initialization, and bootstrap resampling
  all use explicit seeded generators.
- `deterministic: true` enables deterministic algorithms where supported and
  documents device limitations.

### 16.3 What is promised

- Same code, same config, same seed, same device family, same PyTorch version:
  results should be reproducible within numerical tolerance.
- Bit-identical results across CPU/MPS/CUDA are not promised.
- MPS may use nondeterministic kernels; the report records the device and the
  determinism policy.
- Test results include a bootstrap confidence interval rather than a single
  over-precise number.

### 16.4 Worker and memory defaults

- `num_workers: auto`: 4 on CUDA/Linux, 0 on MPS/macOS, 2 on CPU/Linux. The
  resolved value is logged.
- `pin_memory`: true only on CUDA.
- `persistent_workers`: true only when `num_workers > 0`.
- `channels_last`: opt-in; validated on the device.
- `compile`: opt-in; off by default because MPS support and compile time vary.
- Inference latency uses warmup, device synchronization, and repeated runs; the
  measurement method is recorded.

### 16.5 Compute budget and full-run policy

The v1 acceptance run targets Apple Silicon MPS; no CUDA machine is assumed. The
sharp-v1 suite is expected to take roughly 2.5–4 hours and is run in the
background or overnight, in independent stages that can resume:

| Stage | Estimated MPS time |
|---|---|
| Scratch baseline, 50 epochs | 20–30 min |
| Three 32×32 ResNet-18 configs, 30 epochs each | 45–75 min |
| Four core ablations | 30–45 min |
| Optics degradation demo | 30–60 min |
| Bring-your-own-data quickstart | <1 min |
| Reports, comparisons, packaging overhead | 10–20 min |

The implementation plan runs each stage as a separate command, keeps
`checkpoints/last.pt` resumable, and writes a suite manifest. CI never runs the
full suite; it uses the synthetic and quick gates. Full-run figures, summaries,
and HTML reports are committed; checkpoints are not. If a stage fails, the suite
resumes from the last checkpoint rather than restarting the whole benchmark.

---

## 17. Packaging, documentation, and publishing

### 17.1 Package metadata

- Distribution: `cnn-classifier-toolkit` (originally planned as `cnn-classifier`).
- Import: `cnn_classifier`.
- CLI: `cnncls`.
- Python 3.11–3.13.
- License: Apache-2.0.
- Build backend: hatchling; dynamic version from `src/cnn_classifier/_version.py`.
- Core dependencies: torch, torchvision, Pydantic v2, PyYAML, NumPy, Pillow,
  tifffile, matplotlib, Jinja2, Typer, Rich.
- Optional extras:
  - `viz`: torchinfo, scikit-learn (embedding/extra metrics);
  - `tracking`: tensorboard;
  - `docs`: mkdocs-material, mkdocstrings;
  - `dev`: pytest, pytest-cov, hypothesis, ruff, mypy, pre-commit, pandas;
  - `all`: union of the above.
- ONNX export and VisualTorch/TorchView are v1.1 extras, not v1 dependencies.
- Package data: config presets, report template, and style resources resolved
  through `importlib.resources`.
- Canonical URLs:
  - Homepage/repository: `https://github.com/pengzhao-ai/cnn-classifier`
  - Documentation: `https://pengzhao-ai.github.io/cnn-classifier/`
  - Changelog: `https://github.com/pengzhao-ai/cnn-classifier/blob/main/CHANGELOG.md`
  - Issues: `https://github.com/pengzhao-ai/cnn-classifier/issues`
- `mkdocs.yml` uses `site_url: https://pengzhao-ai.github.io/cnn-classifier/` and
  `repo_url: https://github.com/pengzhao-ai/cnn-classifier`.
- README badges and image URLs use the canonical repository path so the PyPI
  page renders.

### 17.2 Documentation site

MkDocs Material, strict build, mkdocstrings, Mermaid diagrams. Navigation:

- Home — hero, benchmark table, gallery, quickstart, architecture.
- Quickstart.
- Concepts — data, model, training, evaluation, reporting.
- Architecture — layering and dependency rules.
- Configuration — schema, inheritance, overrides, examples.
- Data — adapters, splits, image I/O, augmentation, inspection, raw-to-intake contract.
- Bring your own data — folder/manifest walkthrough, 8/16-bit, class map, model contract.
- Models — registry, built-ins, ResNet, transfer learning.
- Training — loop, callbacks, optimizers, schedulers, AMP, checkpointing.
- Evaluation — metrics, calibration, bootstrap, error analysis, fairness.
- Visualization — figure catalog and report format.
- CLI.
- Reproducibility.
- Benchmarks — CIFAR-10 scratch and transfer results.
- Case studies — CIFAR-10 and optical degradation.
- Extending — custom datasets, models, transforms, metrics, trackers.
- API reference.
- Release process.

The home page is the showcase: a clear result table, a hero figure, a short
architecture diagram, install/quickstart commands, and links to the case studies.
It is built with MkDocs, not a second unrelated site. A standalone case-study HTML
page may be generated for a talk if useful, but it is not required for v1.

### 17.3 README

The README mirrors the home page: badges, one-sentence description, benchmark
table, gallery, install, quickstart, demo commands, architecture summary, testing,
docs link, license, and citation. Images use absolute URLs so the PyPI page
renders.

### 17.4 CI and release

- CI:
  - quality: ruff check/format, mypy strict on `src`;
  - tests: 3.11–3.13 on Ubuntu and macOS, CPU-only, coverage gate;
  - smoke: synthetic train/evaluate/report, bring-your-own-data quickstart, CLI
    help, docs build, package build;
  - docs: MkDocs strict build;
  - optional: slow CIFAR subset test, not on every push.
- Docs workflow: deploy to GitHub Pages from `main`.
- Release workflow: on `v*` tag, run tests, build wheel/sdist, `twine check`,
  GitHub release assets, PyPI trusted publishing.
- No long-lived PyPI token.
- Repository creation, enabling GitHub Pages, and configuring the PyPI trusted
  publisher are owner actions. The build includes the workflow files, a release
  checklist, and local build verification; the owner performs the account-level
  setup and the final live verification.
- `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `NOTICE`, `THIRD_PARTY.md`, `LICENSE`.
- Commit messages end with
  `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

### 17.5 Results policy

- `results/<demo>/figures/`: committed PNG/SVG gallery for that demo, regenerated
  by its `make figures` target.
- `results/<demo>/reports/`: committed summaries, JSON, Markdown, and HTML.
- `results/comparison/`: cross-demo tables, paired comparisons, and the Pareto
  figure.
- `runs/`: gitignored; full checkpoints and per-run logs stay local.
- `docs/images/`: mirrored selected figures; a test checks byte-identical copies
  to prevent documentation drift.
- Each demo README links to its own folder under the common `results/` root. This
  keeps one publishing location without scattering result files across `demos/`.
- Checkpoints are not committed. A download script or release asset may provide
  demo weights; reproducibility relies on config + seed + documented environment.
- Timings are measurements, not guarantees.

---

## 18. Testing and quality

### 18.1 Unit tests

- config loading, defaults merge, override parsing, unknown keys, registry name
  validation, path resolution, schema version.
- registry duplicate registration and lookup errors.
- device resolution with mocked capabilities; seeding determinism.
- split policies: deterministic `random`, stratified balance, group leakage,
  explicit splits, edge cases.
- `data.class_names`: `auto`, matching explicit list, mismatched length, wrong
  adapter label mapping.
- manifest parsing: missing columns, duplicate rows, missing files, relative paths.
- image I/O: 8-bit RGB, 8-bit gray, 16-bit gray, 16-bit RGB, value-range policies,
  corrupt files.
- normalization statistics and cache key.
- raw-to-intake contract: `model_contract.json` shape, dtype, range, label map,
  channel policy, and mismatch rejection before training.
- transform builder: train/eval separation, unknown transform, parameter
  validation.
- model builders: output shape, parameter count, freeze modes, stem adaptation,
  grayscale adaptation, differential param groups.
- model summary.
- checkpoint save/load/resume, config-hash mismatch, RNG restoration.
- early stopping and best-checkpoint selection.
- metrics against hand-computed and scikit-learn references (optional dev dependency).
- TTA probability averaging, deterministic view order, and evaluation-only use.
- EMA shadow update, selection metric, and checkpoint contents.
- calibration on known distributions.
- bootstrap determinism and CI coverage on synthetic data.
- fair-comparison refusal on mismatched fingerprints.
- run artifact schemas and atomic writes.
- diagnostics: one-batch overfit, shape trace, gradient/activation reports,
  determinism check, checkpoint inspection, and NaN detection.
- report generation produces non-empty HTML and Markdown.
- CLI `--help`, `doctor`, `train --dry-run`, invalid configs.

### 18.2 Integration tests

- Tiny synthetic dataset (2 classes, 16×16, 32 samples) trains for one epoch on
  CPU; checkpoint, evaluate, report, and compare all succeed.
- Bring-your-own-data quickstart runs under a minute, writes
  `data/model_contract.json`, and completes inspect/train/evaluate/report.
- Every built-in model family builds and runs one forward/backward pass on a tiny
  input.
- CIFAR-10 quick subset is a slow, opt-in test; it downloads only when explicitly
  requested.
- A resumable training run continues from `last.pt` and produces a second metrics
  segment.
- Import-independence test: importing `cnn_classifier.training` in a subprocess
  must not import matplotlib or jinja2.
- Docs build in strict mode.
- Package builds and installs into a fresh environment; smoke import works.

### 18.3 Coverage and style

- Coverage ≥90% for `core`, `config`, `data`, `models`, `training`, `evaluation`,
  and `runs`.
- `viz` is covered by smoke tests and headless report generation, not pixel tests.
- `ruff check`, `ruff format --check`, and `mypy src` must pass.
- Pre-commit runs the same checks locally.
- No network in unit tests; no GPU requirement; no test longer than a few minutes
  in the default suite.

---

## 19. Phased delivery

### Phase 0 — Foundation

Create packaging, `uv`, `src/` skeleton, Pydantic config, device/seeding, CLI
skeleton, test harness, CI, pre-commit, MkDocs skeleton, and the design/plan
documents. Gate: `uv sync`, import, `cnncls doctor`, lint, typecheck, tests, docs
build, package build.

### Phase 1 — Data layer

Implement dataset bundle, torchvision/ImageFolder/manifest/NumPy/synthetic
adapters, split policies, image I/O (8/16-bit), normalization, transforms,
inspection, and tests. Gate: CIFAR-10 and a synthetic manifest train through a
dummy loader; group leakage test passes; augmentation gallery renders headlessly.

### Phase 2 — Models

Implement registry, the simple CNN baseline, the configurable residual CNN, the
torchvision wrapper, stem adaptation, freeze modes, differential param groups,
and summaries. Gate: every model builds, runs a forward pass, and reports
parameter counts; shape tests pass.

### Phase 3 — Training

Implement Trainer, callbacks, optimizers, schedulers, losses, MixUp/CutMix, AMP,
gradient accumulation/clipping, checkpoints, resume, run context, trackers. Gate:
tiny dataset trains, resumes, early-stops, and writes a complete run directory.

### Phase 4 — Evaluation

Implement metrics, calibration, bootstrap, error analysis, latency, prediction,
fair comparison, and evaluation reports. Gate: known-distribution tests pass;
comparison refuses mismatched splits; test split is not touched during training.

### Phase 5 — Visualization and reports

Implement style, data/training/evaluation/feature/architecture/comparison figures,
HTML/Markdown reports, and docs figure generation. Gate: all figures render
headlessly; report is self-contained; optional dependencies degrade cleanly.

### Phase 6 — CIFAR-10 demos and ablations

Run the scratch baseline, the three 32×32 transfer configs, and the four core
ablations on MPS; commit results and analysis. The full stage is expected to take
roughly 1.5–2.5 hours and runs in the background. Gate: scratch test accuracy
≥90% (target ≈92–93%), transfer results reproduced, fair-comparison report
generated, full reports committed.

### Phase 7 — Optical degradation and bring-your-own-data demos

Implement the degradation generator, manifest, group-aware splits, config, and
report, plus the tiny bring-your-own-data fixture and raw-to-intake contract
demonstration. Gate: generator is deterministic, no group leakage, both demos
run end-to-end with committed results, and the bring-your-own-data quickstart
finishes in under a minute. Expected MPS time for the optical demo is 30–60
minutes; the bring-your-own-data demo is under a minute.

### Phase 8 — Documentation and publishing

Complete docs, showcase home, README, release artifacts, CI, Pages, and PyPI
release workflows. Gate: `make gate`, docs strict build, wheel install smoke test.
The owner then creates `pengzhao-ai/cnn-classifier`, enables GitHub Pages, and
configures the PyPI trusted publisher; live publishing is verified through the
release checklist rather than assumed by the build.

### Phase 9 — Optional v1.1 extensions

224×224 transfer tracks, budget-sensitivity sweep, optimizer/label-smoothing/
dropout ablations, sweep runner, ONNX export, profiler, VisualTorch/TorchView
backends, ImageNette/ImageNet-100 case study, advanced Grad-CAM, and embedding
visualization. Not part of the v1 gate.

---

## 20. Acceptance criteria

The project is complete when:

1. A new manifest or ImageFolder dataset can be trained, evaluated, and reported
   from YAML without editing `src/cnn_classifier`.
2. A custom adapter or model can be added by implementing one documented protocol
   and registering it.
3. The CIFAR-10 scratch model reproduces ≥90% test accuracy (target ≈92–93%) with
   committed config, seed, environment, and report.
4. The transfer benchmark compares scratch, linear probe, and 32×32-adapted
   fine-tune with explicit protocol and compute accounting. The 224×224 tracks
   are v1.1.
5. The optical degradation demo runs end-to-end, uses group-aware splits, and
   produces a confusion-matrix analysis.
6. Every key knob listed in G2 is configurable and covered by at least one test.
7. `device: auto` selects MPS on the author's Mac, CUDA on a CUDA machine, and CPU
   otherwise.
8. The full quality gate passes: ruff, format, mypy, tests, coverage, docs,
   package build.
9. The documentation site and README show the benchmark, gallery, quickstart,
   architecture, and case studies.
10. A reviewer can understand the project from the README plus one case study
    without reading the source.
11. The release workflows are complete and, after the owner configures the
    repository environment and PyPI trusted publisher, a tag publishes a
    wheel/sdist to PyPI and assets to GitHub Releases.
12. No test split is used for model selection; no comparison mixes incompatible
    test fingerprints.
13. The bring-your-own-data quickstart runs in under a minute and clearly shows
    both the raw input contract and the exact tensor/label shape the model
    receives.

---

## 21. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scope creep from too many demos and utilities | Phase gates; v1 demos fixed; optional extensions deferred |
| MPS performance or nondeterminism | Device abstraction; CPU fallback; conservative AMP/compile defaults; documented caveats |
| PyTorch version churn | Pin `uv.lock`; test 3.11–3.13; record versions in every run |
| CIFAR-10 download unavailable or slow | Cache under `data/`; checksum; clear error; synthetic tests do not need network |
| Transfer comparison confounded by input size | Stem adaptation at 32×32 for fairness anchor; 224 track clearly separated |
| Data leakage in degradation demo | Split base images first; group-aware split; leakage test |
| Optional visualization dependencies break imports | Optional extras; isolated imports; fallback model card; smoke tests |
| 16-bit image handling surprises | Explicit reader and value-range policy; fixture tests; document channel/range contract |
| Full demo runtime too long for iteration | Quick configs; subset; committed full results; CI uses synthetic data |
| Test set overuse | Test only through explicit evaluate; run metadata records split; comparison checks fingerprint |
| Report drift between docs and results | Regenerate figures from artifacts; byte-identical docs image test |
| PyPI name collision | `cnn-classifier` was unclaimed on 2026-11-02 but rejected on 2026-11-03 because `cnnClassifier` normalizes to the same name; released as `cnn-classifier-toolkit` |
| Over-engineering the config system | Pydantic + small merge resolver; no Hydra; builder review and phase gates |

---

## 22. Decisions made

These are calls made in this design, not unanswered questions:

1. **Pure PyTorch**, not Lightning. The training loop is part of the portfolio.
2. **Pydantic YAML**, not Hydra. Programmatic use, explicit merge semantics, and
   testability win over config-group convenience.
3. **TensorBoard + CSV/JSONL** as defaults; optional W&B integration deferred.
4. **MkDocs Material** as the single documentation site, with a showcase home page.
5. **`cnn-classifier` / `cnn_classifier` / `cnncls`** as distribution, import, and
   CLI names.
6. **CIFAR-10 base images** for the optical degradation demo, with a documented
   higher-resolution extension path.
7. **No full ImageNet.** A 32×32 stem-adapted ResNet-18 is the fairness anchor; the
   224×224 recipe is deferred to v1.1 and is never part of the fairness anchor.
8. **Headless training, artifact-driven reporting.** Training never imports
   matplotlib.
9. **Group-aware splits** in the degradation demo, computed before generation.
10. **16-bit image support** is part of v1, with explicit readers and range policy.
11. **No committed checkpoints**; results and reports are committed, weights are
    reproducible or distributed separately.
12. **No `notebooks/` or `examples/` directories.** Demos and documentation are
    the canonical runnable examples.
13. **No notebook-derived model names in the core API.** The library exposes
    `simple_cnn`, `residual_cnn`, and `torchvision.*`. The reference topology is a
    demo config, not a separate builder. `residual_cnn` uses explicit
    stages/blocks, not an ambiguous depth shorthand. MLP and VGG are not v1
    model families; the codebase stays CNN-focused.
14. **Stem adaptation policy is explicit.** `keep_7x7_stride1` is the fairness anchor
    for 32×32 pretrained comparisons; `adapt_3x3` is an ablation; `upsample_224`
    is reserved for the v1.1 compute-heavy track. A frozen `adapt_3x3` linear
    probe is rejected.
15. **TTA and EMA are optional v1 knobs.** TTA averages probabilities over
    identity/hflip views during evaluation only. EMA is configurable and, when
    enabled, supplies validation/checkpoint weights. Stochastic depth is deferred.
16. **The v1 acceptance run targets MPS.** The sharp-v1 suite is expected to take
    2.5–4 hours and is run in resumable stages. Full results and reports are
    committed; checkpoints are not.
17. **Canonical GitHub identity is `pengzhao-ai/cnn-classifier`.** The docs site
    is `https://pengzhao-ai.github.io/cnn-classifier/`. Repository creation,
    Pages enablement, and PyPI trusted-publisher configuration are owner actions;
    the build supplies the workflows and checklist.
18. **Results live under one common root.** `results/<demo>/` keeps per-demo
    figures/reports together while giving the docs site one gallery to collect.
    Full run artifacts stay in `runs/`.
19. **Bring-your-own-data is a real quickstart demo.** A tiny local fixture runs
    inspect/train/evaluate/report in under a minute and explicitly shows the raw
    input contract versus the tensor/label shape the model receives.

---

## 23. Requirements traceability

| Original request | Design response |
|---|---|
| Configurable, reusable, modular for other CNN classification problems | Sections 7–10: Pydantic config, registries, dataset adapters, model builders, headless trainer, artifact contract. Acceptance criteria 1–2. |
| Showcase understanding of model training and applications | Sections 15 and 17: two case studies, ablation study, fair comparison, reports, tests, docs. Acceptance criteria 3–11. |
| Key techniques: augmentation, overfitting/regularization, BatchNorm, LR management, ResNet, other knobs | G2, 8.3, and 10–13: augmentation, dropout, weight decay, label smoothing, early stopping, BatchNorm, LR schedules/warmup, residual CNN, transfer learning, AMP, EMA, TTA, gradient clipping. Ablations in 15.4. |
| Modular data formatting for new image/label data; no hard-coded parameters | Sections 8–11: `torchvision`, ImageFolder, manifest, NumPy, and custom adapters; configurable split, normalization, transforms, model, optimizer, scheduler, and runtime. Core code contains no demo dataset constants. Acceptance criteria 1–2. |
| Latest PyTorch and TorchVision | G6 and 17.1: verified PyTorch 2.15.0 / TorchVision 0.29.2 on 2026-11-02, compatible lower bounds, exact versions in `uv.lock`. |
| Publication-quality visualization and debugging utilities | Sections 13 and 13.5: consistent style, dashboard, confusion/per-class/calibration/error figures, feature maps, Grad-CAM, architecture card, self-contained report, one-batch overfit, shape trace, gradient/activation diagnostics, determinism check, profiler. Acceptance criterion 8. |
| CIFAR-10 demo, from-scratch vs ResNet-18 transfer comparison, other demo | Section 15: CIFAR-10 case study with scratch baseline and three 32×32 transfer configs; optical degradation case study; ablation study; bring-your-own-data quickstart. |
| Auto CPU/CUDA/MPS | Section 16.1: CUDA → MPS → CPU, explicit override, capability report, MPS caveats. Acceptance criterion 7. |
| Publishable docs/site in the IQ style | Sections 17.2–17.5: MkDocs Material, showcase home, GitHub Pages, README, result reports, PyPI release workflow. Acceptance criteria 9 and 11. |

Deliberately not covered: full ImageNet training, deployment/serving, distributed
training, detection/segmentation, and notebook-only workflows.

---

## 24. Open questions for the author

None block the plan. The following can be settled during implementation without
changing the architecture:

- Whether to publish a small set of demo checkpoints as GitHub release assets.
- Whether the docs home should include an embedded interactive figure or remain
  static SVG/PNG.
- Whether a v1.1 sweep command should use Optuna or a simple grid/random runner.
- Whether the optical demo should add a higher-resolution track in v1.1.
