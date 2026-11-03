# Experiment report — bring_your_own_data_quickstart

- Run directory: `runs/byo_quickstart/20261103T123745Z_bring_your_own_data_quickstart_2a835edf`
- Status: completed
- Config hash: 2a835edfac0ebd5c

## Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.600 |
| Top-k accuracy | 1.000 |
| Macro F1 | 0.505 |
| NLL | 1.064 |
| ECE | 0.235 |
| MCE | 0.235 |
| Brier score | 0.644 |
| Support | 15 |

| Prediction archive | Samples | Saved accuracy |
| --- | --- | --- |
| val_predictions.npz | 15 | 0.600 |

## Configuration

### Experiment

- `name`: bring_your_own_data_quickstart
- `tags`: ["byo", "manifest", "16-bit", "quickstart"]
- `notes`: Tiny committed 16-bit grayscale fixture; runs end to end in under a minute.
- `seed`: 7

### Data

- `adapter`: manifest
- `root`: —
- `manifest`: ./demos/bring_your_own_data/manifest.csv
- `image_root`: ./demos/bring_your_own_data/fixture
- `class_names`: ["circle", "square", "stripe"]
- `num_classes`: 3
- `channels`: gray
- `image_size`: [16, 16]
- `split.mode`: explicit
- `split.val_fraction`: 0.1
- `split.seed`: 7
- `split.group_column`: —
- `normalization.mode`: compute
- `normalization.mean`: —
- `normalization.std`: —
- `normalization.cache`: yes
- `transforms.train`: [{"name": "Resize", "size": [16, 16]}, {"name": "ToTensor"}, {"mean": "auto", "name": "Normalize", "std": "auto"}]
- `transforms.eval`: [{"name": "Resize", "size": [16, 16]}, {"name": "ToTensor"}, {"mean": "auto", "name": "Normalize", "std": "auto"}]
- `value_range.mode`: auto
- `value_range.min`: —
- `value_range.max`: —

### Model

- `name`: simple_cnn
- `params.channels`: [8, 16]
- `params.dropout`: 0
- `params.norm`: batch
- `transfer.weights`: —
- `transfer.freeze`: none
- `transfer.train_bn`: no
- `transfer.unfreeze_schedule`: []
- `transfer.param_groups`: []

### Training

- `epochs`: 2
- `batch_size`: 8
- `grad_accum_steps`: 1
- `selection_metric`: val_accuracy
- `optimizer.name`: adamw
- `optimizer.lr`: 0.001
- `optimizer.weight_decay`: 0.0001
- `optimizer.momentum`: 0.9
- `scheduler.name`: cosine
- `scheduler.warmup_epochs`: 0
- `scheduler.min_lr_ratio`: 0
- `scheduler.step_size`: 10
- `scheduler.gamma`: 0.1
- `scheduler.step_unit`: epoch
- `loss.name`: cross_entropy
- `loss.label_smoothing`: 0
- `grad_clip_norm`: —
- `amp`: auto
- `mixup_alpha`: 0
- `cutmix_alpha`: 0
- `ema.enabled`: no
- `ema.decay`: 0.999
- `early_stopping.enabled`: no
- `early_stopping.monitor`: val_accuracy
- `early_stopping.mode`: max
- `early_stopping.patience`: 5
- `early_stopping.min_delta`: 0
- `checkpoint.monitor`: val_accuracy
- `checkpoint.mode`: max
- `checkpoint.save_last`: yes

### Evaluation

- `metrics`: ["accuracy", "macro_f1", "per_class", "confusion_matrix", "nll"]
- `top_k`: 3
- `bootstrap.enabled`: no
- `bootstrap.n_resamples`: 200
- `bootstrap.confidence`: 0.95
- `bootstrap.stratify`: yes
- `calibration.bins`: 10
- `latency.enabled`: no
- `latency.warmup`: 5
- `latency.iterations`: 20
- `error_analysis.max_examples`: 8
- `tta.enabled`: no
- `tta.transforms`: ["identity", "hflip"]
- `tta.reduction`: mean_probabilities

### Tracking

- `console`: yes
- `csv`: yes
- `tensorboard`: no

### Report

- `enabled`: yes
- `formats`: ["html", "markdown"]
- `figures`: ["training_history", "confusion_matrix", "per_class", "model_summary"]

### Runtime

- `device`: auto
- `deterministic`: yes
- `num_workers`: 0
- `persistent_workers`: no
- `pin_memory`: no
- `channels_last`: no
- `compile`: no

### Other

- `schema_version`: 1.0

## Data contract

- `channel_policy`: gray
- `class_names`: ["circle", "square", "stripe"]
- `dtype`: float32
- `input_shape`: [1, 16, 16]
- `label_map`: circle=0, square=1, stripe=2
- `normalization.mean`: [0.24038831041123357]
- `normalization.std`: [0.2910660097879871]
- `normalization.source`: computed
- `stem_adaptation`: —
- `value_range`: [0.0, 1.0]

## Split integrity

| Split | Count | Fingerprint |
| --- | --- | --- |
| train | 45 | 89db3c66423401c8 |
| val | 15 | 2a94a94f15ecc752 |

Split mode: explicit

Group overlap: []

Duplicate overlap: []

## Model

- Total parameters: 1,323
- Trainable parameters: 1,323
- Input shape: [1, 16, 16]
- Output shape: [1, 3]

| Module | Type | Parameters |
| --- | --- | --- |
| model | SimpleCNN | 1,323 |
| features | Sequential | 1,272 |
| features.0 | Conv2d | 72 |
| features.1 | BatchNorm2d | 16 |
| features.2 | ReLU | 0 |
| features.3 | MaxPool2d | 0 |
| features.4 | Conv2d | 1,152 |
| features.5 | BatchNorm2d | 32 |
| features.6 | ReLU | 0 |
| features.7 | MaxPool2d | 0 |
| head | Sequential | 51 |
| head.0 | AdaptiveAvgPool2d | 0 |
| head.1 | Flatten | 0 |
| head.2 | Dropout | 0 |
| head.3 | Linear | 51 |

## Training

Epochs run: 2
Best validation accuracy: 0.600 (epoch 1)

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy | Learning rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.168 | 1.064 | 0.3333 | 0.6 | 0 |
| 2 | 1.056 | 1.062 | 0.3333 | 0.3333 | 0 |

## Test metrics

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| circle | 0.000 | 0.000 | 0.000 | 5 |
| square | 0.455 | 1.000 | 0.625 | 5 |
| stripe | 1.000 | 0.800 | 0.889 | 5 |

## Calibration and error analysis

![Calibration](figures/calibration.png)
![Misclassifications](figures/errors.png)

## Figures

![Model card](figures/model_card.png)
![Per-class metrics](figures/per_class.png)

## Runtime

- Python: 3.11.8
- PyTorch: 2.15.0
- TorchVision: 0.29.2
- Platform: macOS-27.0.1-arm64-arm-64bit
- Device: {"cuda_available": false, "device": "mps", "mps_available": true, "platform": "macOS-27.0.1-arm64-arm-64bit", "python_version": "3.11.8", "torch_version": "2.15.0"}
- Training device: mps
- Config hash: 2a835edfac0ebd5c
- Epochs run: 2
- Git commit: 42a1941 (dirty)
- Git branch: main

## Reproduction

```bash
cnncls train -c <config.yaml> --runs-dir runs  # produced 20261103T123745Z_bring_your_own_data_quickstart_2a835edf
cnncls evaluate --run runs/byo_quickstart/20261103T123745Z_bring_your_own_data_quickstart_2a835edf --checkpoint runs/byo_quickstart/20261103T123745Z_bring_your_own_data_quickstart_2a835edf/checkpoints/best.pt
cnncls report --run runs/byo_quickstart/20261103T123745Z_bring_your_own_data_quickstart_2a835edf
cnncls compare runs/byo_quickstart/20261103T123745Z_bring_your_own_data_quickstart_2a835edf <other-run>
```
