# Experiment report — ablation_no_batchnorm

- Run directory: `runs/20261103T123447Z_ablation_no_batchnorm_8a7bf7ab`
- Status: completed
- Config hash: 8a7bf7abe135c3f0

## Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.485 |
| Top-k accuracy | 0.818 |
| Macro F1 | 0.473 |
| NLL | 1.386 |
| ECE | 0.017 |
| MCE | 0.041 |
| Brier score | 0.645 |
| Support | 10,000 |
| Accuracy interval | 0.950 CI [0.476, 0.495] |

| Prediction archive | Samples | Saved accuracy |
| --- | --- | --- |
| test_predictions.npz | 10,000 | 0.485 |

## Configuration

### Experiment

- `name`: ablation_no_batchnorm
- `tags`: ["ablation", "batchnorm"]
- `notes`: BatchNorm off; convolutions use bias instead.
- `seed`: 42

### Data

- `adapter`: torchvision
- `root`: ./data
- `manifest`: —
- `image_root`: —
- `class_names`: auto
- `num_classes`: 10
- `channels`: rgb
- `image_size`: [32, 32]
- `split.mode`: random
- `split.val_fraction`: 0.1
- `split.seed`: 42
- `split.group_column`: —
- `normalization.mode`: supplied
- `normalization.mean`: [0.4914, 0.4822, 0.4465]
- `normalization.std`: [0.247, 0.2435, 0.2616]
- `normalization.cache`: yes
- `transforms.train`: [{"name": "RandomHorizontalFlip", "p": 0.5}, {"name": "RandomCrop", "padding": 4, "size": 32}, {"name": "ToTensor"}, {"mean": "auto", "name": "Normalize", "std": "auto"}]
- `transforms.eval`: [{"name": "ToTensor"}, {"mean": "auto", "name": "Normalize", "std": "auto"}]
- `value_range.mode`: auto
- `value_range.min`: —
- `value_range.max`: —
- `adapter_params.dataset`: CIFAR10
- `adapter_params.download`: yes
- `adapter_params.max_train_samples`: 10,000
- `adapter_params.max_val_samples`: 2,000

### Model

- `name`: residual_cnn
- `params.stem.channels`: 32
- `params.stem.kernel_size`: 3
- `params.stem.stride`: 1
- `params.stem.pool`: no
- `params.stages`: [{"blocks": 1, "channels": 32, "conv": false, "downsample": "maxpool"}, {"blocks": 1, "channels": 64, "conv": true, "downsample": "maxpool"}, {"blocks": 1, "channels": 128, "conv": true, "downsample": "none"}]
- `params.dropout`: 0.2
- `params.norm`: none
- `params.residual`: yes
- `transfer.weights`: —
- `transfer.freeze`: none
- `transfer.train_bn`: no
- `transfer.unfreeze_schedule`: []
- `transfer.param_groups`: []

### Training

- `epochs`: 8
- `batch_size`: 128
- `grad_accum_steps`: 1
- `selection_metric`: val_accuracy
- `optimizer.name`: adamw
- `optimizer.lr`: 0.001
- `optimizer.weight_decay`: 0.0005
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
- `early_stopping.enabled`: yes
- `early_stopping.monitor`: val_accuracy
- `early_stopping.mode`: max
- `early_stopping.patience`: 10
- `early_stopping.min_delta`: 0
- `checkpoint.monitor`: val_accuracy
- `checkpoint.mode`: max
- `checkpoint.save_last`: yes

### Evaluation

- `metrics`: ["accuracy", "top_k", "macro_f1", "per_class", "confusion_matrix", "ece", "nll"]
- `top_k`: 3
- `bootstrap.enabled`: yes
- `bootstrap.n_resamples`: 1,000
- `bootstrap.confidence`: 0.95
- `bootstrap.stratify`: yes
- `calibration.bins`: 15
- `latency.enabled`: no
- `latency.warmup`: 20
- `latency.iterations`: 100
- `error_analysis.max_examples`: 32
- `tta.enabled`: no
- `tta.transforms`: ["identity", "hflip"]
- `tta.reduction`: mean_probabilities

### Tracking

- `console`: yes
- `csv`: yes
- `tensorboard`: yes

### Report

- `enabled`: yes
- `formats`: ["html", "markdown"]
- `figures`: ["training_history", "confusion_matrix", "per_class", "calibration", "errors", "model_summary"]

### Runtime

- `device`: auto
- `deterministic`: yes
- `num_workers`: auto
- `persistent_workers`: yes
- `pin_memory`: auto
- `channels_last`: no
- `compile`: no

### Other

- `schema_version`: 1.0

## Data contract

- `channel_policy`: rgb
- `class_names`: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
- `dtype`: float32
- `input_shape`: [3, 32, 32]
- `label_map`: airplane=0, automobile=1, bird=2, cat=3, deer=4, dog=5, frog=6, horse=7, ship=8, truck=9
- `normalization.mean`: [0.4914, 0.4822, 0.4465]
- `normalization.std`: [0.247, 0.2435, 0.2616]
- `normalization.source`: supplied
- `stem_adaptation`: —
- `value_range`: [0.0, 1.0]

## Split integrity

| Split | Count | Fingerprint |
| --- | --- | --- |
| test | 10,000 | 7aac74439fdc9640 |
| train | 10,000 | 346aace5c0a25e9f |
| val | 2,000 | 71306f65f457e9fd |

Split mode: random

Group overlap: []

Duplicate overlap: []

## Model

- Total parameters: 482,058
- Trainable parameters: 482,058
- Input shape: [3, 32, 32]
- Output shape: [1, 10]

| Module | Type | Parameters |
| --- | --- | --- |
| model | ResidualCNN | 482,058 |
| features | Sequential | 480,768 |
| features.0 | Conv2d | 896 |
| features.1 | Identity | 0 |
| features.2 | ReLU | 0 |
| features.3 | ResidualBlock | 18,496 |
| features.3.conv1 | Conv2d | 9,248 |
| features.3.norm1 | Identity | 0 |
| features.3.conv2 | Conv2d | 9,248 |
| features.3.norm2 | Identity | 0 |
| features.4 | MaxPool2d | 0 |
| features.5 | Conv2d | 18,496 |
| features.6 | Identity | 0 |
| features.7 | ReLU | 0 |
| features.8 | ResidualBlock | 73,856 |
| features.8.conv1 | Conv2d | 36,928 |
| features.8.norm1 | Identity | 0 |
| features.8.conv2 | Conv2d | 36,928 |
| features.8.norm2 | Identity | 0 |
| features.9 | MaxPool2d | 0 |
| features.10 | Conv2d | 73,856 |
| features.11 | Identity | 0 |
| features.12 | ReLU | 0 |
| features.13 | ResidualBlock | 295,168 |

## Training

Epochs run: 8
Best validation accuracy: 0.467 (epoch 7)

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy | Learning rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.081 | 1.912 | 0.198 | 0.25 | 0.0009505 |
| 2 | 1.86 | 1.773 | 0.2795 | 0.315 | 0.0008117 |
| 3 | 1.689 | 1.72 | 0.349 | 0.3665 | 0.0006113 |
| 4 | 1.595 | 1.545 | 0.3889 | 0.4255 | 0.0003887 |
| 5 | 1.481 | 1.559 | 0.4362 | 0.4455 | 0.0001883 |
| 6 | 1.433 | 1.447 | 0.4589 | 0.464 | 4.952e-05 |
| 7 | 1.384 | 1.43 | 0.478 | 0.467 | 0 |
| 8 | 1.373 | 1.43 | 0.4834 | 0.467 | 0 |

## Test metrics

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| airplane | 0.467 | 0.434 | 0.450 | 1,000 |
| automobile | 0.617 | 0.676 | 0.645 | 1,000 |
| bird | 0.380 | 0.320 | 0.348 | 1,000 |
| cat | 0.331 | 0.207 | 0.255 | 1,000 |
| deer | 0.411 | 0.220 | 0.287 | 1,000 |
| dog | 0.402 | 0.628 | 0.490 | 1,000 |
| frog | 0.501 | 0.633 | 0.559 | 1,000 |
| horse | 0.481 | 0.624 | 0.544 | 1,000 |
| ship | 0.613 | 0.533 | 0.570 | 1,000 |
| truck | 0.591 | 0.580 | 0.585 | 1,000 |

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
- Config hash: 8a7bf7abe135c3f0
- Epochs run: 8
- Git commit: 42a1941 (dirty)
- Git branch: main

## Reproduction

```bash
cnncls train -c <config.yaml> --runs-dir runs  # produced 20261103T123447Z_ablation_no_batchnorm_8a7bf7ab
cnncls evaluate --run runs/20261103T123447Z_ablation_no_batchnorm_8a7bf7ab --checkpoint runs/20261103T123447Z_ablation_no_batchnorm_8a7bf7ab/checkpoints/best.pt
cnncls report --run runs/20261103T123447Z_ablation_no_batchnorm_8a7bf7ab
cnncls compare runs/20261103T123447Z_ablation_no_batchnorm_8a7bf7ab <other-run>
```
