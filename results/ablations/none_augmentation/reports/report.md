# Experiment report — ablation_no_augmentation

- Run directory: `/Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T123408Z_ablation_no_augmentation_cf3b3f52`
- Status: completed
- Config hash: cf3b3f524a5a4e2c

## Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.711 |
| Top-k accuracy | 0.930 |
| Macro F1 | 0.710 |
| NLL | 0.818 |
| ECE | 0.016 |
| MCE | 0.102 |
| Brier score | 0.395 |
| Support | 10,000 |
| Accuracy interval | 0.950 CI [0.702, 0.719] |

| Prediction archive | Samples | Saved accuracy |
| --- | --- | --- |
| test_predictions.npz | 10,000 | 0.711 |

## Configuration

### Experiment

- `name`: ablation_no_augmentation
- `tags`: ["ablation", "augmentation"]
- `notes`: Augmentation off: tensor conversion and normalization only.
- `seed`: 42

### Data

- `adapter`: torchvision
- `root`: /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/data
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
- `transforms.train`: [{"name": "ToTensor"}, {"mean": "auto", "name": "Normalize", "std": "auto"}]
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
- `params.norm`: batch
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

- Total parameters: 482,730
- Trainable parameters: 482,730
- Input shape: [3, 32, 32]
- Output shape: [1, 10]

| Module | Type | Parameters |
| --- | --- | --- |
| model | ResidualCNN | 482,730 |
| features | Sequential | 481,440 |
| features.0 | Conv2d | 864 |
| features.1 | BatchNorm2d | 64 |
| features.2 | ReLU | 0 |
| features.3 | ResidualBlock | 18,560 |
| features.3.conv1 | Conv2d | 9,216 |
| features.3.norm1 | BatchNorm2d | 64 |
| features.3.conv2 | Conv2d | 9,216 |
| features.3.norm2 | BatchNorm2d | 64 |
| features.4 | MaxPool2d | 0 |
| features.5 | Conv2d | 18,432 |
| features.6 | BatchNorm2d | 128 |
| features.7 | ReLU | 0 |
| features.8 | ResidualBlock | 73,984 |
| features.8.conv1 | Conv2d | 36,864 |
| features.8.norm1 | BatchNorm2d | 128 |
| features.8.conv2 | Conv2d | 36,864 |
| features.8.norm2 | BatchNorm2d | 128 |
| features.9 | MaxPool2d | 0 |
| features.10 | Conv2d | 73,728 |
| features.11 | BatchNorm2d | 256 |
| features.12 | ReLU | 0 |
| features.13 | ResidualBlock | 295,424 |

## Training

Epochs run: 8
Best validation accuracy: 0.709 (epoch 8)

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy | Learning rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.635 | 3.057 | 0.3882 | 0.2485 | 0.0009505 |
| 2 | 1.292 | 1.78 | 0.5342 | 0.4455 | 0.0008117 |
| 3 | 1.089 | 1.254 | 0.6047 | 0.539 | 0.0006113 |
| 4 | 0.9394 | 1.252 | 0.6652 | 0.5655 | 0.0003887 |
| 5 | 0.8151 | 1.122 | 0.7149 | 0.5955 | 0.0001883 |
| 6 | 0.7026 | 0.8913 | 0.7592 | 0.682 | 4.952e-05 |
| 7 | 0.6268 | 0.8471 | 0.7894 | 0.6985 | 0 |
| 8 | 0.595 | 0.8305 | 0.8048 | 0.7085 | 0 |

## Test metrics

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| airplane | 0.731 | 0.743 | 0.737 | 1,000 |
| automobile | 0.861 | 0.861 | 0.861 | 1,000 |
| bird | 0.581 | 0.592 | 0.586 | 1,000 |
| cat | 0.503 | 0.507 | 0.505 | 1,000 |
| deer | 0.663 | 0.609 | 0.635 | 1,000 |
| dog | 0.619 | 0.569 | 0.593 | 1,000 |
| frog | 0.737 | 0.794 | 0.765 | 1,000 |
| horse | 0.741 | 0.757 | 0.749 | 1,000 |
| ship | 0.821 | 0.834 | 0.827 | 1,000 |
| truck | 0.834 | 0.840 | 0.837 | 1,000 |

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
- Config hash: cf3b3f524a5a4e2c
- Epochs run: 8
- Git commit: 42a1941 (dirty)
- Git branch: main

## Reproduction

```bash
cnncls train -c <config.yaml>  # produced 20261103T123408Z_ablation_no_augmentation_cf3b3f52
cnncls evaluate --run /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T123408Z_ablation_no_augmentation_cf3b3f52 --checkpoint /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T123408Z_ablation_no_augmentation_cf3b3f52/checkpoints/best.pt
cnncls report --run /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T123408Z_ablation_no_augmentation_cf3b3f52 --format html
cnncls compare /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T123408Z_ablation_no_augmentation_cf3b3f52 <other-run>
```
