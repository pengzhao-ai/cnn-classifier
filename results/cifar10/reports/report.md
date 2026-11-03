# Experiment report — cifar10_scratch_baseline

- Run directory: `runs/20261103T055515Z_cifar10_scratch_baseline_0283f0f0`
- Status: completed
- Config hash: 0283f0f08d57f258

## Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.922 |
| Top-k accuracy | 0.989 |
| Macro F1 | 0.923 |
| NLL | 0.289 |
| ECE | 0.043 |
| MCE | 0.237 |
| Brier score | 0.123 |
| Support | 10,000 |
| Accuracy interval | 0.950 CI [0.918, 0.928] |

| Prediction archive | Samples | Saved accuracy |
| --- | --- | --- |
| test_predictions.npz | 10,000 | 0.922 |

## Configuration

### Experiment

- `name`: cifar10_scratch_baseline
- `tags`: ["cifar10", "scratch", "baseline", "reference"]
- `notes`: Reference residual-CNN recipe from the project design; not state of the art.
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
- `transforms.train`: [{"name": "RandomHorizontalFlip", "p": 0.5}, {"name": "RandomCrop", "padding": 4, "size": 32}, {"brightness": 0.1, "contrast": 0.1, "name": "ColorJitter"}, {"name": "ToTensor"}, {"mean": "auto", "name": "Normalize", "std": "auto"}, {"name": "RandomErasing", "p": 0.1}]
- `transforms.eval`: [{"name": "ToTensor"}, {"mean": "auto", "name": "Normalize", "std": "auto"}]
- `value_range.mode`: auto
- `value_range.min`: —
- `value_range.max`: —
- `adapter_params.dataset`: CIFAR10
- `adapter_params.download`: yes

### Model

- `name`: residual_cnn
- `params.stem.channels`: 64
- `params.stem.kernel_size`: 3
- `params.stem.stride`: 1
- `params.stem.pool`: no
- `params.stages`: [{"blocks": 1, "channels": 64, "conv": false, "downsample": "maxpool"}, {"blocks": 1, "channels": 128, "conv": true, "downsample": "maxpool"}, {"blocks": 1, "channels": 256, "conv": true, "downsample": "maxpool"}, {"blocks": 1, "channels": 512, "conv": true, "downsample": "none"}]
- `params.dropout`: 0.3
- `params.norm`: batch
- `params.residual`: yes
- `transfer.weights`: —
- `transfer.freeze`: none
- `transfer.train_bn`: no
- `transfer.unfreeze_schedule`: []
- `transfer.param_groups`: []

### Training

- `epochs`: 50
- `batch_size`: 128
- `grad_accum_steps`: 1
- `selection_metric`: val_accuracy
- `optimizer.name`: adam
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
- `latency.enabled`: yes
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
| train | 45,000 | 7b1e5f5d9704b897 |
| val | 5,000 | ce06d441c20dc849 |

Split mode: random

Group overlap: []

Duplicate overlap: []

## Model

- Total parameters: 7,827,786
- Trainable parameters: 7,827,786
- Input shape: [3, 32, 32]
- Output shape: [1, 10]

| Module | Type | Parameters |
| --- | --- | --- |
| model | ResidualCNN | 7,827,786 |
| features | Sequential | 7,822,656 |
| features.0 | Conv2d | 1,728 |
| features.1 | BatchNorm2d | 128 |
| features.2 | ReLU | 0 |
| features.3 | ResidualBlock | 73,984 |
| features.3.conv1 | Conv2d | 36,864 |
| features.3.norm1 | BatchNorm2d | 128 |
| features.3.conv2 | Conv2d | 36,864 |
| features.3.norm2 | BatchNorm2d | 128 |
| features.4 | MaxPool2d | 0 |
| features.5 | Conv2d | 73,728 |
| features.6 | BatchNorm2d | 256 |
| features.7 | ReLU | 0 |
| features.8 | ResidualBlock | 295,424 |
| features.8.conv1 | Conv2d | 147,456 |
| features.8.norm1 | BatchNorm2d | 256 |
| features.8.conv2 | Conv2d | 147,456 |
| features.8.norm2 | BatchNorm2d | 256 |
| features.9 | MaxPool2d | 0 |
| features.10 | Conv2d | 294,912 |
| features.11 | BatchNorm2d | 512 |
| features.12 | ReLU | 0 |
| features.13 | ResidualBlock | 1,180,672 |

## Training

Epochs run: 50
Best validation accuracy: 0.928 (epoch 46)

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy | Learning rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.432 | 1.519 | 0.4731 | 0.5088 | 0.000999 |
| 2 | 0.9987 | 1.023 | 0.6447 | 0.656 | 0.0009959 |
| 3 | 0.8082 | 0.9863 | 0.7181 | 0.677 | 0.0009908 |
| 4 | 0.7082 | 0.8223 | 0.7554 | 0.727 | 0.0009836 |
| 5 | 0.6433 | 0.773 | 0.781 | 0.7392 | 0.0009745 |
| 6 | 0.5942 | 0.7969 | 0.7976 | 0.742 | 0.0009635 |
| 7 | 0.5483 | 0.547 | 0.8164 | 0.8112 | 0.0009505 |
| 8 | 0.5238 | 0.7146 | 0.8241 | 0.7668 | 0.0009357 |
| 9 | 0.4872 | 0.6752 | 0.8348 | 0.7862 | 0.000919 |
| 10 | 0.4617 | 0.4966 | 0.8423 | 0.8322 | 0.0009007 |
| 11 | 0.4337 | 0.5489 | 0.8534 | 0.8168 | 0.0008807 |
| 12 | 0.4204 | 0.439 | 0.8577 | 0.8556 | 0.0008592 |
| 13 | 0.3904 | 0.438 | 0.8678 | 0.8486 | 0.0008362 |
| 14 | 0.3795 | 0.4972 | 0.8718 | 0.839 | 0.0008117 |
| 15 | 0.3674 | 0.4634 | 0.8751 | 0.844 | 0.0007861 |
| 16 | 0.3432 | 0.5354 | 0.8826 | 0.8268 | 0.0007592 |
| 17 | 0.3284 | 0.4961 | 0.8888 | 0.8392 | 0.0007313 |
| 18 | 0.3168 | 0.4371 | 0.8926 | 0.8506 | 0.0007024 |
| 19 | 0.3031 | 0.3903 | 0.8978 | 0.8714 | 0.0006727 |
| 20 | 0.2903 | 0.5495 | 0.9013 | 0.8272 | 0.0006423 |
| 21 | 0.2762 | 0.4264 | 0.9071 | 0.8662 | 0.0006113 |
| 22 | 0.2609 | 0.368 | 0.9105 | 0.8834 | 0.0005798 |
| 23 | 0.2488 | 0.3587 | 0.9156 | 0.8864 | 0.000548 |
| 24 | 0.2386 | 0.3675 | 0.9191 | 0.88 | 0.000516 |
| 25 | 0.2275 | 0.3698 | 0.9228 | 0.8778 | 0.000484 |
| 26 | 0.2136 | 0.3561 | 0.9268 | 0.8926 | 0.000452 |
| 27 | 0.195 | 0.3271 | 0.933 | 0.8898 | 0.0004202 |
| 28 | 0.191 | 0.3307 | 0.9353 | 0.893 | 0.0003887 |
| 29 | 0.177 | 0.3636 | 0.9401 | 0.8932 | 0.0003577 |
| 30 | 0.1673 | 0.3539 | 0.9429 | 0.8946 | 0.0003273 |
| 31 | 0.1503 | 0.333 | 0.9494 | 0.8988 | 0.0002976 |
| 32 | 0.1444 | 0.3235 | 0.9512 | 0.9032 | 0.0002687 |
| 33 | 0.1375 | 0.3197 | 0.9544 | 0.9034 | 0.0002408 |
| 34 | 0.1256 | 0.3346 | 0.9582 | 0.8998 | 0.0002139 |
| 35 | 0.1127 | 0.3251 | 0.9623 | 0.9054 | 0.0001883 |
| 36 | 0.1059 | 0.3016 | 0.9645 | 0.911 | 0.0001638 |
| 37 | 0.09329 | 0.3026 | 0.9693 | 0.911 | 0.0001408 |
| 38 | 0.09066 | 0.3095 | 0.9703 | 0.9138 | 0.0001193 |
| 39 | 0.08011 | 0.287 | 0.9744 | 0.92 | 9.929e-05 |
| 40 | 0.0726 | 0.2836 | 0.9756 | 0.9254 | 8.096e-05 |
| 41 | 0.06761 | 0.2903 | 0.9789 | 0.9194 | 6.434e-05 |
| 42 | 0.06486 | 0.2899 | 0.979 | 0.919 | 4.952e-05 |
| 43 | 0.05945 | 0.2794 | 0.982 | 0.9226 | 3.654e-05 |
| 44 | 0.05661 | 0.2785 | 0.9826 | 0.924 | 2.547e-05 |
| 45 | 0.05262 | 0.2808 | 0.984 | 0.9262 | 1.635e-05 |
| 46 | 0.05147 | 0.2804 | 0.984 | 0.9278 | 9.22e-06 |
| 47 | 0.04846 | 0.276 | 0.9851 | 0.9264 | 4.105e-06 |
| 48 | 0.04897 | 0.2778 | 0.9845 | 0.9268 | 1.027e-06 |
| 49 | 0.0504 | 0.2782 | 0.9841 | 0.9252 | 0 |
| 50 | 0.04964 | 0.2802 | 0.9842 | 0.9266 | 0 |

## Test metrics

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| airplane | 0.908 | 0.925 | 0.916 | 1,000 |
| automobile | 0.975 | 0.962 | 0.968 | 1,000 |
| bird | 0.888 | 0.892 | 0.890 | 1,000 |
| cat | 0.840 | 0.862 | 0.851 | 1,000 |
| deer | 0.923 | 0.934 | 0.928 | 1,000 |
| dog | 0.889 | 0.871 | 0.880 | 1,000 |
| frog | 0.949 | 0.945 | 0.947 | 1,000 |
| horse | 0.962 | 0.934 | 0.948 | 1,000 |
| ship | 0.957 | 0.942 | 0.950 | 1,000 |
| truck | 0.938 | 0.958 | 0.948 | 1,000 |

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
- Config hash: 0283f0f08d57f258
- Epochs run: 50
- Git commit: d42b24e (dirty)
- Git branch: main
- Mean latency (ms): 2.081
- p50 latency (ms): 1.987
- p95 latency (ms): 2.519
- Throughput (images/s): 480.565

## Reproduction

```bash
cnncls train -c <config.yaml> --runs-dir runs  # produced 20261103T055515Z_cifar10_scratch_baseline_0283f0f0
cnncls evaluate --run runs/20261103T055515Z_cifar10_scratch_baseline_0283f0f0 --checkpoint runs/20261103T055515Z_cifar10_scratch_baseline_0283f0f0/checkpoints/best.pt
cnncls report --run runs/20261103T055515Z_cifar10_scratch_baseline_0283f0f0
cnncls compare runs/20261103T055515Z_cifar10_scratch_baseline_0283f0f0 <other-run>
```
