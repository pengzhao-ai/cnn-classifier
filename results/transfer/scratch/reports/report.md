# Experiment report — cifar10_resnet18_scratch

- Run directory: `runs/20261103T114457Z_cifar10_resnet18_scratch_3da6ebe9`
- Status: completed
- Config hash: 3da6ebe93b4fc14f

## Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.909 |
| Top-k accuracy | 0.989 |
| Macro F1 | 0.909 |
| NLL | 0.343 |
| ECE | 0.051 |
| MCE | 0.253 |
| Brier score | 0.144 |
| Support | 10,000 |
| Accuracy interval | 0.950 CI [0.903, 0.914] |

| Prediction archive | Samples | Saved accuracy |
| --- | --- | --- |
| test_predictions.npz | 10,000 | 0.909 |

## Configuration

### Experiment

- `name`: cifar10_resnet18_scratch
- `tags`: ["cifar10", "transfer", "resnet18", "scratch"]
- `notes`: 32x32 ResNet-18 trained from scratch; fairness anchor control.
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
- `normalization.mean`: [0.485, 0.456, 0.406]
- `normalization.std`: [0.229, 0.224, 0.225]
- `normalization.cache`: yes
- `transforms.train`: [{"name": "RandomHorizontalFlip", "p": 0.5}, {"name": "RandomCrop", "padding": 4, "size": 32}, {"brightness": 0.1, "contrast": 0.1, "name": "ColorJitter"}, {"name": "ToTensor"}, {"mean": "auto", "name": "Normalize", "std": "auto"}, {"name": "RandomErasing", "p": 0.1}]
- `transforms.eval`: [{"name": "ToTensor"}, {"mean": "auto", "name": "Normalize", "std": "auto"}]
- `value_range.mode`: auto
- `value_range.min`: —
- `value_range.max`: —
- `adapter_params.dataset`: CIFAR10
- `adapter_params.download`: yes

### Model

- `name`: torchvision
- `params.model`: resnet18
- `params.stem_adaptation`: keep_7x7_stride1
- `params.dropout`: 0
- `transfer.weights`: —
- `transfer.freeze`: none
- `transfer.train_bn`: no
- `transfer.unfreeze_schedule`: []
- `transfer.param_groups`: []

### Training

- `epochs`: 30
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
- `normalization.mean`: [0.485, 0.456, 0.406]
- `normalization.std`: [0.229, 0.224, 0.225]
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

- Total parameters: 11,181,642
- Trainable parameters: 11,181,642
- Input shape: [3, 32, 32]
- Output shape: [1, 10]

| Module | Type | Parameters |
| --- | --- | --- |
| model | ResNet | 11,181,642 |
| conv1 | Conv2d | 9,408 |
| bn1 | BatchNorm2d | 128 |
| relu | ReLU | 0 |
| maxpool | Identity | 0 |
| layer1 | Sequential | 147,968 |
| layer1.0 | BasicBlock | 73,984 |
| layer1.0.conv1 | Conv2d | 36,864 |
| layer1.0.bn1 | BatchNorm2d | 128 |
| layer1.0.relu | ReLU | 0 |
| layer1.0.conv2 | Conv2d | 36,864 |
| layer1.0.bn2 | BatchNorm2d | 128 |
| layer1.1 | BasicBlock | 73,984 |
| layer1.1.conv1 | Conv2d | 36,864 |
| layer1.1.bn1 | BatchNorm2d | 128 |
| layer1.1.relu | ReLU | 0 |
| layer1.1.conv2 | Conv2d | 36,864 |
| layer1.1.bn2 | BatchNorm2d | 128 |
| layer2 | Sequential | 525,568 |
| layer2.0 | BasicBlock | 230,144 |
| layer2.0.conv1 | Conv2d | 73,728 |
| layer2.0.bn1 | BatchNorm2d | 256 |
| layer2.0.relu | ReLU | 0 |
| layer2.0.conv2 | Conv2d | 147,456 |

## Training

Epochs run: 30
Best validation accuracy: 0.912 (epoch 26)

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy | Learning rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.548 | 1.785 | 0.433 | 0.4504 | 0.0009971 |
| 2 | 1.084 | 0.9443 | 0.6099 | 0.671 | 0.0009883 |
| 3 | 0.8455 | 0.8528 | 0.7001 | 0.7062 | 0.0009738 |
| 4 | 0.7188 | 0.7063 | 0.7511 | 0.755 | 0.0009538 |
| 5 | 0.6345 | 0.6314 | 0.7789 | 0.7828 | 0.0009284 |
| 6 | 0.5697 | 0.6725 | 0.8034 | 0.781 | 0.000898 |
| 7 | 0.513 | 0.5868 | 0.8214 | 0.8086 | 0.000863 |
| 8 | 0.4707 | 0.6329 | 0.8364 | 0.8048 | 0.0008237 |
| 9 | 0.4305 | 0.4804 | 0.8518 | 0.8384 | 0.0007806 |
| 10 | 0.3989 | 0.4098 | 0.8617 | 0.86 | 0.0007342 |
| 11 | 0.3596 | 0.4712 | 0.8743 | 0.8442 | 0.0006851 |
| 12 | 0.33 | 0.3983 | 0.8857 | 0.8686 | 0.0006338 |
| 13 | 0.3027 | 0.3972 | 0.8944 | 0.87 | 0.0005809 |
| 14 | 0.2747 | 0.4523 | 0.9045 | 0.8588 | 0.0005271 |
| 15 | 0.2499 | 0.3698 | 0.9139 | 0.8826 | 0.0004729 |
| 16 | 0.2282 | 0.3568 | 0.9193 | 0.8832 | 0.0004191 |
| 17 | 0.2048 | 0.3558 | 0.9283 | 0.8892 | 0.0003662 |
| 18 | 0.1834 | 0.3491 | 0.9361 | 0.893 | 0.0003149 |
| 19 | 0.1621 | 0.3439 | 0.9438 | 0.897 | 0.0002658 |
| 20 | 0.1476 | 0.3286 | 0.9485 | 0.8992 | 0.0002194 |
| 21 | 0.1329 | 0.3333 | 0.954 | 0.9038 | 0.0001763 |
| 22 | 0.1144 | 0.328 | 0.9598 | 0.9062 | 0.000137 |
| 23 | 0.1047 | 0.3374 | 0.9634 | 0.9056 | 0.000102 |
| 24 | 0.09416 | 0.335 | 0.9678 | 0.9096 | 7.157e-05 |
| 25 | 0.08412 | 0.3344 | 0.9714 | 0.9078 | 4.621e-05 |
| 26 | 0.07978 | 0.3205 | 0.9734 | 0.9122 | 2.617e-05 |
| 27 | 0.07741 | 0.3251 | 0.9748 | 0.9102 | 1.169e-05 |
| 28 | 0.07359 | 0.325 | 0.9756 | 0.9092 | 2.931e-06 |
| 29 | 0.07246 | 0.3255 | 0.9756 | 0.9102 | 0 |
| 30 | 0.07054 | 0.3226 | 0.9764 | 0.9104 | 0 |

## Test metrics

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| airplane | 0.903 | 0.918 | 0.910 | 1,000 |
| automobile | 0.961 | 0.961 | 0.961 | 1,000 |
| bird | 0.870 | 0.871 | 0.871 | 1,000 |
| cat | 0.801 | 0.813 | 0.807 | 1,000 |
| deer | 0.879 | 0.928 | 0.903 | 1,000 |
| dog | 0.870 | 0.859 | 0.865 | 1,000 |
| frog | 0.947 | 0.930 | 0.938 | 1,000 |
| horse | 0.956 | 0.915 | 0.935 | 1,000 |
| ship | 0.952 | 0.952 | 0.952 | 1,000 |
| truck | 0.955 | 0.941 | 0.948 | 1,000 |

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
- Config hash: 3da6ebe93b4fc14f
- Epochs run: 30
- Git commit: 42a1941
- Git branch: main
- Mean latency (ms): 3.837
- p50 latency (ms): 3.712
- p95 latency (ms): 4.616
- Throughput (images/s): 260.594

## Reproduction

```bash
cnncls train -c <config.yaml> --runs-dir runs  # produced 20261103T114457Z_cifar10_resnet18_scratch_3da6ebe9
cnncls evaluate --run runs/20261103T114457Z_cifar10_resnet18_scratch_3da6ebe9 --checkpoint runs/20261103T114457Z_cifar10_resnet18_scratch_3da6ebe9/checkpoints/best.pt
cnncls report --run runs/20261103T114457Z_cifar10_resnet18_scratch_3da6ebe9
cnncls compare runs/20261103T114457Z_cifar10_resnet18_scratch_3da6ebe9 <other-run>
```
