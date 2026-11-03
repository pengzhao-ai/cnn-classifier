# Experiment report — cifar10_resnet18_finetune

- Run directory: `runs/20261103T121258Z_cifar10_resnet18_finetune_4ec58124`
- Status: completed
- Config hash: 4ec5812470045bf0

## Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.950 |
| Top-k accuracy | 0.995 |
| Macro F1 | 0.950 |
| NLL | 0.221 |
| ECE | 0.032 |
| MCE | 0.279 |
| Brier score | 0.081 |
| Support | 10,000 |
| Accuracy interval | 0.950 CI [0.946, 0.954] |

| Prediction archive | Samples | Saved accuracy |
| --- | --- | --- |
| test_predictions.npz | 10,000 | 0.950 |

## Configuration

### Experiment

- `name`: cifar10_resnet18_finetune
- `tags`: ["cifar10", "transfer", "resnet18", "finetune"]
- `notes`: ImageNet ResNet-18 fine-tuned end to end with a higher head learning rate.
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
- `transfer.weights`: IMAGENET1K_V1
- `transfer.freeze`: none
- `transfer.train_bn`: no
- `transfer.unfreeze_schedule`: []
- `transfer.param_groups`: [{"lr_multiplier": 10.0, "pattern": "^fc\\.", "weight_decay_multiplier": 1.0}]

### Training

- `epochs`: 30
- `batch_size`: 128
- `grad_accum_steps`: 1
- `selection_metric`: val_accuracy
- `optimizer.name`: adamw
- `optimizer.lr`: 0.0001
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
Best validation accuracy: 0.956 (epoch 27)

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy | Learning rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.6676 | 0.3536 | 0.7681 | 0.8772 | 9.971e-05 |
| 2 | 0.3354 | 0.2693 | 0.8847 | 0.9072 | 9.883e-05 |
| 3 | 0.2425 | 0.2222 | 0.9168 | 0.927 | 9.738e-05 |
| 4 | 0.2001 | 0.2227 | 0.9318 | 0.9262 | 9.538e-05 |
| 5 | 0.1647 | 0.2069 | 0.9426 | 0.933 | 9.284e-05 |
| 6 | 0.1433 | 0.1959 | 0.9506 | 0.939 | 8.98e-05 |
| 7 | 0.1184 | 0.2242 | 0.9591 | 0.9332 | 8.63e-05 |
| 8 | 0.1088 | 0.2123 | 0.9623 | 0.9366 | 8.237e-05 |
| 9 | 0.09552 | 0.2203 | 0.9675 | 0.9348 | 7.806e-05 |
| 10 | 0.08591 | 0.2241 | 0.9704 | 0.9388 | 7.342e-05 |
| 11 | 0.07222 | 0.2081 | 0.975 | 0.942 | 6.851e-05 |
| 12 | 0.06753 | 0.2147 | 0.977 | 0.9436 | 6.338e-05 |
| 13 | 0.06213 | 0.2131 | 0.979 | 0.9436 | 5.809e-05 |
| 14 | 0.05669 | 0.2097 | 0.9804 | 0.9458 | 5.271e-05 |
| 15 | 0.05108 | 0.2088 | 0.982 | 0.9466 | 4.729e-05 |
| 16 | 0.04366 | 0.1889 | 0.9851 | 0.949 | 4.191e-05 |
| 17 | 0.04204 | 0.2046 | 0.9858 | 0.9492 | 3.662e-05 |
| 18 | 0.03783 | 0.1985 | 0.9872 | 0.9522 | 3.149e-05 |
| 19 | 0.03616 | 0.2015 | 0.9874 | 0.9526 | 2.658e-05 |
| 20 | 0.03384 | 0.2025 | 0.9885 | 0.9492 | 2.194e-05 |
| 21 | 0.03183 | 0.1939 | 0.99 | 0.9544 | 1.763e-05 |
| 22 | 0.02785 | 0.1876 | 0.9904 | 0.9528 | 1.37e-05 |
| 23 | 0.02606 | 0.1888 | 0.9914 | 0.9552 | 1.02e-05 |
| 24 | 0.02575 | 0.1912 | 0.9917 | 0.9544 | 7.157e-06 |
| 25 | 0.02395 | 0.1856 | 0.9916 | 0.9548 | 4.621e-06 |
| 26 | 0.02489 | 0.1816 | 0.9917 | 0.9544 | 2.617e-06 |
| 27 | 0.02393 | 0.1857 | 0.9923 | 0.9558 | 1.169e-06 |
| 28 | 0.02276 | 0.1852 | 0.9928 | 0.955 | 2.931e-07 |
| 29 | 0.02267 | 0.184 | 0.9924 | 0.955 | 0 |
| 30 | 0.0205 | 0.1852 | 0.9936 | 0.955 | 0 |

## Test metrics

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| airplane | 0.961 | 0.962 | 0.962 | 1,000 |
| automobile | 0.970 | 0.971 | 0.971 | 1,000 |
| bird | 0.931 | 0.950 | 0.941 | 1,000 |
| cat | 0.904 | 0.878 | 0.891 | 1,000 |
| deer | 0.957 | 0.947 | 0.952 | 1,000 |
| dog | 0.917 | 0.910 | 0.914 | 1,000 |
| frog | 0.960 | 0.977 | 0.968 | 1,000 |
| horse | 0.965 | 0.961 | 0.963 | 1,000 |
| ship | 0.962 | 0.981 | 0.971 | 1,000 |
| truck | 0.970 | 0.961 | 0.965 | 1,000 |

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
- Config hash: 4ec5812470045bf0
- Epochs run: 30
- Git commit: 42a1941 (dirty)
- Git branch: main
- Mean latency (ms): 3.704
- p50 latency (ms): 3.724
- p95 latency (ms): 3.835
- Throughput (images/s): 269.974

## Reproduction

```bash
cnncls train -c <config.yaml> --runs-dir runs  # produced 20261103T121258Z_cifar10_resnet18_finetune_4ec58124
cnncls evaluate --run runs/20261103T121258Z_cifar10_resnet18_finetune_4ec58124 --checkpoint runs/20261103T121258Z_cifar10_resnet18_finetune_4ec58124/checkpoints/best.pt
cnncls report --run runs/20261103T121258Z_cifar10_resnet18_finetune_4ec58124
cnncls compare runs/20261103T121258Z_cifar10_resnet18_finetune_4ec58124 <other-run>
```
