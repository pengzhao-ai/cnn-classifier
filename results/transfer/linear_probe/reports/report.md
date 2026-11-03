# Experiment report — cifar10_resnet18_linear_probe

- Run directory: `/Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T120432Z_cifar10_resnet18_linear_probe_97e9ed00`
- Status: completed
- Config hash: 97e9ed0020862733

## Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.627 |
| Top-k accuracy | 0.885 |
| Macro F1 | 0.625 |
| NLL | 1.081 |
| ECE | 0.011 |
| MCE | 0.045 |
| Brier score | 0.502 |
| Support | 10,000 |
| Accuracy interval | 0.950 CI [0.618, 0.636] |

| Prediction archive | Samples | Saved accuracy |
| --- | --- | --- |
| test_predictions.npz | 10,000 | 0.627 |

## Configuration

### Experiment

- `name`: cifar10_resnet18_linear_probe
- `tags`: ["cifar10", "transfer", "resnet18", "linear-probe"]
- `notes`: Frozen ImageNet ResNet-18 backbone with a trained linear head.
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
- `transfer.freeze`: backbone
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
- Trainable parameters: 5,130
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
Best validation accuracy: 0.628 (epoch 23)

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy | Learning rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.499 | 1.218 | 0.484 | 0.583 | 0.0009971 |
| 2 | 1.279 | 1.16 | 0.5585 | 0.6014 | 0.0009883 |
| 3 | 1.246 | 1.138 | 0.5687 | 0.6042 | 0.0009738 |
| 4 | 1.235 | 1.162 | 0.5713 | 0.5984 | 0.0009538 |
| 5 | 1.221 | 1.135 | 0.5781 | 0.6086 | 0.0009284 |
| 6 | 1.218 | 1.15 | 0.5782 | 0.6064 | 0.000898 |
| 7 | 1.203 | 1.138 | 0.5836 | 0.611 | 0.000863 |
| 8 | 1.203 | 1.118 | 0.5843 | 0.6124 | 0.0008237 |
| 9 | 1.203 | 1.128 | 0.5849 | 0.6074 | 0.0007806 |
| 10 | 1.194 | 1.111 | 0.5876 | 0.613 | 0.0007342 |
| 11 | 1.194 | 1.101 | 0.5857 | 0.6166 | 0.0006851 |
| 12 | 1.192 | 1.105 | 0.5848 | 0.6138 | 0.0006338 |
| 13 | 1.184 | 1.109 | 0.588 | 0.6218 | 0.0005809 |
| 14 | 1.184 | 1.108 | 0.5915 | 0.6158 | 0.0005271 |
| 15 | 1.178 | 1.108 | 0.5933 | 0.6182 | 0.0004729 |
| 16 | 1.183 | 1.103 | 0.5924 | 0.6194 | 0.0004191 |
| 17 | 1.174 | 1.104 | 0.5942 | 0.6218 | 0.0003662 |
| 18 | 1.179 | 1.08 | 0.589 | 0.6268 | 0.0003149 |
| 19 | 1.179 | 1.088 | 0.5935 | 0.6226 | 0.0002658 |
| 20 | 1.171 | 1.092 | 0.5951 | 0.6224 | 0.0002194 |
| 21 | 1.174 | 1.084 | 0.5948 | 0.6256 | 0.0001763 |
| 22 | 1.169 | 1.087 | 0.5966 | 0.6224 | 0.000137 |
| 23 | 1.165 | 1.078 | 0.5993 | 0.6284 | 0.000102 |
| 24 | 1.168 | 1.08 | 0.5951 | 0.6254 | 7.157e-05 |
| 25 | 1.162 | 1.088 | 0.5966 | 0.623 | 4.621e-05 |
| 26 | 1.162 | 1.086 | 0.5992 | 0.6254 | 2.617e-05 |
| 27 | 1.167 | 1.079 | 0.5961 | 0.6268 | 1.169e-05 |
| 28 | 1.165 | 1.081 | 0.6004 | 0.6248 | 2.931e-06 |
| 29 | 1.157 | 1.087 | 0.6032 | 0.6234 | 0 |
| 30 | 1.157 | 1.081 | 0.6002 | 0.627 | 0 |

## Test metrics

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| airplane | 0.616 | 0.699 | 0.655 | 1,000 |
| automobile | 0.722 | 0.714 | 0.718 | 1,000 |
| bird | 0.526 | 0.466 | 0.494 | 1,000 |
| cat | 0.489 | 0.435 | 0.460 | 1,000 |
| deer | 0.620 | 0.585 | 0.602 | 1,000 |
| dog | 0.541 | 0.626 | 0.580 | 1,000 |
| frog | 0.652 | 0.707 | 0.678 | 1,000 |
| horse | 0.640 | 0.668 | 0.654 | 1,000 |
| ship | 0.761 | 0.685 | 0.721 | 1,000 |
| truck | 0.701 | 0.681 | 0.691 | 1,000 |

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
- Config hash: 97e9ed0020862733
- Epochs run: 30
- Git commit: 42a1941 (dirty)
- Git branch: main
- Mean latency (ms): 3.749
- p50 latency (ms): 3.722
- p95 latency (ms): 4.045
- Throughput (images/s): 266.730

## Reproduction

```bash
cnncls train -c <config.yaml>  # produced 20261103T120432Z_cifar10_resnet18_linear_probe_97e9ed00
cnncls evaluate --run /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T120432Z_cifar10_resnet18_linear_probe_97e9ed00 --checkpoint /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T120432Z_cifar10_resnet18_linear_probe_97e9ed00/checkpoints/best.pt
cnncls report --run /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T120432Z_cifar10_resnet18_linear_probe_97e9ed00 --format html
cnncls compare /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T120432Z_cifar10_resnet18_linear_probe_97e9ed00 <other-run>
```
