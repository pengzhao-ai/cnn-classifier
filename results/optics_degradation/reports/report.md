# Experiment report — optics_degradation_classifier

- Run directory: `/Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T123724Z_optics_degradation_classifier_06cb9622`
- Status: completed
- Config hash: 06cb9622a1012df5

## Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.857 |
| Top-k accuracy | 0.981 |
| Macro F1 | 0.856 |
| NLL | 0.448 |
| ECE | 0.099 |
| MCE | 0.228 |
| Brier score | 0.232 |
| Support | 420 |
| Accuracy interval | 0.950 CI [0.824, 0.888] |

| Prediction archive | Samples | Saved accuracy |
| --- | --- | --- |
| test_predictions.npz | 420 | 0.857 |

## Configuration

### Experiment

- `name`: optics_degradation_classifier
- `tags`: ["optics", "degradation", "manifest", "diagnostics"]
- `notes`: Classify the dominant optical degradation in generated 32x32 samples.
- `seed`: 1

### Data

- `adapter`: manifest
- `root`: —
- `manifest`: /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/data/optics_degradation/manifest.csv
- `image_root`: /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/data/optics_degradation
- `class_names`: ["clean", "defocus", "motion_blur", "chromatic_aberration", "vignetting", "sensor_noise", "jpeg"]
- `num_classes`: 7
- `channels`: rgb
- `image_size`: [32, 32]
- `split.mode`: explicit
- `split.val_fraction`: 0.1
- `split.seed`: 1
- `split.group_column`: —
- `normalization.mode`: compute
- `normalization.mean`: —
- `normalization.std`: —
- `normalization.cache`: yes
- `transforms.train`: [{"name": "Resize", "size": [32, 32]}, {"name": "ToTensor"}, {"mean": "auto", "name": "Normalize", "std": "auto"}]
- `transforms.eval`: [{"name": "Resize", "size": [32, 32]}, {"name": "ToTensor"}, {"mean": "auto", "name": "Normalize", "std": "auto"}]
- `value_range.mode`: auto
- `value_range.min`: —
- `value_range.max`: —

### Model

- `name`: simple_cnn
- `params.channels`: [32, 64, 128]
- `params.dropout`: 0.2
- `params.norm`: batch
- `transfer.weights`: —
- `transfer.freeze`: none
- `transfer.train_bn`: no
- `transfer.unfreeze_schedule`: []
- `transfer.param_groups`: []

### Training

- `epochs`: 20
- `batch_size`: 64
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
- `early_stopping.patience`: 5
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
- `class_names`: ["clean", "defocus", "motion_blur", "chromatic_aberration", "vignetting", "sensor_noise", "jpeg"]
- `dtype`: float32
- `input_shape`: [3, 32, 32]
- `label_map`: chromatic_aberration=3, clean=0, defocus=1, jpeg=6, motion_blur=2, sensor_noise=5, vignetting=4
- `normalization.mean`: [0.45008078599656304, 0.44376823804568777, 0.41168341724981666]
- `normalization.std`: [0.24959659829645284, 0.24001520784474484, 0.25475223433798966]
- `normalization.source`: computed
- `stem_adaptation`: —
- `value_range`: [0.0, 1.0]

## Split integrity

| Split | Count | Fingerprint |
| --- | --- | --- |
| test | 420 | cae1475a686f8bc4 |
| train | 1,260 | c68481f200ed71b6 |
| val | 420 | 6fc14f1d198b83b2 |

Split mode: explicit

Group overlap: []

Duplicate overlap: []

## Model

- Total parameters: 94,375
- Trainable parameters: 94,375
- Input shape: [3, 32, 32]
- Output shape: [1, 7]

| Module | Type | Parameters |
| --- | --- | --- |
| model | SimpleCNN | 94,375 |
| features | Sequential | 93,472 |
| features.0 | Conv2d | 864 |
| features.1 | BatchNorm2d | 64 |
| features.2 | ReLU | 0 |
| features.3 | MaxPool2d | 0 |
| features.4 | Conv2d | 18,432 |
| features.5 | BatchNorm2d | 128 |
| features.6 | ReLU | 0 |
| features.7 | MaxPool2d | 0 |
| features.8 | Conv2d | 73,728 |
| features.9 | BatchNorm2d | 256 |
| features.10 | ReLU | 0 |
| features.11 | MaxPool2d | 0 |
| head | Sequential | 903 |
| head.0 | AdaptiveAvgPool2d | 0 |
| head.1 | Flatten | 0 |
| head.2 | Dropout | 0 |
| head.3 | Linear | 903 |

## Training

Epochs run: 20
Best validation accuracy: 0.824 (epoch 16)

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy | Learning rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.498 | 2.031 | 0.4071 | 0.1738 | 0.0009932 |
| 2 | 1.097 | 1.257 | 0.5667 | 0.4738 | 0.0009729 |
| 3 | 0.9258 | 0.9688 | 0.6508 | 0.6357 | 0.0009397 |
| 4 | 0.8256 | 0.969 | 0.6786 | 0.6214 | 0.0008946 |
| 5 | 0.7445 | 0.8516 | 0.7222 | 0.6476 | 0.0008386 |
| 6 | 0.6605 | 0.8123 | 0.7571 | 0.6548 | 0.0007735 |
| 7 | 0.6184 | 0.7997 | 0.769 | 0.6857 | 0.0007008 |
| 8 | 0.6154 | 0.8035 | 0.7786 | 0.7048 | 0.0006227 |
| 9 | 0.5636 | 0.6653 | 0.8 | 0.7286 | 0.0005413 |
| 10 | 0.5104 | 0.6886 | 0.827 | 0.7071 | 0.0004587 |
| 11 | 0.4989 | 0.6729 | 0.8198 | 0.7238 | 0.0003773 |
| 12 | 0.4509 | 0.5969 | 0.8484 | 0.7381 | 0.0002992 |
| 13 | 0.4153 | 0.544 | 0.8778 | 0.8048 | 0.0002265 |
| 14 | 0.4092 | 0.5189 | 0.869 | 0.8 | 0.0001614 |
| 15 | 0.3857 | 0.5006 | 0.8857 | 0.819 | 0.0001054 |
| 16 | 0.3697 | 0.4876 | 0.8992 | 0.8238 | 6.026e-05 |
| 17 | 0.362 | 0.492 | 0.9024 | 0.8143 | 2.709e-05 |
| 18 | 0.3529 | 0.4874 | 0.896 | 0.8048 | 6.819e-06 |
| 19 | 0.3603 | 0.4925 | 0.9024 | 0.8048 | 0 |
| 20 | 0.3555 | 0.4861 | 0.8937 | 0.8119 | 0 |

## Test metrics

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| chromatic_aberration | 1.000 | 1.000 | 1.000 | 60 |
| clean | 0.741 | 0.667 | 0.702 | 60 |
| defocus | 0.905 | 0.950 | 0.927 | 60 |
| jpeg | 0.803 | 0.817 | 0.810 | 60 |
| motion_blur | 0.850 | 0.850 | 0.850 | 60 |
| sensor_noise | 0.903 | 0.933 | 0.918 | 60 |
| vignetting | 0.783 | 0.783 | 0.783 | 60 |

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
- Config hash: 06cb9622a1012df5
- Epochs run: 20
- Git commit: 42a1941 (dirty)
- Git branch: main
- Mean latency (ms): 0.700
- p50 latency (ms): 0.613
- p95 latency (ms): 1.173
- Throughput (images/s): 1428.199

## Reproduction

```bash
cnncls train -c <config.yaml>  # produced 20261103T123724Z_optics_degradation_classifier_06cb9622
cnncls evaluate --run /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T123724Z_optics_degradation_classifier_06cb9622 --checkpoint /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T123724Z_optics_degradation_classifier_06cb9622/checkpoints/best.pt
cnncls report --run /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T123724Z_optics_degradation_classifier_06cb9622 --format html
cnncls compare /Users/pengzhao/Documents/Projects/git_repos/CNN_Classifier/runs/20261103T123724Z_optics_degradation_classifier_06cb9622 <other-run>
```
