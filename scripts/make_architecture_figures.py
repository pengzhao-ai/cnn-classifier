"""Generate concise architecture pipeline figures for the demos."""

from __future__ import annotations

from pathlib import Path

from cnn_classifier.config import load_config
from cnn_classifier.diagnostics.architecture import architecture_stages
from cnn_classifier.models import build_model
from cnn_classifier.viz import plot_cnn_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs" / "images"

SPECS = (
    (
        "cifar10",
        "Residual CNN — CIFAR-10 scratch",
        "configs/cifar10/scratch_baseline.yaml",
    ),
    (
        "transfer",
        "ResNet-18 linear probe — 32x32 transfer",
        "configs/cifar10/resnet18_linear_probe_cifar.yaml",
    ),
    (
        "ablations",
        "Residual CNN — ablation budget",
        "configs/ablations/batchnorm.yaml",
    ),
    (
        "optics",
        "Simple CNN — optical degradation",
        "configs/optics_degradation/degradation_classifier.yaml",
    ),
    (
        "byo",
        "Simple CNN — bring your own data",
        "demos/bring_your_own_data/config.yaml",
    ),
)


def _build(config_path: Path):
    config = load_config(config_path)
    model_config = config.model
    if model_config.name == "torchvision":
        # Shapes and freeze flags are identical; avoid downloading weights.
        transfer = model_config.transfer.model_copy(update={"weights": None})
        model_config = model_config.model_copy(update={"transfer": transfer})
    num_classes = config.data.num_classes or 10
    in_channels = 3 if config.data.channels == "rgb" else 1
    model = build_model(
        model_config,
        num_classes=num_classes,
        in_channels=in_channels,
        image_size=config.data.image_size,
    )
    return config, model, in_channels, num_classes


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, title, relative in SPECS:
        config, model, in_channels, num_classes = _build(REPO_ROOT / relative)
        input_shape = (in_channels, *config.data.image_size)
        stages = architecture_stages(model, input_shape)
        path = plot_cnn_pipeline(
            stages,
            input_shape=input_shape,
            num_classes=num_classes,
            path=OUTPUT_DIR / f"architecture_{name}.png",
            title=title,
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
