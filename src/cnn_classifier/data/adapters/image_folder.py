"""ImageFolder adapter with explicit train/val/test directories."""

from __future__ import annotations

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.core.errors import ConfigError, DataError
from cnn_classifier.core.types import DatasetBundle
from cnn_classifier.data.adapters._common import (
    RawSample,
    assign_split,
    build_bundle,
    resolve_class_names,
    stable_sample_fingerprint,
)
from cnn_classifier.data.adapters.registry import register_dataset

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@register_dataset("image_folder")
class ImageFolderAdapter:
    """Read class directories from train/val/test split folders."""

    @staticmethod
    def build(config: DataConfig) -> DatasetBundle:
        if config.root is None:
            raise ConfigError("image_folder adapter requires data.root")
        split_dirs = config.adapter_params.get(
            "split_dirs", {"train": "train", "val": "val", "test": "test"}
        )
        samples: list[RawSample] = []
        observed_classes: set[str] = set()
        for split_name in ("train", "val", "test"):
            directory = config.root / split_dirs[split_name]
            if not directory.is_dir():
                if split_name == "test":
                    continue
                raise DataError(f"image_folder split directory does not exist: {directory}")
            for class_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
                observed_classes.add(class_dir.name)
                for image_path in sorted(class_dir.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in _IMAGE_SUFFIXES:
                        samples.append(
                            RawSample(
                                source=image_path,
                                label=class_dir.name,
                                split=split_name,
                            )
                        )
        if not samples:
            raise DataError(f"image_folder found no images under {config.root}")
        class_names = resolve_class_names(sorted(observed_classes), config)
        explicit = config.model_copy(
            update={"split": config.split.model_copy(update={"mode": "explicit"})}
        )
        split, mode = assign_split(samples, explicit)
        fingerprint = stable_sample_fingerprint(samples, config.split.seed)
        return build_bundle(
            samples,
            explicit,
            class_names=class_names,
            split=split,
            mode=mode,
            fingerprint=fingerprint,
        )
