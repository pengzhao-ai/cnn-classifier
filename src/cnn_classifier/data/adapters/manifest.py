"""CSV manifest adapter."""

from __future__ import annotations

from cnn_classifier.config.schema import DataConfig
from cnn_classifier.core.errors import ConfigError
from cnn_classifier.core.types import DatasetBundle
from cnn_classifier.data.adapters._common import (
    RawSample,
    assign_split,
    build_bundle,
    resolve_class_names,
)
from cnn_classifier.data.adapters.registry import register_dataset
from cnn_classifier.data.manifest import Manifest, fingerprint_records


@register_dataset("manifest")
class ManifestAdapter:
    """Read images and labels from a CSV manifest."""

    @staticmethod
    def build(config: DataConfig) -> DatasetBundle:
        if config.manifest is None:
            raise ConfigError("manifest adapter requires data.manifest")
        image_root = config.image_root or config.manifest.parent
        columns = config.adapter_params.get(
            "columns",
            {
                "path": "path",
                "label": "label",
                "split": "split",
                "group": "group",
                "severity": "severity",
            },
        )
        manifest = Manifest.from_csv(config.manifest, image_root, columns)
        samples = [
            RawSample(
                source=record.path,
                label=record.label,
                split=record.split,
                group=record.group,
                severity=record.severity,
            )
            for record in manifest.records
        ]
        class_names = resolve_class_names([sample.label for sample in samples], config)
        explicit = config.model_copy(
            update={"split": config.split.model_copy(update={"mode": "explicit"})}
        )
        split, mode = assign_split(samples, explicit)
        fingerprint = fingerprint_records(manifest.records)
        return build_bundle(
            samples,
            explicit,
            class_names=class_names,
            split=split,
            mode=mode,
            fingerprint=fingerprint,
            group_column=config.split.group_column,
        )
