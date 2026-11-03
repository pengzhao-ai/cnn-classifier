"""Dataset adapter registry and public builders."""

from cnn_classifier.data.adapters.image_folder import ImageFolderAdapter
from cnn_classifier.data.adapters.manifest import ManifestAdapter
from cnn_classifier.data.adapters.numpy import NumpyAdapter
from cnn_classifier.data.adapters.registry import DATASETS, build_dataset, register_dataset
from cnn_classifier.data.adapters.synthetic import SyntheticAdapter
from cnn_classifier.data.adapters.torchvision import TorchvisionAdapter

__all__ = [
    "DATASETS",
    "ImageFolderAdapter",
    "ManifestAdapter",
    "NumpyAdapter",
    "SyntheticAdapter",
    "TorchvisionAdapter",
    "build_dataset",
    "register_dataset",
]
