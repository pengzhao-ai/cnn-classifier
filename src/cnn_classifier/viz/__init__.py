"""Visualization and reporting helpers."""

from cnn_classifier.viz.architecture import plot_model_card
from cnn_classifier.viz.compare import plot_model_comparison
from cnn_classifier.viz.data import plot_augmentation_gallery, plot_dataset_overview
from cnn_classifier.viz.evaluation import (
    plot_calibration,
    plot_confusion_matrix,
    plot_error_gallery,
    plot_per_class_metrics,
)
from cnn_classifier.viz.features import (
    compute_gradcam,
    plot_feature_maps,
    plot_filters,
    plot_gradcam,
)
from cnn_classifier.viz.report import build_markdown_report, build_report
from cnn_classifier.viz.style import PALETTE, save_figure, set_style
from cnn_classifier.viz.training import plot_training_history

__all__ = [
    "PALETTE",
    "build_markdown_report",
    "build_report",
    "compute_gradcam",
    "plot_augmentation_gallery",
    "plot_calibration",
    "plot_confusion_matrix",
    "plot_dataset_overview",
    "plot_error_gallery",
    "plot_feature_maps",
    "plot_filters",
    "plot_gradcam",
    "plot_model_card",
    "plot_model_comparison",
    "plot_per_class_metrics",
    "plot_training_history",
    "save_figure",
    "set_style",
]
