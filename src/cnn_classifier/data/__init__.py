"""Reusable data layer for dataset adapters, splits, and image contracts."""

from cnn_classifier.data.base import DatasetAdapter
from cnn_classifier.data.manifest import Manifest, fingerprint_records
from cnn_classifier.data.splits import (
    SplitIndices,
    assert_no_group_overlap,
    group_split_indices,
    random_split_indices,
    stratified_split_indices,
)

__all__ = [
    "DatasetAdapter",
    "Manifest",
    "SplitIndices",
    "assert_no_group_overlap",
    "fingerprint_records",
    "group_split_indices",
    "random_split_indices",
    "stratified_split_indices",
]
