import pytest

from cnn_classifier.core.errors import DataError
from cnn_classifier.data.splits import (
    SplitIndices,
    assert_no_group_overlap,
    group_split_indices,
    random_split_indices,
    stratified_split_indices,
)


def test_random_split_is_deterministic_and_disjoint():
    labels = [0] * 10 + [1] * 10
    a = random_split_indices(labels, 0.2, seed=42)
    b = random_split_indices(labels, 0.2, seed=42)
    assert a.train_indices == b.train_indices
    assert set(a.train_indices).isdisjoint(a.val_indices)
    assert len(a.val_indices) == 4


def test_stratified_split_preserves_class_proportions():
    labels = [0] * 20 + [1] * 20
    split = stratified_split_indices(labels, 0.25, seed=7)
    val_labels = [labels[i] for i in split.val_indices]
    assert val_labels.count(0) == 5
    assert val_labels.count(1) == 5


def test_group_split_has_no_overlap():
    groups = [f"g{i // 5}" for i in range(40)]
    split = group_split_indices(groups, 0.25, seed=7)
    train_groups = {groups[i] for i in split.train_indices}
    val_groups = {groups[i] for i in split.val_indices}
    assert train_groups.isdisjoint(val_groups)


def test_assert_no_group_overlap_raises():
    groups = ["a", "a", "b", "b"]
    split = SplitIndices(train_indices=(0, 2), val_indices=(1, 3))
    with pytest.raises(DataError, match="group"):
        assert_no_group_overlap(split, groups)
