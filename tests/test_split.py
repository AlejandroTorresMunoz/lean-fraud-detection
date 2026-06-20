"""Unit tests for the strict time-based split (transform/split.py).

The headline invariant: no future leaks into train — t[train].max() < t[val].min() < t[test].min(),
regardless of input order and even when timestamps tie at a cutoff.
"""

from __future__ import annotations

import numpy as np

from lean_fraud.data.transform.split import TEST, TRAIN, VAL, time_split


def test_split_sizes_and_dtype():
    split = time_split(np.arange(100), test_size=0.2, val_size=0.1)
    assert split.dtype == np.int8
    assert (split == TRAIN).sum() == 70
    assert (split == VAL).sum() == 10
    assert (split == TEST).sum() == 20


def test_no_future_leaks_into_train_when_shuffled():
    t = np.random.default_rng(0).permutation(1000)  # timestamps in arbitrary order
    split = time_split(t, test_size=0.2, val_size=0.1)
    assert t[split == TRAIN].max() < t[split == VAL].min()
    assert t[split == VAL].max() < t[split == TEST].min()


def test_ties_at_cutoff_keep_train_in_the_past():
    # Five identical timestamps at the start; a tie must not straddle a split boundary.
    t = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])
    split = time_split(t, test_size=0.2, val_size=0.1)
    assert t[split == TRAIN].max() < t[split == VAL].min() < t[split == TEST].min()
