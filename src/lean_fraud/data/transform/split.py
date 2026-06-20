"""Strict time-based train/val/test split (no future leaks into train).

Global split BY TIME: the earliest (1 - test - val) fraction of transactions -> train, the
next `val` fraction -> val, the most-recent `test` fraction -> test. Each row is labelled by
its OWN timestamp, so the boundaries are honest time cutoffs — the model trains on the past
and scores the future, as in production.
"""

from __future__ import annotations

import numpy as np

TRAIN, VAL, TEST = 0, 1, 2


def time_split(t: np.ndarray, test_size: float = 0.2, val_size: float = 0.1) -> np.ndarray:
    """Per-transaction split label (int8: 0=train, 1=val, 2=test) from unix timestamps `t`.

    `t` may be in any order. Cutoffs are the time quantiles placing `test_size` of rows
    (most recent) in test and `val_size` just before that in val. Ties at a cutoff fall into
    the EARLIER split, which keeps train strictly in the past (t[train].max() < t[val].min()).
    """
    n = len(t)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_val - n_test
    t_sorted = np.sort(t, kind="stable")
    train_max_t = t_sorted[n_train - 1]
    val_max_t = t_sorted[n_train + n_val - 1]
    return np.where(t <= train_max_t, TRAIN, np.where(t <= val_max_t, VAL, TEST)).astype(np.int8)
