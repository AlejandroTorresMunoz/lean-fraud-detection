"""Unit tests for lazy causal windowing (data/windows.py).

Pin causality (a window ends at its target row), per-user isolation (a card's first transaction
never pulls in the previous card's rows), and left zero-padding for short histories.
"""

from __future__ import annotations

import numpy as np

from lean_fraud.data.windows import make_windows


def _xu() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(1, 6, dtype=np.float32).reshape(5, 1)  # values 1..5, one feature
    user = np.array([0, 0, 0, 1, 1])
    return x, user


def test_shape_and_window_ends_at_target():
    x, user = _xu()
    out = make_windows(x, user, seq_len=3)
    assert out.shape == (5, 3, 1)
    for i in range(5):  # the last row of each window is the target row itself
        assert out[i, -1, 0] == x[i, 0]


def test_left_padding_for_short_history():
    x, user = _xu()
    out = make_windows(x, user, seq_len=3)
    assert out[0].ravel().tolist() == [0.0, 0.0, 1.0]  # first tx: itself, left-padded
    assert out[2].ravel().tolist() == [1.0, 2.0, 3.0]  # full window


def test_window_does_not_cross_users():
    x, user = _xu()
    out = make_windows(x, user, seq_len=3)
    # user 1's first tx (row 3, value 4) must NOT pull in user 0's rows (1, 2, 3)
    assert out[3].ravel().tolist() == [0.0, 0.0, 4.0]
    assert out[4].ravel().tolist() == [0.0, 4.0, 5.0]


def test_indices_subset_matches_full_pass():
    x, user = _xu()
    full = make_windows(x, user, seq_len=3)
    sub = make_windows(x, user, seq_len=3, indices=[4])
    assert sub.shape == (1, 3, 1)
    assert np.array_equal(sub[0], full[4])
