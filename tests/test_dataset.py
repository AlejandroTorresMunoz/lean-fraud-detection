"""Unit tests for SequenceDataset (data/dataset.py).

Pin the same causal contract as make_windows but through the Dataset surface: a window ends at its
target, never crosses cards, is left zero-padded, and a val target may reach back into that card's
earlier (train) rows — past context, not leakage.
"""

from __future__ import annotations

import numpy as np

from lean_fraud.data.dataset import SequenceDataset


def _data() -> dict[str, np.ndarray]:
    # One card (user 0) with 4 rows, one feature = the row's value; split: first 2 train, last 2 val.
    x = np.arange(1, 5, dtype=np.float32).reshape(4, 1)
    y = np.array([0, 0, 0, 1], dtype=np.int8)
    user = np.array([0, 0, 0, 0])
    split = np.array([0, 0, 1, 1], dtype=np.int8)  # 0=train, 1=val
    return {"X": x, "y": y, "user": user, "split": split}


def test_len_matches_split():
    d = _data()
    train = SequenceDataset(d["X"], d["y"], d["user"], d["split"], "train", seq_len=3)
    val = SequenceDataset(d["X"], d["y"], d["user"], d["split"], "val", seq_len=3)
    assert len(train) == 2 and len(val) == 2


def test_window_ends_at_target_and_left_padded():
    d = _data()
    train = SequenceDataset(d["X"], d["y"], d["user"], d["split"], "train", seq_len=3)
    win, label = train[0]  # first train row: itself, left-padded
    assert win.shape == (3, 1)
    assert win.ravel().tolist() == [0.0, 0.0, 1.0]
    assert label.item() == 0.0


def test_val_window_reaches_into_earlier_rows():
    d = _data()
    val = SequenceDataset(d["X"], d["y"], d["user"], d["split"], "val", seq_len=3)
    # val target 0 is global row 2 (value 3); its window pulls the card's earlier rows 1,2,3.
    win, _ = val[0]
    assert win.ravel().tolist() == [1.0, 2.0, 3.0]
    # val target 1 is global row 3 (value 4, the fraud); window = rows 2,3,4.
    win, label = val[1]
    assert win.ravel().tolist() == [2.0, 3.0, 4.0]
    assert label.item() == 1.0


def test_window_does_not_cross_cards():
    # Two cards; card 1's first row must not pull card 0's rows.
    x = np.arange(1, 6, dtype=np.float32).reshape(5, 1)
    y = np.zeros(5, dtype=np.int8)
    user = np.array([0, 0, 0, 1, 1])
    split = np.zeros(5, dtype=np.int8)
    ds = SequenceDataset(x, y, user, split, "train", seq_len=3)
    win, _ = ds[3]  # card 1's first row (value 4)
    assert win.ravel().tolist() == [0.0, 0.0, 4.0]


def test_pos_weight():
    d = _data()
    val = SequenceDataset(d["X"], d["y"], d["user"], d["split"], "val", seq_len=3)
    assert val.pos_weight == 1.0  # 1 negative / 1 positive among val targets
