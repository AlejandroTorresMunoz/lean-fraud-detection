"""Lazy, causal, per-user sequence windows over the processed feature table.

The processed table is kept 2-D (one row per transaction). At train/score time each target
row is expanded into the seq_len transactions ending at it (inclusive) within the same user,
left-padded with zeros — avoiding a multi-GB (n, seq_len, n_features) materialization. Used by
the training SequenceDataset, not by build_sequences.
"""

from __future__ import annotations

import numpy as np


def make_windows(
    x: np.ndarray, user: np.ndarray, seq_len: int, indices: np.ndarray | None = None
) -> np.ndarray:
    """Build causal, per-user, left-zero-padded windows for the given target rows.

    `x` and `user` must be sorted by (user, t). Row i's window is the seq_len transactions
    ending at i (inclusive) within the same user. Returns (len(indices), seq_len, n_features).
    Pass `indices` (e.g. one split or one batch) to avoid materializing every window at once.
    """
    n, n_features = x.shape
    idx = np.arange(n) if indices is None else np.asarray(indices)

    change = np.ones(n, dtype=bool)
    change[1:] = user[1:] != user[:-1]
    starts = np.flatnonzero(change)
    user_start = starts[np.cumsum(change) - 1]  # first row of each row's user block

    out = np.zeros((len(idx), seq_len, n_features), dtype=np.float32)
    for k, i in enumerate(idx):
        lo = max(int(user_start[i]), int(i) - seq_len + 1)
        window = x[lo : i + 1]
        out[k, seq_len - window.shape[0] :] = window
    return out
