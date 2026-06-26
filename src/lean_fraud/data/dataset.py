"""PyTorch Dataset over the processed feature table (data/processed/sequences.npz).

Each target row is expanded on the fly into the `seq_len` transactions ending at it (inclusive)
within the same card, left-padded with zeros — the same causal contract as `windows.make_windows`,
but with the per-user offsets precomputed once so __getitem__ is O(seq_len), not O(n).

A split's targets are the rows whose `split` label matches, yet their windows index into the FULL
table: a val/test window may legitimately reach back into that card's earlier train rows. That is
past context, not leakage (the label boundary is by time, per row).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lean_fraud.data.transform.split import TEST, TRAIN, VAL

_SPLIT_CODE = {"train": TRAIN, "val": VAL, "test": TEST}


class SequenceDataset(Dataset):
    """Causal, per-card sequence windows for one split, served as (window, label) tensors.

    Args:
        x: (n, n_features) processed feature table, sorted by (user, t).
        y: (n,) binary labels.
        user: (n,) contiguous card id (same sort order as `x`).
        split: (n,) split labels (0=train, 1=val, 2=test).
        which: which split this dataset serves ("train" | "val" | "test").
        seq_len: window length.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        user: np.ndarray,
        split: np.ndarray,
        which: str,
        seq_len: int,
    ) -> None:
        if which not in _SPLIT_CODE:
            raise ValueError(f"which must be one of {list(_SPLIT_CODE)}, got {which!r}")
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.seq_len = int(seq_len)
        self.n_features = self.x.shape[1]

        # First row of each card's block, broadcast to every row — so a window never crosses cards.
        n = len(user)
        change = np.ones(n, dtype=bool)
        change[1:] = user[1:] != user[:-1]
        starts = np.flatnonzero(change)
        self.user_start = starts[np.cumsum(change) - 1]

        # Target rows for this split (windows still index into the full table).
        self.targets = np.flatnonzero(split == _SPLIT_CODE[which]).astype(np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def _window(self, i: int) -> np.ndarray:
        """The seq_len rows ending at row i (inclusive), clipped to i's card, left zero-padded."""
        lo = max(int(self.user_start[i]), i - self.seq_len + 1)
        window = self.x[lo : i + 1]
        out = np.zeros((self.seq_len, self.n_features), dtype=np.float32)
        out[self.seq_len - window.shape[0] :] = window
        return out

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(self.targets[k])
        return torch.from_numpy(self._window(i)), torch.tensor(self.y[i], dtype=torch.float32)

    @property
    def pos_weight(self) -> float:
        """neg/pos ratio over this split's targets — for BCE class weighting."""
        labels = self.y[self.targets]
        pos = float(labels.sum())
        return (len(labels) - pos) / pos if pos > 0 else 1.0


def load_processed(processed_dir: str | Path) -> dict[str, np.ndarray]:
    """Load the npz produced by build_sequences into a dict of arrays."""
    with np.load(Path(processed_dir) / "sequences.npz") as npz:
        return {k: npz[k] for k in ("X", "y", "user", "t", "split")}
