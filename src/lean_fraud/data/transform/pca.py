"""Triple class-conditioned PCA feature engineering (the thesis ablation).

Three PCA subspaces are fit on the TRAIN rows only — one on legit transactions, one on fraud, one
on all — and every row is projected onto all three; the projections are concatenated. A legit row
sits well in the legit subspace and poorly in the fraud one (and vice versa), so this injects
class-discriminative structure into the features before the model sees them. It replays, on
financial data, the triple-PCA preprocessing from Alejandro's FDI publication.

Fitting is strictly train-only (standardization stats and every PCA basis), so no validation/test
information leaks into the representation.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def triple_pca(
    num_x: np.ndarray, y: np.ndarray, is_train: np.ndarray, variance: float = 0.9
) -> tuple[np.ndarray, list[str]]:
    """Project the numeric block onto three train-fit class subspaces and concatenate.

    Args:
        num_x: (n, n_numeric) numeric feature block (raw, unscaled).
        y: (n,) binary labels (used only on train rows to split the subspaces).
        is_train: (n,) boolean mask of train rows — the only rows any fit may touch.
        variance: cumulative explained-variance kept per subspace (sklearn picks the components).

    Returns:
        (z, names): z is (n, k_legit + k_fraud + k_all) float32; names are the column labels.
    """
    y = np.asarray(y)
    is_train = np.asarray(is_train, dtype=bool)

    # Standardize using train statistics only (PCA is scale-sensitive).
    mean = num_x[is_train].mean(axis=0)
    std = num_x[is_train].std(axis=0) + 1e-8
    xs = ((num_x - mean) / std).astype(np.float64)

    subsets = {
        "legit": is_train & (y == 0),
        "fraud": is_train & (y == 1),
        "all": is_train,
    }
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for tag, mask in subsets.items():
        pca = PCA(n_components=variance, svd_solver="full").fit(xs[mask])
        z = pca.transform(xs)  # project EVERY row onto this train-fit subspace
        blocks.append(z)
        names.extend(f"pca_{tag}_{i}" for i in range(z.shape[1]))
    return np.hstack(blocks).astype(np.float32), names
