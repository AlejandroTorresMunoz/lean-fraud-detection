"""Unit tests for the triple class-conditioned PCA (transform/pca.py).

Pin the output shape/labels and the anti-leakage invariant: the projection is fit on train rows
only, so corrupting val/test rows must not move the train-row projections.
"""

from __future__ import annotations

import numpy as np

from lean_fraud.data.transform.pca import triple_pca


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 300
    num_x = rng.normal(size=(n, 5)).astype(np.float32)
    y = (rng.random(n) < 0.3).astype(np.int8)
    is_train = np.zeros(n, dtype=bool)
    is_train[:200] = True
    return num_x, y, is_train


def test_shapes_and_names():
    num_x, y, is_train = _data()
    z, names = triple_pca(num_x, y, is_train, variance=0.9)
    assert z.shape[0] == num_x.shape[0]
    assert z.shape[1] == len(names)
    assert z.shape[1] <= 3 * num_x.shape[1]  # three subspaces, each <= n_numeric components
    assert names[0].startswith("pca_legit_")
    assert any(nm.startswith("pca_fraud_") for nm in names)
    assert any(nm.startswith("pca_all_") for nm in names)


def test_fit_uses_train_only():
    # Corrupting non-train rows must not change the train-row projections (fit is train-only).
    num_x, y, is_train = _data()
    z1, _ = triple_pca(num_x, y, is_train)
    corrupted = num_x.copy()
    corrupted[~is_train] += 100.0
    z2, _ = triple_pca(corrupted, y, is_train)
    np.testing.assert_allclose(z1[is_train], z2[is_train], rtol=1e-4, atol=1e-4)
