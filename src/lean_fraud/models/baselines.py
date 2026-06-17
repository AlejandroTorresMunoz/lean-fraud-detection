"""Tabular baselines: logistic regression and XGBoost.

These are the real industry standard for fraud — the TCN must match or beat them.
Operate on flattened/aggregated features (no temporal order).
TODO: wire feature matrix from lean_fraud.data.build_sequences.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def train_logreg(x: np.ndarray, y: np.ndarray, **kwargs: Any):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, class_weight="balanced", **kwargs)
    model.fit(x, y)
    return model


def train_xgboost(x: np.ndarray, y: np.ndarray, **kwargs: Any):
    from xgboost import XGBClassifier

    # scale_pos_weight handles the heavy class imbalance typical in fraud data.
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        **kwargs,
    )
    model.fit(x, y)
    return model
