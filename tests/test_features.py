"""Unit tests for the causal feature engineering (transform/features.py).

Tiny synthetic frames (no dataset needed) that pin the anti-leakage invariants: rolling spend
excludes the current transaction, time deltas reset per card, and the feature columns keep a fixed
order.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lean_fraud.data.transform.features import treat_num_features

FEATS = {
    "amount_log": True,
    "time_deltas": True,
    "rolling_aggs": True,
    "geo_distance": True,
    "time_features": True,
}


def _frame() -> pd.DataFrame:
    # Two cards, already sorted by (user, unix_time) as the orchestrator guarantees.
    return pd.DataFrame(
        {
            "user": ["A", "A", "A", "B", "B"],
            "unix_time": [100, 160, 220, 50, 50 + 86400],
            "amt": [10.0, 20.0, 30.0, 5.0, 7.0],
            "lat": [0.0, 0.0, 0.0, 0.0, 0.0],
            "long": [0.0, 0.0, 0.0, 0.0, 0.0],
            "merch_lat": [3.0, 0.0, 0.0, 0.0, 0.0],
            "merch_long": [4.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_num_cols_have_fixed_order():
    _, num_cols = treat_num_features(_frame(), FEATS)
    expected = ["amt", "amt_log", "dt", "amt_roll_mean", "amt_count", "geo_dist", "hour", "dow"]
    assert num_cols == expected


def test_rolling_mean_excludes_current_tx():
    # amt_roll_mean[i] = mean of that card's PRIOR amounts (current excluded; first row = 0).
    df, _ = treat_num_features(_frame(), FEATS)
    assert df["amt_roll_mean"].tolist() == [0.0, 10.0, 15.0, 0.0, 5.0]
    assert df["amt_count"].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_time_delta_resets_per_card():
    # dt = seconds since the card's previous tx; 0 at each card's first row.
    df, _ = treat_num_features(_frame(), FEATS)
    assert df["dt"].tolist() == [0.0, 60.0, 60.0, 0.0, 86400.0]


def test_geo_distance_and_amount_log():
    df, _ = treat_num_features(_frame(), FEATS)
    assert df["geo_dist"].iloc[0] == pytest.approx(5.0)  # sqrt(3^2 + 4^2)
    assert df["amt_log"].iloc[0] == pytest.approx(2.397895, rel=1e-5)


def test_toggled_off_features_are_dropped():
    feats = {**FEATS, "geo_distance": False, "time_features": False}
    df, num_cols = treat_num_features(_frame(), feats)
    assert "geo_dist" not in num_cols and "hour" not in num_cols
    assert "geo_dist" not in df.columns
