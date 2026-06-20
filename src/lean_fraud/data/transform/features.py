"""Causal feature engineering for the Sparkov transaction table.

Given a per-card, time-sorted DataFrame, add the engineered numeric feature columns and
return (df, ordered numeric column names). Every feature is CAUSAL — it uses only the current
row and that card's PAST rows, never future transactions — so there is no temporal leakage.

Precondition: `df` is sorted by (user, unix_time). The orchestrator guarantees it; the
groupby diff/rolling ops rely on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIME_COL = "unix_time"


def treat_num_features(df: pd.DataFrame, feats: dict) -> tuple[pd.DataFrame, list[str]]:
    """Add causal NUMERIC features and return (df, numeric_column_names).

    `feats` is the config `features` block (per-feature toggles). Column order is fixed and
    must match the numeric block of meta.json's feature_names. Categorical encoding lives in
    encode.py, not here.
    """
    num_cols: list[str] = ["amt"]
    df["amt"] = df["amt"].astype("float32")

    if feats.get("amount_log", True):
        df["amt_log"] = np.log1p(df["amt"].clip(lower=0)).astype("float32")
        num_cols.append("amt_log")

    if feats.get("time_deltas", True):
        df["dt"] = df.groupby("user")[TIME_COL].diff().fillna(0).astype("float32")
        num_cols.append("dt")

    if feats.get("rolling_aggs", True):
        amt = df.groupby("user")["amt"]
        # shift() drops the current tx -> mean of strictly PRIOR spend (causal)
        df["amt_roll_mean"] = (
            amt.transform(lambda s: s.shift().expanding().mean()).fillna(0).astype("float32")
        )
        df["amt_count"] = amt.cumcount().astype("float32")
        num_cols += ["amt_roll_mean", "amt_count"]

    if feats.get("geo_distance", True):  # cardholder <-> merchant distance: classic fraud signal
        df["geo_dist"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2
        ).astype("float32")
        num_cols.append("geo_dist")

    if feats.get("time_features", True):  # each tx's own hour-of-day / day-of-week (causal)
        ts = pd.to_datetime(df[TIME_COL], unit="s")
        df["hour"] = ts.dt.hour.astype("float32")
        df["dow"] = ts.dt.dayofweek.astype("float32")
        num_cols += ["hour", "dow"]

    return df, num_cols
