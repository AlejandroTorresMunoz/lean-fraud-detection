"""Categorical encoding + numeric scaling for the feature matrix — fit on TRAIN only.

The leakage-critical stage: category code maps and the standardization mean/std are learned
from the train split exclusively, then applied to val/test. Categorical codes are integers
(0 reserved for unknown/NA, i.e. categories unseen in train) and are NOT scaled. Split into
fit/apply so the train-only fit is explicit and unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def encode_categoricals(
    df: pd.DataFrame, cat_cols: list[str], is_train: np.ndarray
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Add `{col}_code` integer columns (codes fit on TRAIN; 0 = unknown/NA).

    Returns (code_column_names, maps). Only columns present in `df` are used; the mapping is
    built from values seen in the train split, so categories that appear only in val/test map
    to 0 (unknown) rather than leaking train-unseen levels.
    """
    code_cols: list[str] = []
    cat_maps: dict[str, dict[str, int]] = {}
    for col in [c for c in cat_cols if c in df.columns]:
        values = df[col].astype("string").fillna("NA")
        mapping = {v: i + 1 for i, v in enumerate(sorted(values[is_train].unique()))}
        df[f"{col}_code"] = values.map(mapping).fillna(0).astype("float32")
        cat_maps[col] = mapping
        code_cols.append(f"{col}_code")
    return code_cols, cat_maps


def apply_categoricals(
    df: pd.DataFrame, cat_maps: dict[str, dict[str, int]]
) -> tuple[list[str], np.ndarray]:
    """Apply already-fitted category maps (e.g. from meta.json) — the serving-time counterpart.

    Mirrors `encode_categoricals` but never fits: values unseen in the saved map (or NA) become 0,
    exactly as train-unseen levels did at fit time. Iterates `cat_maps` in insertion order, which
    matches the fit-time column order, so the returned code block lines up with meta's feature_names.
    Returns (code_column_names, code_block) with the codes also written onto `df`.
    """
    code_cols: list[str] = []
    blocks: list[np.ndarray] = []
    for col, mapping in cat_maps.items():
        values = df[col].astype("string").fillna("NA")
        codes = values.map(mapping).fillna(0).astype("float32")
        df[f"{col}_code"] = codes
        code_cols.append(f"{col}_code")
        blocks.append(codes.to_numpy(dtype=np.float32).reshape(-1, 1))
    block = np.hstack(blocks) if blocks else np.empty((len(df), 0), dtype=np.float32)
    return code_cols, block


def fit_scaler(
    x: np.ndarray, n_numeric: int, is_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of the first `n_numeric` columns over TRAIN rows only (std==0 -> 1)."""
    mean = x[is_train, :n_numeric].mean(axis=0)
    std = x[is_train, :n_numeric].std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(x: np.ndarray, n_numeric: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize the numeric block of `x` in place and return it (categorical codes untouched)."""
    x[:, :n_numeric] = (x[:, :n_numeric] - mean) / std
    return x
