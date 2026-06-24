"""Orchestrate the Sparkov ETL: raw transactions -> a per-card, time-ordered feature table.

This is the thin orchestrator. It only does Extract (load + per-card ordering) and Load (write
the processed table + meta), delegating each Transform stage to a focused, unit-testable module:

  transform.features.treat_num_features  -> causal numeric features
  transform.split.time_split             -> strict time-based train/val/test split
  transform.encode.encode_categoricals   -> categorical -> int codes (fit on train)
  transform.encode.fit_scaler/apply_scaler -> standardize the numeric block (fit on train)

Output (data/processed/sequences.npz), one row per transaction, sorted by (user, unix_time):
  X      float32 (n, n_features)   engineered + scaled features
  y      int8    (n,)              is_fraud label
  user   int64   (n,)             contiguous card id (for grouping)
  t      int64   (n,)             unix_time (ordering within a card)
  split  int8    (n,)             0=train, 1=val, 2=test (by the row's own time)
plus meta.json (feature names, scaler stats, category maps, split sizes / fraud rates).

We keep ONE table (not three) on purpose: a val/test sample's causal window may legitimately
include that card's earlier train rows — that is past context, not leakage. Sequence windows are
built lazily from this table with `windows.make_windows`, avoiding a multi-GB materialization.

Usage: python -m lean_fraud.data.build_sequences --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lean_fraud.config import load_config
from lean_fraud.data.transform.encode import apply_scaler, encode_categoricals, fit_scaler
from lean_fraud.data.transform.features import treat_num_features
from lean_fraud.data.transform.pca import triple_pca
from lean_fraud.data.transform.split import time_split

RAW_FILES = ["fraudTrain.csv", "fraudTest.csv"]
TIME_COL = "unix_time"  # epoch seconds; orders transactions within a card
OUT_NAME = "sequences.npz"

# Only the columns we actually use (the raw CSVs carry PII-ish fields we deliberately ignore).
USE_COLS = [
    "cc_num",
    "unix_time",
    "amt",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "category",
    "gender",
    "state",
    "is_fraud",
]


def _load_raw(raw_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(raw_dir / fname, usecols=USE_COLS)
        for fname in RAW_FILES
        if (raw_dir / fname).exists()
    ]
    if not frames:
        raise SystemExit(
            f"No Sparkov CSVs in {raw_dir}. Run `python -m lean_fraud.data.download` first."
        )
    return pd.concat(frames, ignore_index=True)


def _user_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    return df[present].astype("string").fillna("NA").agg("|".join, axis=1)


def build(cfg: dict) -> Path:
    """Build the processed feature table from a config dict; return the output npz path.

    `features.engineering` selects the numeric representation: `raw` (the causal features as-is) or
    `triple_pca` (the class-conditioned triple-PCA ablation). Both then share the categorical
    encoding, scaling and windowing — so the two datasets differ only in their numeric block.
    """
    ds, feats = cfg["dataset"], cfg["features"]
    engineering = feats.get("engineering", "raw")

    out_dir = Path(ds["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- extract: load + per-card time ordering (every downstream stage relies on this order) ---
    df = _load_raw(Path(ds["raw_dir"]))
    df["user"] = _user_key(df, feats.get("user_key", ["cc_num"]))
    df = df.sort_values(["user", TIME_COL]).reset_index(drop=True)

    # --- transform: causal features -> time split -> (optional triple-PCA) -> codes -> scale ---
    df, num_cols = treat_num_features(df, feats)

    t = df[TIME_COL].to_numpy()
    split = time_split(t, ds.get("test_size", 0.2), ds.get("val_size", 0.1))
    is_train = split == 0
    y = df["is_fraud"].to_numpy(dtype=np.int8)

    if engineering == "triple_pca":
        num_block, num_cols = triple_pca(df[num_cols].to_numpy(dtype=np.float32), y, is_train)
    elif engineering == "raw":
        num_block = df[num_cols].to_numpy(dtype=np.float32)
    else:
        raise SystemExit(f"Unknown features.engineering={engineering!r} (expected raw|triple_pca).")

    code_cols, cat_maps = encode_categoricals(df, feats.get("categorical", []), is_train)
    code_block = (
        df[code_cols].to_numpy(dtype=np.float32)
        if code_cols
        else np.empty((len(df), 0), dtype=np.float32)
    )

    n_numeric = num_block.shape[1]
    x = np.hstack([num_block, code_block]).astype(np.float32)
    mean, std = fit_scaler(
        x, n_numeric, is_train
    )  # standardize the numeric/PCA block (fit on train)
    x = apply_scaler(x, n_numeric, mean, std)

    user = pd.factorize(df["user"])[0].astype(np.int64)  # contiguous (df is user-sorted)
    feature_cols = num_cols + code_cols

    # --- load: write the processed table + meta ---
    np.savez_compressed(out_dir / OUT_NAME, X=x, y=y, user=user, t=t.astype(np.int64), split=split)
    meta = {
        "dataset": ds.get("name", "sparkov"),
        "engineering": engineering,
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "n_numeric": n_numeric,
        "sequence_length": ds.get("sequence_length", 32),
        "scaler": {"mean": mean.tolist(), "std": std.tolist()},
        "categorical_maps": cat_maps,
        "n_users": int(user.max()) + 1,
        "splits": {
            name: {
                "rows": int((split == code).sum()),
                "fraud_rate": float(y[split == code].mean()) if (split == code).any() else 0.0,
            }
            for name, code in (("train", 0), ("val", 1), ("test", 2))
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    n = len(df)
    print(
        f"[build_sequences] engineering={engineering}  {n} rows, "
        f"{len(feature_cols)} features ({n_numeric} numeric), {meta['n_users']} users"
    )
    for name in ("train", "val", "test"):
        s = meta["splits"][name]
        print(f"[build_sequences]   {name:5s}: {s['rows']:>7} rows  fraud={s['fraud_rate']:.4f}")
    print(f"[build_sequences] wrote {out_dir / OUT_NAME} + meta.json")
    return out_dir / OUT_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Sparkov sequence feature table.")
    parser.add_argument("--config", default="configs/base.yaml")
    build(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
