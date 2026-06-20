"""Turn the raw Sparkov transactions into a per-card, time-ordered feature table.

Sparkov ships one row per credit-card transaction with a card number (cc_num), a unix timestamp and
an is_fraud label, so the "user" is just the card. We merge fraudTrain.csv + fraudTest.csv (we make
our OWN split, ignoring the dataset's), order each card's transactions by unix_time, engineer a
compact set of causal features, integer-encode a few low-cardinality categoricals, do a STRICT
time-based train/val/test split (no future in train), and standardize the numeric block using train
statistics only.

Output (data/processed/sequences.npz), one row per transaction, sorted by (user, unix_time):
  X      float32 (n, n_features)   engineered + scaled features
  y      int8    (n,)              is_fraud label
  user   int64   (n,)             contiguous card id (for grouping)
  t      int64   (n,)             unix_time (ordering within a card)
  split  int8    (n,)             0=train, 1=val, 2=test (by the row's own time)
plus meta.json (feature names, scaler stats, category maps, split sizes / fraud rates).

We keep ONE table (not three) on purpose: a val/test sample's causal window may legitimately include
that card's earlier train rows — that is past context, not leakage. Sequence windows are built lazily
from this table with `make_windows`, avoiding a multi-GB (n, seq_len, n_features) materialization.

Usage: python -m lean_fraud.data.build_sequences --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lean_fraud.config import load_config

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


def make_windows(
    x: np.ndarray, user: np.ndarray, seq_len: int, indices: np.ndarray | None = None
) -> np.ndarray:
    """Build causal, per-user, zero-(left-)padded windows for the given target rows.

    `x` and `user` must be sorted by (user, t). Row i's window is the seq_len transactions ending at
    i (inclusive) within the same user. Returns (len(indices), seq_len, n_features). Pass `indices`
    (e.g. one split or one batch) to avoid materializing every window at once.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Sparkov sequence feature table.")
    parser.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(parser.parse_args().config)
    ds, feats = cfg["dataset"], cfg["features"]

    out_dir = Path(ds["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_raw(Path(ds["raw_dir"]))
    df["user"] = _user_key(df, feats.get("user_key", ["cc_num"]))
    df = df.sort_values(["user", TIME_COL]).reset_index(drop=True)

    # --- numeric features (causal where time-dependent) ---
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
        df["amt_roll_mean"] = (
            amt.transform(lambda s: s.shift().expanding().mean()).fillna(0).astype("float32")
        )
        df["amt_count"] = amt.cumcount().astype("float32")
        num_cols += ["amt_roll_mean", "amt_count"]
    if feats.get("geo_distance", True):  # cardholder <-> merchant distance: a classic fraud signal
        df["geo_dist"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2
        ).astype("float32")
        num_cols.append("geo_dist")
    if feats.get("time_features", True):  # hour-of-day / day-of-week: each tx's own time (causal)
        ts = pd.to_datetime(df[TIME_COL], unit="s")
        df["hour"] = ts.dt.hour.astype("float32")
        df["dow"] = ts.dt.dayofweek.astype("float32")
        num_cols += ["hour", "dow"]

    # --- strict time-based split by unix_time (no future in train) ---
    n = len(df)
    n_test = int(n * ds.get("test_size", 0.2))
    n_val = int(n * ds.get("val_size", 0.1))
    n_train = n - n_val - n_test
    t = df[TIME_COL].to_numpy()
    t_sorted = np.sort(t, kind="stable")
    train_max_t, val_max_t = t_sorted[n_train - 1], t_sorted[n_train + n_val - 1]
    split = np.where(t <= train_max_t, 0, np.where(t <= val_max_t, 1, 2)).astype(np.int8)
    is_train = split == 0

    # --- categoricals -> integer codes, fit on train only (0 = unknown/NA) ---
    cat_maps: dict[str, dict[str, int]] = {}
    code_cols: list[str] = []
    for col in [c for c in feats.get("categorical", []) if c in df.columns]:
        values = df[col].astype("string").fillna("NA")
        mapping = {v: i + 1 for i, v in enumerate(sorted(values[is_train].unique()))}
        df[f"{col}_code"] = values.map(mapping).fillna(0).astype("float32")
        cat_maps[col] = mapping
        code_cols.append(f"{col}_code")

    # --- assemble matrix; standardize the numeric block with train mean/std ---
    feature_cols = num_cols + code_cols
    x = df[feature_cols].to_numpy(dtype=np.float32)
    mean = x[is_train, : len(num_cols)].mean(axis=0)
    std = x[is_train, : len(num_cols)].std(axis=0)
    std[std == 0] = 1.0
    x[:, : len(num_cols)] = (x[:, : len(num_cols)] - mean) / std

    y = df["is_fraud"].to_numpy(dtype=np.int8)
    user = pd.factorize(df["user"])[0].astype(np.int64)  # contiguous (df is user-sorted)

    np.savez_compressed(out_dir / OUT_NAME, X=x, y=y, user=user, t=t.astype(np.int64), split=split)
    meta = {
        "dataset": ds.get("name", "sparkov"),
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "n_numeric": len(num_cols),
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

    print(f"[build_sequences] {n} rows, {len(feature_cols)} features, {meta['n_users']} users")
    for name in ("train", "val", "test"):
        s = meta["splits"][name]
        print(f"[build_sequences]   {name:5s}: {s['rows']:>7} rows  fraud={s['fraud_rate']:.4f}")
    print(f"[build_sequences] wrote {out_dir / OUT_NAME} + meta.json")


if __name__ == "__main__":
    main()
