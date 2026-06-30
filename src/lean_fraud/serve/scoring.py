"""Shared fraud-scoring core used by BOTH the FastAPI service and the Kinesis consumer.

The single scoring path, so the sync API and the stream never drift apart. It rebuilds the exact
training-time features from a card's raw transaction history — reusing `data.transform` so there is
no train/serve skew — standardizes with the train scaler from meta.json, runs the trained model, and
thresholds with the val-tuned decision threshold saved by `evaluate`.

The rolling/expanding features (`amt_roll_mean`, `amt_count`, `dt`) are causal over a card's PAST,
so pass the card's full available history for an exact reproduction; a truncated history degrades
those features gracefully rather than leaking the future.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lean_fraud.data.transform.encode import apply_categoricals, apply_scaler
from lean_fraud.data.transform.features import treat_num_features
from lean_fraud.models import load_checkpoint


@dataclass
class Scorer:
    """Everything needed to turn a card's raw tx history into a fraud probability + decision."""

    model: torch.nn.Module
    feats_cfg: dict[str, Any]  # the `features` config block — same toggles used at training time
    seq_len: int
    n_numeric: int
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    categorical_maps: dict[str, dict[str, int]]
    threshold: float

    @property
    def n_features(self) -> int:
        return self.n_numeric + len(self.categorical_maps)


def _run_name(cfg: dict) -> str:
    features_tag = cfg.get("features", {}).get("engineering", "raw")
    return cfg["mlflow"].get("run_name") or f"{cfg['model']['type']}-{features_tag}"


def load_scorer(cfg: dict) -> Scorer:
    """Assemble a Scorer from the config: model checkpoint + data meta.json + val-tuned threshold.

    Loads the durable local artifacts (not the ephemeral LocalStack S3) so serving is reproducible.
    Only raw-feature models are supported; triple_pca would also need the saved PCA transform.
    """
    if cfg.get("features", {}).get("engineering", "raw") != "raw":
        raise ValueError(
            "load_scorer supports raw-feature models only "
            "(triple_pca needs the train-fit PCA transform, which serving does not persist)."
        )

    artifacts_dir = Path(cfg.get("artifacts", {}).get("dir", "artifacts")) / _run_name(cfg)
    ckpt_path = artifacts_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Train first: python -m lean_fraud.train"
        )
    model, ckpt_meta = load_checkpoint(ckpt_path)

    meta_path = Path(cfg["dataset"]["processed_dir"]) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No {meta_path}. Build the data first: python -m lean_fraud.data.build_sequences"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # The decision threshold is F1-optimal on validation, persisted by evaluate.py.
    metrics_path = artifacts_dir / "test_metrics.json"
    threshold = 0.5
    if metrics_path.exists():
        threshold = float(
            json.loads(metrics_path.read_text(encoding="utf-8")).get("threshold", 0.5)
        )

    return Scorer(
        model=model,
        feats_cfg=cfg["features"],
        seq_len=int(ckpt_meta.get("seq_len", meta.get("sequence_length", 32))),
        n_numeric=int(meta["n_numeric"]),
        scaler_mean=np.asarray(meta["scaler"]["mean"], dtype=np.float32),
        scaler_std=np.asarray(meta["scaler"]["std"], dtype=np.float32),
        categorical_maps=meta["categorical_maps"],
        threshold=threshold,
    )


def build_feature_window(scorer: Scorer, raw_tx: list[dict]) -> np.ndarray:
    """Engineer the (seq_len, n_features) model input from a card's raw tx history (oldest first).

    Reuses the training transforms (`treat_num_features` + the saved category maps + the train
    scaler) so the features match exactly, then takes the last `seq_len` rows left zero-padded —
    the same causal window contract as `SequenceDataset`. Pass the card's full history for exact
    rolling features.
    """
    if not raw_tx:
        return np.zeros((scorer.seq_len, scorer.n_features), dtype=np.float32)

    df = pd.DataFrame(raw_tx)
    # A request is one card's history: a constant user key makes treat_num_features' groupby a single
    # group. (Deriving it from cc_num would risk a null/NA key, which pandas drops -> NaN features.)
    df["user"] = "card"
    df = df.sort_values("unix_time").reset_index(drop=True)

    df, num_cols = treat_num_features(df, scorer.feats_cfg)
    num_block = df[num_cols].to_numpy(dtype=np.float32)
    _, code_block = apply_categoricals(df, scorer.categorical_maps)

    x = np.hstack([num_block, code_block]).astype(np.float32)
    x = apply_scaler(x, scorer.n_numeric, scorer.scaler_mean, scorer.scaler_std)

    window = x[-scorer.seq_len :]
    out = np.zeros((scorer.seq_len, x.shape[1]), dtype=np.float32)
    out[scorer.seq_len - window.shape[0] :] = window
    return out


def score(scorer: Scorer, window: np.ndarray) -> tuple[float, bool, float]:
    """Run inference on a prepared window. Returns (fraud_probability, is_fraud, latency_ms).

    latency_ms covers inference only (matching benchmark.measure_latency semantics), so the API can
    echo it as the serving SLA figure.
    """
    t0 = time.perf_counter()
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(window)).unsqueeze(0)  # (1, seq_len, n_features)
        prob = float(torch.sigmoid(scorer.model(x)).item())
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return prob, prob >= scorer.threshold, latency_ms


def score_history(scorer: Scorer, raw_tx: list[dict]) -> tuple[float, bool, float]:
    """Convenience: build the window from raw history and score it in one call."""
    return score(scorer, build_feature_window(scorer, raw_tx))
