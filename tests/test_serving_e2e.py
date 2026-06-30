"""Fast end-to-end serving smoke for CI: synthetic mini-dataset -> build -> tiny train -> score.

Ingests a small synthetic Sparkov-shaped CSV and runs the REAL pipeline on a tiny model offline
(MLflow disabled), then loads the produced artifacts through the serving path and scores a card
history. This catches integration breaks a pure unit test misses: feature-name/scaler-shape drift,
the meta.json <-> checkpoint contract, and the threshold hand-off — end to end, no Kaggle, no network.

train/evaluate run via the real `python -m` entrypoints in subprocesses (the same contract the
Airflow DAG uses). That also isolates each torch run in its own process — which keeps the suite
stable on Windows, where many tiny convs sharing one interpreter can hit a fatal stack overflow.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from lean_fraud.config import load_config
from lean_fraud.data.build_sequences import build
from lean_fraud.serve.scoring import build_feature_window, load_scorer, score_history

CATEGORIES = ["a", "b", "c"]
STATES = ["CA", "NY", "TX"]


def _synthetic_raw(n_rows: int = 600, n_cards: int = 8, seed: int = 0) -> pd.DataFrame:
    """A small Sparkov-shaped raw table: the USE_COLS the ETL consumes, plus is_fraud."""
    rng = np.random.default_rng(seed)
    cc = rng.integers(0, n_cards, size=n_rows)
    return pd.DataFrame(
        {
            "cc_num": 4_000_000_000_000_000 + cc,  # big ints like real card numbers
            "unix_time": np.sort(rng.integers(1_500_000_000, 1_600_000_000, size=n_rows)),
            "amt": rng.gamma(2.0, 30.0, size=n_rows).round(2),
            "lat": rng.uniform(25, 49, size=n_rows),
            "long": rng.uniform(-124, -67, size=n_rows),
            "merch_lat": rng.uniform(25, 49, size=n_rows),
            "merch_long": rng.uniform(-124, -67, size=n_rows),
            "category": rng.choice(CATEGORIES, size=n_rows),
            "gender": rng.choice(["M", "F"], size=n_rows),
            "state": rng.choice(STATES, size=n_rows),
            "is_fraud": (rng.random(n_rows) < 0.12).astype(
                int
            ),  # scattered -> present in all splits
        }
    )


def _tiny_config(tmp_path) -> dict:
    """Base config retargeted at temp dirs and shrunk so the whole run finishes in seconds."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _synthetic_raw().to_csv(raw_dir / "fraudTrain.csv", index=False)

    cfg = load_config("configs/base.yaml")
    cfg["dataset"]["raw_dir"] = str(raw_dir)
    cfg["dataset"]["processed_dir"] = str(tmp_path / "processed")
    cfg["dataset"]["sequence_length"] = 8
    cfg["artifacts"]["dir"] = str(tmp_path / "artifacts")
    cfg["model"]["tcn"]["channels"] = [8, 8]
    cfg["training"].update(epochs=2, max_train_batches=2, batch_size=64, device="cpu")
    cfg["mlflow"]["enabled"] = False  # offline: no server, no network
    return cfg


def _run(module: str, config_path) -> None:
    subprocess.run(
        [sys.executable, "-m", module, "--config", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_pipeline_to_serving_end_to_end(tmp_path):
    cfg = _tiny_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    build(cfg)  # pure pandas/numpy, no torch -> safe in-process
    _run("lean_fraud.train", config_path)  # subprocess: trains + saves best.pt
    _run("lean_fraud.evaluate", config_path)  # subprocess: writes test_metrics.json + threshold

    scorer = load_scorer(cfg)
    assert scorer.n_features == 11  # 8 numeric + category/gender/state codes
    assert scorer.seq_len == 8
    assert 0.0 <= scorer.threshold <= 1.0  # loaded from test_metrics.json, not a hardcoded 0.5

    history = [
        {
            "unix_time": 1_500_000_000 + 60 * i,
            "amt": 20.0 + i,
            "lat": 40.0,
            "long": -75.0,
            "merch_lat": 41.0,
            "merch_long": -74.0,
            "category": "a",
            "gender": "M",
            "state": "CA",
        }
        for i in range(5)
    ]
    window = build_feature_window(scorer, history)
    assert window.shape == (8, 11)

    prob, is_fraud, latency_ms = score_history(scorer, history)
    assert 0.0 <= prob <= 1.0
    assert is_fraud == (prob >= scorer.threshold)
    assert latency_ms > 0.0
