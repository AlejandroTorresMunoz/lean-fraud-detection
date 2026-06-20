"""Validate that configs/base.yaml has the structure the data pipeline expects.

Fails fast if the experiment config drifts (missing keys, bad ranges). Pure config checks — no torch
or FastAPI — so they run instantly.
"""

from __future__ import annotations

from lean_fraud.config import load_config

CONFIG_PATH = "configs/base.yaml"


def test_config_loads_as_dict():
    assert isinstance(load_config(CONFIG_PATH), dict)


def test_dataset_section():
    ds = load_config(CONFIG_PATH)["dataset"]
    assert ds["name"] in {"sparkov", "tabformer", "ieee-cis"}
    assert isinstance(ds["raw_dir"], str) and ds["raw_dir"]
    assert isinstance(ds["processed_dir"], str) and ds["processed_dir"]
    assert isinstance(ds["sequence_length"], int) and ds["sequence_length"] > 0
    assert 0 < ds["test_size"] < 1
    assert 0 < ds["val_size"] < 1
    assert ds["test_size"] + ds["val_size"] < 1  # train split stays non-empty


def test_features_section():
    feats = load_config(CONFIG_PATH)["features"]
    for flag in ("amount_log", "time_deltas", "rolling_aggs", "geo_distance", "time_features"):
        assert isinstance(feats[flag], bool)
    assert isinstance(feats["user_key"], list) and feats["user_key"]
    assert isinstance(feats["categorical"], list) and feats["categorical"]
