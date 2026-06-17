"""Smoke tests: the package imports, the TCN runs a forward pass, the API answers."""

from __future__ import annotations

import torch
from fastapi.testclient import TestClient

from lean_fraud.config import load_config
from lean_fraud.models.tcn import TCNClassifier
from lean_fraud.serve.api import app


def test_config_loads():
    cfg = load_config("configs/base.yaml")
    assert cfg["model"]["type"] == "tcn"


def test_tcn_forward_and_param_count():
    model = TCNClassifier(n_features=8, channels=[16, 16])
    x = torch.randn(4, 32, 8)  # (batch, seq_len, features)
    out = model(x)
    assert out.shape == (4,)
    assert model.count_parameters() > 0


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_endpoint():
    client = TestClient(app)
    payload = {
        "sequence": [{"amount": 1200.0, "merchant_category": "electronics", "country": "ES"}]
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert "latency_ms" in body
