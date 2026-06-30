"""Unit tests for the shared scoring core (serve/scoring.py) and the FastAPI /predict surface.

Pure tests: a hand-built Scorer with a tiny TCN and synthetic scaler/maps — no trained artifacts or
dataset needed, so they run instantly in CI. They pin the serving contract: a card's raw history is
turned into a correctly shaped, left-padded window and a probability in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pytest

from lean_fraud.models.tcn import TCNClassifier
from lean_fraud.serve.scoring import Scorer, build_feature_window, score, score_history

FEATS = {
    "amount_log": True,
    "time_deltas": True,
    "rolling_aggs": True,
    "geo_distance": True,
    "time_features": True,
    "categorical": ["category", "gender", "state"],
}
N_NUMERIC = 8  # amt, amt_log, dt, amt_roll_mean, amt_count, geo_dist, hour, dow
SEQ_LEN = 4


def _scorer(threshold: float = 0.5) -> Scorer:
    n_features = N_NUMERIC + 3  # + category/gender/state codes
    model = TCNClassifier(n_features=n_features, channels=[8, 8], kernel_size=2)
    model.eval()
    return Scorer(
        model=model,
        feats_cfg=FEATS,
        seq_len=SEQ_LEN,
        n_numeric=N_NUMERIC,
        scaler_mean=np.zeros(
            N_NUMERIC, dtype=np.float32
        ),  # identity scaling keeps it deterministic
        scaler_std=np.ones(N_NUMERIC, dtype=np.float32),
        categorical_maps={
            "category": {"a": 1, "b": 2},
            "gender": {"M": 1, "F": 2},
            "state": {"CA": 1, "NY": 2},
        },
        threshold=threshold,
    )


def _tx(t: int, amt: float, **over) -> dict:
    base = dict(
        cc_num="card-1",
        unix_time=t,
        amt=amt,
        lat=0.0,
        long=0.0,
        merch_lat=3.0,
        merch_long=4.0,
        category="a",
        gender="M",
        state="CA",
    )
    base.update(over)
    return base


def test_window_shape_and_left_padding():
    scorer = _scorer()
    # Two transactions, seq_len 4 -> the two real rows sit at the bottom, top two are zero pad.
    window = build_feature_window(scorer, [_tx(100, 10.0), _tx(160, 20.0)])
    assert window.shape == (SEQ_LEN, N_NUMERIC + 3)
    assert np.allclose(window[:2], 0.0)  # left padding
    assert not np.allclose(window[2:], 0.0)  # the real rows carry signal


def test_window_keeps_last_seq_len_when_history_is_long():
    scorer = _scorer()
    history = [_tx(100 + 10 * i, float(i + 1)) for i in range(10)]  # 10 > seq_len
    window = build_feature_window(scorer, history)
    assert window.shape == (SEQ_LEN, N_NUMERIC + 3)
    assert not np.allclose(window, 0.0)  # no padding when history exceeds the window


def test_empty_history_is_all_zeros():
    scorer = _scorer()
    window = build_feature_window(scorer, [])
    assert window.shape == (SEQ_LEN, N_NUMERIC + 3)
    assert np.allclose(window, 0.0)


def test_unknown_category_maps_to_zero():
    scorer = _scorer()
    window = build_feature_window(scorer, [_tx(100, 10.0, category="UNSEEN")])
    # category code is the first categorical column, right after the numeric block.
    assert window[-1, N_NUMERIC] == 0.0  # unseen level -> 0, never a leaked code


def test_score_returns_probability_and_latency():
    scorer = _scorer()
    window = build_feature_window(scorer, [_tx(100, 10.0), _tx(160, 500.0)])
    prob, is_fraud, latency_ms = score(scorer, window)
    assert 0.0 <= prob <= 1.0
    assert isinstance(is_fraud, bool)
    assert is_fraud == (prob >= scorer.threshold)
    assert latency_ms > 0.0


def test_threshold_controls_decision():
    window = build_feature_window(_scorer(), [_tx(100, 10.0)])
    prob, _, _ = score(_scorer(), window)
    always = score(_scorer(threshold=0.0), window)
    never = score(_scorer(threshold=1.0001), window)
    assert always[1] is True  # everything flagged at threshold 0
    assert never[1] is False  # nothing flagged above the max probability


def test_score_history_matches_build_then_score():
    scorer = _scorer()
    history = [_tx(100, 10.0), _tx(160, 20.0)]
    direct = score(scorer, build_feature_window(scorer, history))
    combined = score_history(scorer, history)
    assert direct[0] == pytest.approx(combined[0])  # same probability via either path


def test_api_predict_uses_injected_scorer():
    # Drive the FastAPI handler directly (no httpx/TestClient needed): inject a scorer, call predict.
    from lean_fraud.serve.api import RawTransaction, ScoreRequest, app, predict

    app.state.scorer = _scorer()
    req = ScoreRequest(sequence=[RawTransaction(amt=10.0, unix_time=100, category="a")])
    resp = predict(req)
    assert 0.0 <= resp.fraud_probability <= 1.0
    assert resp.latency_ms > 0.0


def test_api_predict_503_when_model_missing():
    from fastapi import HTTPException

    from lean_fraud.serve.api import RawTransaction, ScoreRequest, app, predict

    app.state.scorer = None
    with pytest.raises(HTTPException) as exc:
        predict(ScoreRequest(sequence=[RawTransaction(amt=10.0, unix_time=100)]))
    assert exc.value.status_code == 503
