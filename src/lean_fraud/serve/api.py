"""FastAPI fraud-scoring service.

Exposes /health and /predict. Loads the trained TCN once at startup and scores real transaction
sequences; the response echoes the server-side inference latency so the SLA is observable.

If the model artifacts are absent (e.g. a fresh CI checkout that never trained), startup degrades
gracefully: the service still boots and /predict returns 503 until a model exists.

Run: uvicorn lean_fraud.serve.api:app --port 8000
Config: LEAN_FRAUD_CONFIG (default configs/base.yaml).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lean_fraud.config import load_config
from lean_fraud.serve.scoring import Scorer, build_feature_window, load_scorer, score

CONFIG_PATH = os.getenv("LEAN_FRAUD_CONFIG", "configs/base.yaml")


class RawTransaction(BaseModel):
    """One raw transaction in a card's history — the same fields build_sequences consumes.

    Features (amount stats, inter-tx delta, geo distance, time-of-day, category codes) are engineered
    server-side from these, so clients send raw values, not pre-computed features.
    """

    amt: float = Field(..., description="Transaction amount")
    unix_time: int = Field(..., description="Epoch seconds; orders the card's transactions")
    category: str = ""
    gender: str = ""
    state: str = ""
    lat: float = 0.0
    long: float = 0.0
    merch_lat: float = 0.0
    merch_long: float = 0.0
    cc_num: str | None = Field(None, description="Card id; optional (one card per request)")


class ScoreRequest(BaseModel):
    sequence: list[RawTransaction] = Field(
        ..., description="A card's recent transaction history, oldest first"
    )


class ScoreResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    latency_ms: float


def _try_load_scorer() -> Scorer | None:
    """Load the scorer, or None (with a logged reason) if artifacts/data are missing."""
    try:
        return load_scorer(load_config(CONFIG_PATH))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[api] model not loaded ({exc}); /predict returns 503 until artifacts exist.")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = _try_load_scorer()
    yield


app = FastAPI(title="Lean Fraud Detection", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str(getattr(app.state, "scorer", None) is not None)}


@app.post("/predict", response_model=ScoreResponse)
def predict(req: ScoreRequest) -> ScoreResponse:
    scorer: Scorer | None = getattr(app.state, "scorer", None)
    if scorer is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded — train and build data first."
        )
    window = build_feature_window(scorer, [tx.model_dump() for tx in req.sequence])
    prob, is_fraud, latency_ms = score(scorer, window)
    return ScoreResponse(fraud_probability=prob, is_fraud=is_fraud, latency_ms=latency_ms)
