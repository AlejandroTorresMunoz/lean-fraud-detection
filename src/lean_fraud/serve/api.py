"""FastAPI fraud-scoring service.

Exposes /health and /predict. Designed for sub-50 ms scoring; the response echoes the server-side
latency so the SLA is observable.

Run: uvicorn lean_fraud.serve.api:app --port 8000
"""

from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Lean Fraud Detection", version="0.1.0")


class Transaction(BaseModel):
    """One step in a transaction sequence (minimal demo schema)."""

    amount: float = Field(..., description="Transaction amount")
    merchant_category: str = ""
    country: str = ""
    seconds_since_prev: float = 0.0


class ScoreRequest(BaseModel):
    sequence: list[Transaction] = Field(..., description="Recent transaction history, oldest first")


class ScoreResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    latency_ms: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=ScoreResponse)
def predict(req: ScoreRequest) -> ScoreResponse:
    t0 = time.perf_counter()
    # TODO: load the trained TCN once at startup and run real inference here.
    # Placeholder heuristic so the endpoint is runnable end-to-end before the model lands.
    last_amount = req.sequence[-1].amount if req.sequence else 0.0
    prob = 1.0 / (1.0 + pow(2.718, -(last_amount - 500.0) / 200.0))  # toy sigmoid on amount
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return ScoreResponse(fraud_probability=prob, is_fraud=prob > 0.5, latency_ms=latency_ms)
