"""FastAPI fraud-scoring service.

Endpoints:
  - GET  /health                        — liveness + whether a model is loaded.
  - POST /predict                       — score an explicit raw transaction sequence (stateless).
  - GET  /assess/{card_id}              — the "one call": look the card up, score its history with the
                                          TCN, run the LLM triage agent, return BOTH results.
  - GET  /cards                         — list known card ids (?limit).
  - GET  /cards/{card_id}/transactions  — a card's raw transaction history (?limit, most recent).
  - GET  /cards/{card_id}/stream        — NDJSON stream: score each of the card's tx live (?limit,?rate_hz).
  - GET  /stream                        — NDJSON firehose: score all cards in time order (?limit,?rate_hz).

If the model artifacts are absent (e.g. a fresh CI checkout that never trained), startup degrades
gracefully: the service still boots and the scoring endpoints return 503 until a model exists.

Run: uvicorn lean_fraud.serve.api:app --port 8000
Config: LEAN_FRAUD_CONFIG (default configs/base.yaml). AGENT_PROVIDER overrides agent.provider
(e.g. `ollama` in the container so /assess returns a real LLM rationale).
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from lean_fraud.agent.graph import build_agent, triage
from lean_fraud.agent.llm import build_chat_model
from lean_fraud.agent.schema import AlertContext
from lean_fraud.agent.store import TransactionStore
from lean_fraud.config import load_config
from lean_fraud.serve.scoring import Scorer, build_feature_window, load_scorer, score, score_history

CONFIG_PATH = os.getenv("LEAN_FRAUD_CONFIG", "configs/base.yaml")
# Fields echoed back per transaction in the list / stream endpoints.
_TX_FIELDS = ("unix_time", "amt", "category", "state")


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


class AssessResponse(BaseModel):
    """The combined TCN + agent verdict for a card."""

    card_id: str
    transactions_considered: int
    fraud_probability: float
    is_fraud: bool
    latency_ms: float
    triage: dict  # the agent's Decision (action / rationale / confidence)


def _load_cfg() -> dict:
    """Load the config and apply the AGENT_PROVIDER env override (used to flip to ollama in Docker)."""
    cfg = load_config(CONFIG_PATH)
    provider = os.getenv("AGENT_PROVIDER")
    if provider:
        cfg.setdefault("agent", {})["provider"] = provider
    return cfg


def _try_load_scorer(cfg: dict) -> Scorer | None:
    """Load the scorer, or None (with a logged reason) if artifacts/data are missing."""
    try:
        return load_scorer(cfg)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[api] model not loaded ({exc}); scoring returns 503 until artifacts exist.")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cfg = _load_cfg()
    app.state.scorer = _try_load_scorer(app.state.cfg)
    # The card store (reads the raw CSVs) and the triage agent are heavier, so build them lazily on
    # first use and cache them, keeping /health and /predict fast.
    app.state.store = None
    app.state.agent = None
    yield


app = FastAPI(title="Lean Fraud Detection", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send the bare host to the interactive Swagger docs."""
    return RedirectResponse(url="/docs")


def _require_scorer() -> Scorer:
    scorer: Scorer | None = getattr(app.state, "scorer", None)
    if scorer is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded — train and build data first."
        )
    return scorer


def _require_store() -> TransactionStore:
    """Lazily build + cache the card store, or 503 if the raw data is missing."""
    if app.state.store is None:
        try:
            app.state.store = TransactionStore.from_config(app.state.cfg)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=503, detail=f"Card data unavailable: {exc}")
    return app.state.store


def _require_agent(store: TransactionStore):
    """Lazily build + cache (agent, model) for triage."""
    if app.state.agent is None:
        model = build_chat_model(app.state.cfg)
        app.state.agent = (build_agent(app.state.cfg, store=store, model=model), model)
    return app.state.agent


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str(getattr(app.state, "scorer", None) is not None)}


@app.post("/predict", response_model=ScoreResponse)
def predict(req: ScoreRequest) -> ScoreResponse:
    scorer = _require_scorer()
    window = build_feature_window(scorer, [tx.model_dump() for tx in req.sequence])
    prob, is_fraud, latency_ms = score(scorer, window)
    return ScoreResponse(fraud_probability=prob, is_fraud=is_fraud, latency_ms=latency_ms)


@app.get("/assess/{card_id}", response_model=AssessResponse)
def assess(card_id: str) -> AssessResponse:
    """Score a card by id, then triage it: TCN fraud probability + the agent's decision & rationale."""
    scorer = _require_scorer()
    store = _require_store()
    history = store.raw_history(card_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"No transactions found for card {card_id}.")

    prob, is_fraud, latency_ms = score_history(scorer, history)
    # The agent investigates the flagged card (its tools read the same store) and always returns a
    # Decision — a real LLM rationale with the ollama backend, or a deterministic threshold fallback.
    agent, model = _require_agent(store)
    alert = {"tx": history[-1], "prob": prob}
    decision = triage(AlertContext.from_alert(alert), app.state.cfg, agent=agent, model=model)
    return AssessResponse(
        card_id=card_id,
        transactions_considered=len(history),
        fraud_probability=prob,
        is_fraud=is_fraud,
        latency_ms=latency_ms,
        triage=decision.model_dump(),
    )


@app.get("/cards")
def list_cards(limit: int = 100) -> dict:
    """List known card ids. `limit<=0` returns all (999)."""
    store = _require_store()
    ids = store.card_ids()
    shown = ids if limit <= 0 else ids[:limit]
    return {"count": len(ids), "returned": len(shown), "card_ids": shown}


@app.get("/cards/{card_id}/transactions")
def card_transactions(card_id: str, limit: int = 20) -> dict:
    """A card's raw transaction history (most recent `limit`; `limit<=0` returns all)."""
    store = _require_store()
    history = store.raw_history(card_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"No transactions found for card {card_id}.")
    txs = history if limit <= 0 else history[-limit:]
    return {"card_id": card_id, "count": len(history), "returned": len(txs), "transactions": txs}


def _ndjson(rows: Iterator[dict], rate_hz: float) -> Iterator[str]:
    """Serialize scored rows as newline-delimited JSON, optionally paced at ~rate_hz rows/s."""
    for row in rows:
        yield json.dumps(row) + "\n"
        if rate_hz and rate_hz > 0:
            time.sleep(1.0 / rate_hz)


@app.get("/cards/{card_id}/stream")
def stream_card(card_id: str, limit: int = 0, rate_hz: float = 0.0) -> StreamingResponse:
    """Stream (NDJSON) the card's transactions scored live, oldest -> newest, causal per tx."""
    scorer = _require_scorer()
    store = _require_store()
    history = store.raw_history(card_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"No transactions found for card {card_id}.")
    n = len(history) if limit <= 0 else min(limit, len(history))

    def rows() -> Iterator[dict]:
        for i in range(n):
            prob, is_fraud, latency_ms = score_history(scorer, history[: i + 1])
            yield {
                "seq": i,
                "card_id": card_id,
                **{k: history[i].get(k) for k in _TX_FIELDS},
                "fraud_probability": round(prob, 4),
                "is_fraud": is_fraud,
                "latency_ms": round(latency_ms, 3),
            }

    return StreamingResponse(_ndjson(rows(), rate_hz), media_type="application/x-ndjson")


@app.get("/stream")
def stream_all(limit: int = 200, rate_hz: float = 0.0) -> StreamingResponse:
    """Stream (NDJSON) all cards' transactions scored in global time order (the firehose).

    Keeps per-card history so the causal features match training, exactly like the stream consumer.
    `limit<=0` streams the whole dataset (~1.85M tx) — bound it for a terminal demo.
    """
    scorer = _require_scorer()
    store = _require_store()

    def rows() -> Iterator[dict]:
        history: dict[str, list[dict]] = {}
        for tx in store.iter_all_raw(limit):
            card = str(tx.get("cc_num", "unknown"))
            history.setdefault(card, []).append(tx)
            prob, is_fraud, _ = score_history(scorer, history[card])
            yield {
                "card_id": card,
                **{k: tx.get(k) for k in _TX_FIELDS},
                "fraud_probability": round(prob, 4),
                "is_fraud": is_fraud,
            }

    return StreamingResponse(_ndjson(rows(), rate_hz), media_type="application/x-ndjson")
