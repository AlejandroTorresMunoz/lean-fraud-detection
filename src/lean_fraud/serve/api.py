"""FastAPI fraud-scoring service.

Exposes /health, /predict and /assess/{card_id}. Loads the trained TCN once at startup and scores
real transaction sequences; the response echoes the server-side inference latency so the SLA is
observable.

- /predict   — score an explicit raw transaction sequence (stateless).
- /assess/{card_id} — the "one call" endpoint: look the card up in the data, score its history with
  the TCN, then run the LLM triage agent, and return BOTH the model result and the agent decision.

If the model artifacts are absent (e.g. a fresh CI checkout that never trained), startup degrades
gracefully: the service still boots and the scoring endpoints return 503 until a model exists.

Run: uvicorn lean_fraud.serve.api:app --port 8000
Config: LEAN_FRAUD_CONFIG (default configs/base.yaml). AGENT_PROVIDER overrides agent.provider
(e.g. `ollama` in the container so /assess returns a real LLM rationale).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lean_fraud.agent.graph import build_agent, triage
from lean_fraud.agent.llm import build_chat_model
from lean_fraud.agent.schema import AlertContext
from lean_fraud.agent.store import TransactionStore
from lean_fraud.config import load_config
from lean_fraud.serve.scoring import Scorer, build_feature_window, load_scorer, score, score_history

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
    # The card store + triage agent are heavier (they read the raw CSVs), so build them lazily on the
    # first /assess call and cache them, keeping /health and /predict fast.
    app.state.assessor = None
    yield


app = FastAPI(title="Lean Fraud Detection", version="0.1.0", lifespan=lifespan)


def _get_assessor(cfg: dict):
    """Lazily build + cache (store, agent, model) for /assess. Raises if data is missing."""
    if app.state.assessor is None:
        store = TransactionStore.from_config(cfg)
        model = build_chat_model(cfg)
        agent = build_agent(cfg, store=store, model=model)
        app.state.assessor = (store, agent, model)
    return app.state.assessor


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


@app.get("/assess/{card_id}", response_model=AssessResponse)
def assess(card_id: str) -> AssessResponse:
    """Score a card by id, then triage it: TCN fraud probability + the agent's decision & rationale."""
    scorer: Scorer | None = getattr(app.state, "scorer", None)
    if scorer is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded — train and build data first."
        )
    try:
        store, agent, model = _get_assessor(app.state.cfg)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=f"Card data unavailable: {exc}")

    history = store.raw_history(card_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"No transactions found for card {card_id}.")

    prob, is_fraud, latency_ms = score_history(scorer, history)
    # The agent investigates the flagged card (its tools read the same store) and always returns a
    # Decision — a real LLM rationale with the ollama backend, or a deterministic threshold fallback.
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
