"""Pydantic contracts for the fraud-triage agent.

Two shapes bound the agent: `AlertContext` is the typed wrapper around the alert the consumer emits
(`{"tx": {...}, "prob": float}`), and `Decision` is the structured output the agent must return —
enforced via LangChain `with_structured_output(Decision)` so the LLM can never hand back free text.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Decision(BaseModel):
    """The agent's triage decision for a flagged transaction."""

    action: Literal["block", "review", "allow"]
    rationale: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Transaction(BaseModel):
    """The raw fields of the flagged transaction the agent reasons about."""

    amt: float
    category: str
    state: str
    unix_time: int
    merch_lat: float | None = None
    merch_long: float | None = None


class AlertContext(BaseModel):
    """The alert the agent triages: the flagged tx + its TCN score + the card id."""

    card_id: str
    score: float = Field(ge=0.0, le=1.0)  # TCN fraud probability from the consumer
    transaction: Transaction

    @classmethod
    def from_alert(cls, alert: dict) -> "AlertContext":
        """Build from the consumer's ``{"tx": {...}, "prob": float}`` alert payload."""
        tx = alert["tx"]
        return cls(
            card_id=str(tx.get("cc_num", tx.get("user_id", "unknown"))),
            score=alert["prob"],
            transaction=Transaction(**{k: tx[k] for k in Transaction.model_fields if k in tx}),
        )
