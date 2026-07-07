"""Tests for the triage agent graph (agent/graph.py).

The mock chat model cannot bind tools or emit structured output, so these tests do NOT simulate real
tool-calling. They pin the contract that actually matters: triage ALWAYS returns a valid Decision.
Happy-path extraction is checked by injecting a fake compiled agent that returns a structured
response; every failure mode (no structured answer, an exception, and the real tool-less mock
backend) must resolve to the deterministic threshold fallback. No network, no Ollama.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from lean_fraud.agent.graph import _fallback, triage
from lean_fraud.agent.schema import AlertContext, Decision
from lean_fraud.agent.store import TransactionStore

CFG = {
    "agent": {
        "provider": "mock",
        "recursion_limit": 8,
        "review_threshold": 0.5,
        "block_threshold": 0.9,
    }
}


def _ctx(score: float) -> AlertContext:
    return AlertContext.from_alert(
        {
            "tx": {
                "cc_num": 111,
                "amt": 500.0,
                "category": "shopping",
                "state": "CA",
                "unix_time": 1,
            },
            "prob": score,
        }
    )


class _FakeAgent:
    """Stand-in for a compiled create_agent graph: returns a canned result or raises."""

    def __init__(self, result: dict | None = None, error: Exception | None = None):
        self._result = result
        self._error = error

    def invoke(self, *args, **kwargs):
        if self._error is not None:
            raise self._error
        return self._result


def test_triage_returns_the_agents_structured_decision():
    decision = Decision(action="block", rationale="clear fraud", confidence=0.99)
    out = triage(_ctx(0.6), CFG, agent=_FakeAgent(result={"structured_response": decision}))
    assert out is decision


def test_triage_falls_back_when_no_structured_response():
    out = triage(_ctx(0.95), CFG, agent=_FakeAgent(result={"messages": []}))
    assert out.action == "block"  # 0.95 >= block_threshold
    assert out.rationale == "fallback"


def test_triage_falls_back_on_exception():
    out = triage(_ctx(0.6), CFG, agent=_FakeAgent(error=RuntimeError("boom")))
    assert out.action == "review"  # review_threshold <= 0.6 < block_threshold
    assert out.rationale == "fallback"


def test_triage_extracts_decision_from_prose_via_structured_output():
    # Tier 2: the agent reasoned in prose (no structured_response); the model's structured-output
    # call recovers the Decision. Small local models decide but fail to emit the schema in one turn.
    decision = Decision(action="review", rationale="prose-derived")
    agent = _FakeAgent(result={"messages": [("assistant", "Decision: review ...")]})
    model = SimpleNamespace(
        with_structured_output=lambda schema: SimpleNamespace(invoke=lambda messages: decision)
    )
    out = triage(_ctx(0.6), CFG, agent=agent, model=model)
    assert out is decision


def test_triage_with_real_mock_backend_falls_back_without_crashing():
    # The mock model can't bind tools or emit structured output; triage must still return a Decision
    # by falling through to the threshold fallback — never crash, never hang.
    df = pd.DataFrame(
        {
            "cc_num": [111],
            "unix_time": [1],
            "amt": [500.0],
            "category": ["shopping"],
            "state": ["CA"],
            "merch_lat": [0.0],
            "merch_long": [0.0],
            "is_fraud": [0],
        }
    )
    out = triage(_ctx(0.3), CFG, store=TransactionStore(df))
    assert isinstance(out, Decision)
    assert out.action == "allow"  # 0.3 < review_threshold
    assert out.rationale == "fallback"


@pytest.mark.parametrize("score,action", [(0.95, "block"), (0.6, "review"), (0.2, "allow")])
def test_fallback_thresholds(score, action):
    decision = _fallback(_ctx(score), CFG)
    assert decision.action == action
    assert decision.rationale == "fallback"
    assert decision.confidence == score
