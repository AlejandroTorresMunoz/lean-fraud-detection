"""Unit tests for the agent tools (agent/tools.py) over a synthetic TransactionStore.

No LLM, no network: build_tools wraps a small in-memory store, so these run instantly in CI and pin
the tool contract — correct lookups on the happy path, and readable {"error": ...} observations
(never exceptions) on bad arguments so the agent can see the error and retry. Data edge cases
(unknown card, unseen segment) are delegated to the store, so here they must NOT surface as errors.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lean_fraud.agent.store import TransactionStore
from lean_fraud.agent.tools import build_tools


def _store() -> TransactionStore:
    # Card 111: three tx (grocery/CA, shopping/NY, grocery/CA), one fraud. Card 222: one tx.
    df = pd.DataFrame(
        {
            "cc_num": [111, 111, 111, 222],
            "unix_time": [1, 2, 3, 5],
            "amt": [20.0, 200.0, 10.0, 7.0],
            "category": ["grocery", "shopping", "grocery", "gas"],
            "state": ["CA", "NY", "CA", "TX"],
            "merch_lat": [0.0, 0.0, 0.0, 0.0],
            "merch_long": [0.0, 0.0, 0.0, 0.0],
            "is_fraud": [1, 0, 0, 0],
        }
    )
    return TransactionStore(df)


@pytest.fixture
def tools() -> dict:
    return {t.name: t for t in build_tools(_store())}


def test_build_tools_exposes_the_three_named_tools(tools):
    assert set(tools) == {
        "get_card_profile",
        "get_recent_transactions",
        "get_population_fraud_rate",
    }


def test_card_profile_happy_path(tools):
    out = tools["get_card_profile"].invoke({"card_id": "111"})
    assert out["tx_count"] == 3
    assert out["home_state"] == "CA"  # mode of [CA, NY, CA]
    assert "grocery" in out["top_categories"]


def test_card_profile_unknown_card_is_zeroed_not_error(tools):
    out = tools["get_card_profile"].invoke({"card_id": "999"})
    assert out["tx_count"] == 0
    assert "error" not in out  # data edge case, not an argument error


def test_card_profile_blank_id_returns_error_observation(tools):
    out = tools["get_card_profile"].invoke({"card_id": "  "})
    assert isinstance(out, dict) and "error" in out


def test_recent_transactions_returns_last_k_oldest_first(tools):
    out = tools["get_recent_transactions"].invoke({"card_id": "111", "k": 2})
    assert [r["unix_time"] for r in out] == [2, 3]  # last two of 111, oldest -> newest


def test_recent_transactions_non_positive_k_is_error(tools):
    out = tools["get_recent_transactions"].invoke({"card_id": "111", "k": 0})
    assert isinstance(out, dict) and "error" in out


def test_population_fraud_rate_happy_path(tools):
    rate = tools["get_population_fraud_rate"].invoke({"category": "grocery", "state": "CA"})
    assert rate == 0.5  # two grocery+CA rows, one fraudulent


def test_population_fraud_rate_unseen_segment_is_zero(tools):
    rate = tools["get_population_fraud_rate"].invoke({"category": "x", "state": "ZZ"})
    assert rate == 0.0


def test_population_fraud_rate_blank_arg_is_error(tools):
    out = tools["get_population_fraud_rate"].invoke({"category": "grocery", "state": ""})
    assert isinstance(out, dict) and "error" in out
