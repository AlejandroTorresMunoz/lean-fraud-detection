"""Unit tests for the TransactionStore lookups that back the listing / streaming API endpoints.

No LLM, no network, no on-disk data — a small in-memory DataFrame pins the contract of card_ids,
raw_history and iter_all_raw (used by /cards, /cards/{id}/transactions and the stream endpoints).
"""

from __future__ import annotations

import pandas as pd

from lean_fraud.agent.store import TransactionStore


def _store() -> TransactionStore:
    # Card 111: three tx (t=1,2,3). Card 222: one tx (t=5). Rows deliberately out of time order.
    df = pd.DataFrame(
        {
            "cc_num": [111, 111, 222, 111],
            "unix_time": [2, 1, 5, 3],
            "amt": [200.0, 20.0, 7.0, 10.0],
            "category": ["shopping", "grocery", "gas", "grocery"],
            "state": ["NY", "CA", "TX", "CA"],
            "merch_lat": [0.0, 0.0, 0.0, 0.0],
            "merch_long": [0.0, 0.0, 0.0, 0.0],
            "is_fraud": [0, 1, 0, 0],
        }
    )
    return TransactionStore(df)


def test_card_ids_lists_every_card_as_string():
    assert set(_store().card_ids()) == {"111", "222"}


def test_raw_history_is_full_and_time_ordered():
    hist = _store().raw_history("111")
    assert [tx["unix_time"] for tx in hist] == [1, 2, 3]  # oldest -> newest
    assert "is_fraud" not in hist[0]  # label never leaks into the scoring rows
    assert hist[0]["cc_num"] == "111"


def test_raw_history_unknown_card_is_empty():
    assert _store().raw_history("999") == []


def test_iter_all_raw_is_global_time_order():
    rows = list(_store().iter_all_raw())
    assert [r["unix_time"] for r in rows] == [1, 2, 3, 5]  # across BOTH cards, by time


def test_iter_all_raw_respects_limit():
    assert len(list(_store().iter_all_raw(limit=2))) == 2
    assert len(list(_store().iter_all_raw(limit=0))) == 4  # <=0 -> all
