from __future__ import annotations
from langchain_core.tools import BaseTool, tool
from lean_fraud.agent.store import TransactionStore


def build_tools(store: TransactionStore) -> list[BaseTool]:
    """Build the three read-only tools bound to a given TransactionStore."""

    @tool
    def get_card_profile(card_id: str) -> dict:
        """Get a card's baseline spending behaviour."""
        if not card_id or not str(card_id).strip():
            return {"error": "card_id must be a non-empty string."}
        return store.card_profile(card_id)

    @tool
    def get_recent_transactions(card_id: str, k: int = 5) -> list[dict]:
        """Get a card's k most recent transactions (oldest to newest)"""
        if not card_id or not str(card_id).strip():
            return {"error": "card_id must be a non-empty string."}
        if not isinstance(k, int) or k <= 0:
            return {"error": "k must be a positive integer."}
        return store.recent_transactions(str(card_id), k)

    @tool
    def get_population_fraud_rate(category: str, state: str) -> float | dict:
        """Get the base fraud rate for a (category, state) segment — the share of transactions in
        that segment that were fraudulent. Use this for context on how risky the segment is"""
        if not category or not str(category).strip():
            return {"error": "category must be a non-empty string."}
        if not state or not str(state).strip():
            return {"error": "state must be a non-empty string."}
        return store.population_fraud_rate(str(category), str(state))

    return [get_card_profile, get_recent_transactions, get_population_fraud_rate]
