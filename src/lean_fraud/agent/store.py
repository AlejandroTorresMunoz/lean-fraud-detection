"""Read-only lookups over the raw Sparkov transactions, backing the agent's tools.

The agent reasons over human-readable values (euros, category, state, the fraud base rate), so this
reads the RAW CSVs — not the scaled/encoded `sequences.npz`. These are the same files `producer.py`
replays. Lookups are indexed once at construction (999 cards, so a per-card dict is cheap) and every
method returns plain Python types the LLM tools can serialize. Nothing here touches LangChain.

Inject a DataFrame for tests; use `from_config` in production to load `data/raw/` per the config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lean_fraud.data.transform.split import TEST, time_split

RAW_FILES = ["fraudTrain.csv", "fraudTest.csv"]
USE_COLS = [
    "cc_num",
    "unix_time",
    "amt",
    "category",
    "state",
    "merch_lat",
    "merch_long",
    "is_fraud",
]


class TransactionStore:
    """Per-card history + population fraud rates over the raw Sparkov table."""

    def __init__(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df["card"] = df["cc_num"].astype(str)  # big ints -> str, matches AlertContext.card_id
        df = df.sort_values("unix_time").reset_index(drop=True)
        self._df = df  # kept for time-based test-split sampling (sample_test_transaction)
        # 999 cards -> a per-card dict is small and gives O(1) history lookups.
        self._by_card: dict[str, pd.DataFrame] = {
            card: group for card, group in df.groupby("card", sort=False)
        }
        # Base fraud rate per (category, state) segment, precomputed once for cheap lookups.
        self._pop_fraud = df.groupby(["category", "state"])["is_fraud"].mean()

    @classmethod
    def from_config(cls, cfg: dict) -> "TransactionStore":
        """Load the raw Sparkov CSVs from ``cfg['dataset']['raw_dir']``."""
        raw_dir = Path(cfg["dataset"]["raw_dir"])
        frames = [
            pd.read_csv(raw_dir / fname, usecols=USE_COLS)
            for fname in RAW_FILES
            if (raw_dir / fname).exists()
        ]
        if not frames:
            raise FileNotFoundError(
                f"No Sparkov CSVs in {raw_dir}. Run `python -m lean_fraud.data.download` first."
            )
        return cls(pd.concat(frames, ignore_index=True))

    def card_profile(self, card_id: str) -> dict:
        """Baseline behaviour for a card. Unknown card -> a zeroed profile, never an error."""
        group = self._by_card.get(str(card_id))
        if group is None or group.empty:
            return {
                "card_id": str(card_id),
                "tx_count": 0,
                "median_amt": 0.0,
                "avg_amt": 0.0,
                "top_categories": [],
                "home_state": None,
            }
        states = group["state"].mode()
        return {
            "card_id": str(card_id),
            "tx_count": int(len(group)),
            "median_amt": round(float(group["amt"].median()), 2),
            "avg_amt": round(float(group["amt"].mean()), 2),
            "top_categories": group["category"].value_counts().head(3).index.tolist(),
            "home_state": None if states.empty else str(states.iloc[0]),
        }

    def recent_transactions(self, card_id: str, k: int = 5) -> list[dict]:
        """The card's k most recent transactions (oldest -> newest). Unknown card -> []."""
        group = self._by_card.get(str(card_id))
        if group is None or group.empty or k <= 0:
            return []
        tail = group.tail(int(k))  # already globally sorted by unix_time
        return [
            {
                "unix_time": int(row.unix_time),
                "amt": round(float(row.amt), 2),
                "category": str(row.category),
                "state": str(row.state),
            }
            for row in tail.itertuples(index=False)
        ]

    def population_fraud_rate(self, category: str, state: str) -> float:
        """Base fraud rate for a (category, state) segment. Unseen segment -> 0.0."""
        try:
            return round(float(self._pop_fraud.loc[(category, state)]), 4)
        except KeyError:
            return 0.0

    def sample_test_transaction(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        rng: np.random.Generator | None = None,
        fraud_only: bool = False,
    ) -> dict:
        """Return a random raw transaction from the time-based TEST split (most-recent fraction).

        Reuses the training split logic (`transform.split.time_split`), so the demo/eval draws from
        genuinely held-out data — no train/val rows leak in. Returns the raw tx columns as a dict.
        """
        labels = time_split(self._df["unix_time"].to_numpy(), test_size, val_size)
        mask = labels == TEST
        if fraud_only:
            mask &= self._df["is_fraud"].to_numpy() == 1
        pool = self._df[mask]
        if pool.empty:
            raise ValueError("no TEST-split transactions match the filter")
        rng = rng or np.random.default_rng()
        return pool.iloc[int(rng.integers(len(pool)))].to_dict()
