"""Turn raw transactions into per-user, time-ordered sequences (windows of N past tx).

Critical: split STRICTLY by time per user to avoid temporal leakage. Output goes to
data/processed/ as arrays/parquet consumed by training.

Usage: python -m lean_fraud.data.build_sequences
"""

from __future__ import annotations

from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # TODO:
    #  1) load raw tx, sort by (user, timestamp)
    #  2) engineer features (amount_log, time_deltas, rolling aggs, encode categoricals)
    #  3) build sliding windows of length `sequence_length`
    #  4) time-based train/val/test split (no future in train)
    print(f"[build_sequences] output dir: {PROCESSED_DIR.resolve()}")
    print("[build_sequences] TODO: implement sequence construction + temporal split.")


if __name__ == "__main__":
    main()
