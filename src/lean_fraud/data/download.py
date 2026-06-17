"""Download a public transaction dataset into data/raw/.

Datasets (see README): tabformer (primary), ieee-cis, paysim. Kaggle ones need the Kaggle API
(`kaggle datasets download ...`); document credentials in the README rather than committing them.

Usage: python -m lean_fraud.data.download
"""

from __future__ import annotations

from pathlib import Path

RAW_DIR = Path("data/raw")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: implement per-dataset download (Kaggle API / direct URL).
    # For a first run, PaySim is the smallest and quickest to iterate on.
    print(f"[download] target dir: {RAW_DIR.resolve()}")
    print("[download] TODO: fetch dataset (start with PaySim, then TabFormer).")


if __name__ == "__main__":
    main()
