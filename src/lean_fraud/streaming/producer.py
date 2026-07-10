"""Replay the dataset as a live stream of RAW transactions into a Kinesis stream.

Reads the raw Sparkov CSVs (the same USE_COLS the ETL consumes), replays them time-ordered so the
consumer can engineer features and score exactly as in training. Talks to the LOCAL LocalStack
endpoint (AWS_ENDPOINT_URL), so there is no AWS cost.

Usage: python -m lean_fraud.streaming.producer
Config: LEAN_FRAUD_CONFIG (default configs/base.yaml).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import boto3
import pandas as pd

from lean_fraud.config import load_config
from lean_fraud.data.build_sequences import RAW_FILES, TIME_COL, USE_COLS, _load_raw

REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
TX_STREAM = os.getenv("KINESIS_TX_STREAM", "tx-stream")
ENDPOINT = os.getenv("AWS_ENDPOINT_URL")
CONFIG_PATH = os.getenv("LEAN_FRAUD_CONFIG", "configs/base.yaml")


def _load_stream(raw_dir: str) -> pd.DataFrame:
    """Raw transactions sorted by time — a global time order replays a realistic live stream."""
    df = _load_raw(Path(raw_dir))
    return df[USE_COLS].sort_values(TIME_COL).reset_index(drop=True)


def main(rate_hz: float = 500.0, limit: int | None = None, batch_size: int = 500) -> None:
    if not ENDPOINT:
        print("WARNING: AWS_ENDPOINT_URL is not set — refusing to hit real AWS. Set it first.")
        return

    cfg = load_config(CONFIG_PATH)
    df = _load_stream(cfg["dataset"]["raw_dir"])
    if limit is not None:
        df = df.head(limit)

    kinesis = boto3.client("kinesis", endpoint_url=ENDPOINT, region_name=REGION)
    print(f"[producer] replaying {len(df)} tx to {TX_STREAM} at ~{rate_hz} tx/s (from {RAW_FILES})")

    # Batch with put_records (up to 500/call): one HTTP round-trip per record throttles the stream to
    # a crawl, so we send in batches and pace ~rate_hz tx/s between them. Order is preserved (records
    # stay time-sorted within and across batches), so the consumer's causal features are unaffected.
    batch: list[dict] = []
    sent = 0
    for tx in df.to_dict(orient="records"):
        tx["cc_num"] = str(tx["cc_num"])  # JSON-safe + stable partition/card key
        batch.append({"Data": json.dumps(tx).encode("utf-8"), "PartitionKey": tx["cc_num"]})
        if len(batch) >= batch_size:
            kinesis.put_records(StreamName=TX_STREAM, Records=batch)
            sent += len(batch)
            batch = []
            time.sleep(batch_size / rate_hz)  # pace the stream ~rate_hz tx/s
    if batch:
        kinesis.put_records(StreamName=TX_STREAM, Records=batch)
        sent += len(batch)
    print(f"[producer] done ({sent} tx)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay Sparkov transactions into Kinesis.")
    parser.add_argument("--rate-hz", type=float, default=500.0, help="transactions per second")
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after N transactions (default: full replay)"
    )
    parser.add_argument("--batch-size", type=int, default=500, help="records per put_records call")
    args = parser.parse_args()
    main(rate_hz=args.rate_hz, limit=args.limit, batch_size=args.batch_size)
