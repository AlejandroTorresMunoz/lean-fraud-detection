"""Replay a dataset as a live stream of transactions into a Kinesis stream.

Talks to the LOCAL LocalStack endpoint (AWS_ENDPOINT_URL), so there is no AWS cost.

Usage: python -m lean_fraud.streaming.producer
"""

from __future__ import annotations

import json
import os
import time

import boto3

REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
TX_STREAM = os.getenv("KINESIS_TX_STREAM", "tx-stream")
ENDPOINT = os.getenv("AWS_ENDPOINT_URL")


def main(rate_hz: float = 50.0) -> None:
    if not ENDPOINT:
        print("WARNING: AWS_ENDPOINT_URL is not set — refusing to hit real AWS. Set it first.")
        return

    kinesis = boto3.client("kinesis", endpoint_url=ENDPOINT, region_name=REGION)
    print(f"[producer] putting records to {TX_STREAM} at ~{rate_hz} tx/s")

    # TODO: iterate over processed transactions instead of this demo event.
    demo_tx = {
        "user_id": "u-001",
        "amount": 42.0,
        "merchant_category": "groceries",
        "country": "ES",
    }
    while True:
        kinesis.put_record(
            StreamName=TX_STREAM,
            Data=json.dumps(demo_tx).encode("utf-8"),
            PartitionKey=str(demo_tx["user_id"]),
        )
        time.sleep(1.0 / rate_hz)


if __name__ == "__main__":
    main()
