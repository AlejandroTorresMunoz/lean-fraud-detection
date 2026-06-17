"""Consume the transaction stream, score each tx, and put fraud alerts on the alerts stream.

Reads from the tx Kinesis stream on the LOCAL LocalStack endpoint, scores with the model
(or the toy heuristic until the model lands), and re-publishes flagged tx to the alerts stream.

Usage: python -m lean_fraud.streaming.consumer
"""

from __future__ import annotations

import json
import os
import time

import boto3

REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
TX_STREAM = os.getenv("KINESIS_TX_STREAM", "tx-stream")
ALERTS_STREAM = os.getenv("KINESIS_ALERTS_STREAM", "alerts-stream")
ENDPOINT = os.getenv("AWS_ENDPOINT_URL")


def _score(tx: dict) -> float:
    # TODO: maintain per-user sequence state and run the real TCN.
    amount = float(tx.get("amount", 0.0))
    return 1.0 / (1.0 + pow(2.718, -(amount - 500.0) / 200.0))


def main(poll_seconds: float = 1.0) -> None:
    if not ENDPOINT:
        print("WARNING: AWS_ENDPOINT_URL is not set — refusing to hit real AWS. Set it first.")
        return

    kinesis = boto3.client("kinesis", endpoint_url=ENDPOINT, region_name=REGION)
    shards = kinesis.describe_stream(StreamName=TX_STREAM)["StreamDescription"]["Shards"]
    iterators = {
        s["ShardId"]: kinesis.get_shard_iterator(
            StreamName=TX_STREAM, ShardId=s["ShardId"], ShardIteratorType="LATEST"
        )["ShardIterator"]
        for s in shards
    }
    print(f"[consumer] reading {TX_STREAM} across {len(iterators)} shard(s)")

    while True:
        for shard_id, iterator in list(iterators.items()):
            resp = kinesis.get_records(ShardIterator=iterator, Limit=100)
            iterators[shard_id] = resp["NextShardIterator"]
            for rec in resp["Records"]:
                tx = json.loads(rec["Data"].decode("utf-8"))
                prob = _score(tx)
                if prob > 0.5:
                    kinesis.put_record(
                        StreamName=ALERTS_STREAM,
                        Data=json.dumps({"tx": tx, "prob": prob}).encode("utf-8"),
                        PartitionKey=str(tx.get("user_id", "unknown")),
                    )
                    print(f"[consumer] ALERT prob={prob:.3f} tx={tx}")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
