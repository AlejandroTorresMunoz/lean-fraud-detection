"""Consume the transaction stream, score each tx with the real TCN, and emit fraud alerts.

Reads from the tx Kinesis stream on the LOCAL LocalStack endpoint, keeps a per-card history so the
causal rolling features (amt_roll_mean / amt_count / dt) reproduce training exactly, scores with the
shared `serve.scoring` core, and re-publishes flagged transactions to the alerts stream using the
val-tuned decision threshold (not a hardcoded 0.5).

Usage: python -m lean_fraud.streaming.consumer
Config: LEAN_FRAUD_CONFIG (default configs/base.yaml).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import boto3

from lean_fraud.config import load_config
from lean_fraud.serve.scoring import load_scorer, score_history

REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
TX_STREAM = os.getenv("KINESIS_TX_STREAM", "tx-stream")
ALERTS_STREAM = os.getenv("KINESIS_ALERTS_STREAM", "alerts-stream")
ENDPOINT = os.getenv("AWS_ENDPOINT_URL")
CONFIG_PATH = os.getenv("LEAN_FRAUD_CONFIG", "configs/base.yaml")


def _card_key(tx: dict) -> str:
    return str(tx.get("cc_num", tx.get("user_id", "unknown")))


def main(poll_seconds: float = 1.0) -> None:
    if not ENDPOINT:
        print("WARNING: AWS_ENDPOINT_URL is not set — refusing to hit real AWS. Set it first.")
        return

    try:
        scorer = load_scorer(load_config(CONFIG_PATH))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[consumer] cannot load model ({exc}). Train + build data first.")
        return

    kinesis = boto3.client("kinesis", endpoint_url=ENDPOINT, region_name=REGION)
    shards = kinesis.describe_stream(StreamName=TX_STREAM)["StreamDescription"]["Shards"]
    iterators = {
        s["ShardId"]: kinesis.get_shard_iterator(
            StreamName=TX_STREAM, ShardId=s["ShardId"], ShardIteratorType="LATEST"
        )["ShardIterator"]
        for s in shards
    }
    # Per-card raw history so rolling/expanding features match training (unbounded for the demo;
    # trimming to a max history is a documented future improvement).
    history: dict[str, list[dict]] = defaultdict(list)
    print(
        f"[consumer] reading {TX_STREAM} across {len(iterators)} shard(s) @ thr={scorer.threshold:.4f}"
    )

    while True:
        for shard_id, iterator in list(iterators.items()):
            resp = kinesis.get_records(ShardIterator=iterator, Limit=100)
            iterators[shard_id] = resp["NextShardIterator"]
            for rec in resp["Records"]:
                tx = json.loads(rec["Data"].decode("utf-8"))
                key = _card_key(tx)
                history[key].append(tx)
                prob, is_fraud, _ = score_history(scorer, history[key])
                if is_fraud:
                    kinesis.put_record(
                        StreamName=ALERTS_STREAM,
                        Data=json.dumps({"tx": tx, "prob": prob}).encode("utf-8"),
                        PartitionKey=key,
                    )
                    print(f"[consumer] ALERT prob={prob:.3f} card={key} amt={tx.get('amt')}")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
