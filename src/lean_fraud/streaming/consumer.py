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
from lean_fraud.agent.graph import build_agent, triage
from lean_fraud.agent.schema import AlertContext

REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
TX_STREAM = os.getenv("KINESIS_TX_STREAM", "tx-stream")
ALERTS_STREAM = os.getenv("KINESIS_ALERTS_STREAM", "alerts-stream")
ENDPOINT = os.getenv("AWS_ENDPOINT_URL")
CONFIG_PATH = os.getenv("LEAN_FRAUD_CONFIG", "configs/base.yaml")


def _card_key(tx: dict) -> str:
    return str(tx.get("cc_num", tx.get("user_id", "unknown")))


def _put_fraud_rate(cloudwatch, namespace: str, alerts: int, total: int) -> None:
    """Emit the windowed fraud-alert rate to CloudWatch so the alarm can watch it.

    This is the only thing actionable live: with no ground-truth labels at scoring time we can't
    monitor F1/PR-AUC in real time, but a spike in the alert rate flags an attack or drift. Best
    effort — a monitoring hiccup must never take the scorer down.
    """
    if total <= 0:
        return
    try:
        cloudwatch.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {"MetricName": "FraudAlertRate", "Value": alerts / total, "Unit": "None"},
                {"MetricName": "TransactionsScored", "Value": total, "Unit": "Count"},
            ],
        )
    except Exception as exc:  # noqa: BLE001 - monitoring must not crash the consumer
        print(f"[consumer] CloudWatch put_metric_data failed ({exc}); continuing.")


def main(poll_seconds: float = 1.0) -> None:
    if not ENDPOINT:
        print("WARNING: AWS_ENDPOINT_URL is not set — refusing to hit real AWS. Set it first.")
        return

    cfg = load_config(CONFIG_PATH)
    try:
        scorer = load_scorer(cfg)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[consumer] cannot load model ({exc}). Train + build data first.")
        return

    # Cascade: the cheap TCN flags the ~0.5%, then the agent triages only those. Build it once
    # (compiling the graph + loading the store is expensive). Degrade gracefully — if the agent can't
    # be built (no data / backend), keep scoring and just emit alerts without a decision.
    try:
        agent = build_agent(cfg)
    except Exception as exc:
        print(f"[consumer] triage agent unavailable ({exc}); alerts will carry no decision.")
        agent = None

    stream_cfg = cfg.get("streaming", {})
    cw_namespace = stream_cfg.get("cloudwatch_namespace", "LeanFraud")
    flush_seconds = stream_cfg.get("metric_flush_seconds", 30)

    kinesis = boto3.client("kinesis", endpoint_url=ENDPOINT, region_name=REGION)
    cloudwatch = boto3.client("cloudwatch", endpoint_url=ENDPOINT, region_name=REGION)
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

    # Windowed counters for the CloudWatch FraudAlertRate metric (see _put_fraud_rate).
    window_total = 0
    window_alerts = 0
    last_flush = time.time()

    while True:
        for shard_id, iterator in list(iterators.items()):
            resp = kinesis.get_records(ShardIterator=iterator, Limit=100)
            iterators[shard_id] = resp["NextShardIterator"]
            for rec in resp["Records"]:
                tx = json.loads(rec["Data"].decode("utf-8"))
                key = _card_key(tx)
                history[key].append(tx)
                prob, is_fraud, _ = score_history(scorer, history[key])
                window_total += 1
                if is_fraud:
                    window_alerts += 1
                    alert = {"tx": tx, "prob": prob}
                    if agent is not None:
                        decision = triage(AlertContext.from_alert(alert), cfg, agent=agent)
                        alert["decision"] = decision.model_dump()
                    kinesis.put_record(
                        StreamName=ALERTS_STREAM,
                        Data=json.dumps(alert).encode("utf-8"),
                        PartitionKey=key,
                    )
                    verdict = alert.get("decision", {}).get("action", "n/a")
                    print(
                        f"[consumer] ALERT prob={prob:.3f} card={key} "
                        f"amt={tx.get('amt')} decision={verdict}"
                    )

        # Flush the windowed fraud-alert rate to CloudWatch on a fixed cadence, then reset.
        if time.time() - last_flush >= flush_seconds and window_total > 0:
            _put_fraud_rate(cloudwatch, cw_namespace, window_alerts, window_total)
            window_total = window_alerts = 0
            last_flush = time.time()

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
