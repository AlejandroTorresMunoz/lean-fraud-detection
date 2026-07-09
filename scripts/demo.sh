#!/usr/bin/env bash
#
# End-to-end real-time demo — ready to screen-record for the README GIF.
#
# Brings up the emulated AWS stack (LocalStack: Kinesis + S3 + CloudWatch) + MLflow, provisions it
# with the real Terraform via tflocal, then streams a bounded batch of transactions through the TCN
# scorer and prints the fraud alerts + LLM triage decisions live.
#
# Prerequisites (fail fast if missing):
#   - Docker running, and `.env` created (`cp .env.example .env`).
#   - Data built + a model trained, so the scorer has a checkpoint:
#       uv run python -m lean_fraud.data.download
#       uv run python -m lean_fraud.data.build_sequences
#       uv run python -m lean_fraud.train    --config configs/base.yaml
#       uv run python -m lean_fraud.evaluate --config configs/base.yaml
#
# Run from the repo root:
#   bash scripts/demo.sh                 # ~2000 tx at 200 tx/s
#   TX_LIMIT=5000 RATE_HZ=300 bash scripts/demo.sh
#
set -euo pipefail

# Stream enough tx that per-card history builds up and fraud alerts actually fire (the first few
# thousand time-sorted tx are history-sparse and rarely cross the threshold).
TX_LIMIT="${TX_LIMIT:-20000}"
RATE_HZ="${RATE_HZ:-1000}"

[ -f .env ] || { echo "ERROR: no .env — run 'cp .env.example .env' first."; exit 1; }

echo "[demo] 1/4 bringing up the emulated AWS stack (LocalStack + MLflow + API)..."
docker compose up -d localstack mlflow api

echo "[demo] 2/4 provisioning Kinesis + S3 + CloudWatch alarm via tflocal..."
uv run bash infra/init_localstack.sh

echo "[demo] 3/4 starting the stream consumer (TCN scoring + triage) in the background..."
uv run python -m lean_fraud.streaming.consumer &
CONSUMER_PID=$!
# Always stop the consumer on exit (normal, error, or Ctrl-C).
trap 'echo "[demo] stopping consumer (pid ${CONSUMER_PID})"; kill "${CONSUMER_PID}" 2>/dev/null || true' EXIT
sleep 3   # let it attach to the shards (ShardIteratorType=LATEST) before we produce

echo "[demo] 4/4 replaying ${TX_LIMIT} transactions at ~${RATE_HZ} tx/s..."
uv run python -m lean_fraud.streaming.producer --limit "${TX_LIMIT}" --rate-hz "${RATE_HZ}"

sleep 3   # let the consumer drain the last records
echo "[demo] done. Tear down with: docker compose down -v"
