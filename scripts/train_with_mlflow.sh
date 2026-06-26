#!/usr/bin/env bash
#
# One command: bring up MLflow, then run the full experiment matrix logging into it.
#
# Starts the MLflow tracking server (the single owner of mlflow.db, so the UI can stay open while
# training — see configs/base.yaml), waits until it is reachable, then delegates to
# scripts/run_experiments.sh. If MLflow is already running on the port it is reused, not restarted.
# The server is LEFT RUNNING after training so you can browse the runs; the script prints how to
# stop it.
#
# Run from the repo root:
#   bash scripts/train_with_mlflow.sh                     # all 4 cells, configs/base.yaml
#   bash scripts/train_with_mlflow.sh configs/base.yaml   # explicit config
#   MLFLOW_PORT=5001 bash scripts/train_with_mlflow.sh    # override port (match tracking_uri!)
#
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"
MLFLOW_HOST="127.0.0.1"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_URL="http://${MLFLOW_HOST}:${MLFLOW_PORT}"
BACKEND="sqlite:///mlflow.db"

is_up() { curl -sf "${MLFLOW_URL}/health" >/dev/null 2>&1; }

started_here=0
if is_up; then
  echo "[train_all] MLflow already up at ${MLFLOW_URL} — reusing it."
else
  echo "[train_all] starting MLflow at ${MLFLOW_URL} (backend ${BACKEND})..."
  mkdir -p .logs
  # nohup so it survives this script exiting -> the UI stays browsable after training.
  nohup uv run mlflow ui \
    --backend-store-uri "${BACKEND}" \
    --host "${MLFLOW_HOST}" --port "${MLFLOW_PORT}" \
    > .logs/mlflow.log 2>&1 &
  MLFLOW_PID=$!
  started_here=1

  echo -n "[train_all] waiting for MLflow to come up"
  for _ in $(seq 1 60); do
    if is_up; then echo " — ready (pid ${MLFLOW_PID})."; break; fi
    if ! kill -0 "${MLFLOW_PID}" 2>/dev/null; then
      echo; echo "[train_all] MLflow exited during startup. Last log lines:"
      tail -n 20 .logs/mlflow.log; exit 1
    fi
    echo -n "."; sleep 1
  done
  if ! is_up; then
    echo; echo "[train_all] MLflow not reachable after 60s — see .logs/mlflow.log"; exit 1
  fi
fi

echo
echo "[train_all] UI: ${MLFLOW_URL}   (training logs into this server)"
echo
bash scripts/run_experiments.sh "${CONFIG}"

echo
if [ "${started_here}" -eq 1 ]; then
  echo "[train_all] done. MLflow still running at ${MLFLOW_URL} (pid ${MLFLOW_PID})."
  echo "[train_all] stop it with:  kill ${MLFLOW_PID}"
else
  echo "[train_all] done. MLflow (started elsewhere) left running at ${MLFLOW_URL}."
fi
