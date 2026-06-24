#!/usr/bin/env bash
#
# Launch the full experiment matrix: {tcn, transformer} x {raw, triple_pca} = 4 trainings.
#
# For each feature variant it builds the processed dataset if missing (raw -> data/processed,
# triple_pca -> data/processed_pca), then trains + evaluates + benchmarks every cell, logging to
# MLflow and printing the README results table at the end. Uses the GPU automatically when
# available (device: auto in the config).
#
# Run from the repo root:
#   bash scripts/run_experiments.sh                     # all 4 cells, configs/base.yaml
#   bash scripts/run_experiments.sh configs/base.yaml   # explicit config
#
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"

echo "[run_experiments] config=${CONFIG}"
echo "[run_experiments] launching 4 cells: {tcn, transformer} x {raw, triple_pca}"
echo "[run_experiments] watch progress live in MLflow -> http://127.0.0.1:5000"
echo

# Force unbuffered output: uv pipes the child's stdout (not a TTY), which otherwise block-buffers
# the per-batch heartbeats and makes a running epoch look frozen. Without this you can't see
# progress live and it's tempting to kill a training that is actually fine.
export PYTHONUNBUFFERED=1
uv run python -u -m lean_fraud.experiments \
  --config "${CONFIG}" \
  --models tcn transformer \
  --features raw triple_pca

echo
echo "[run_experiments] done — results table printed above; all 4 runs are in MLflow."
