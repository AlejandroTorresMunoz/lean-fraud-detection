# Efficiency Beats Scale: Real-Time Transaction Fraud Detection

> A reproducible, **real-time** fraud-detection system over streams of transactions that shows a
> **lightweight temporal encoder (TCN) can match or beat a heavy Transformer** at fraud detection —
> with **~5–10× fewer parameters** and **sub-50 ms inference latency**. Packaged with production-grade
> MLOps over a **free, fully local AWS-emulated stack** (Kinesis + S3 + Airflow + FastAPI on
> LocalStack), where the **Terraform actually applies** via `tflocal`.

[![CI](https://github.com/AlejandroTorresMunoz/lean-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/AlejandroTorresMunoz/lean-fraud-detection/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)

---

## TL;DR — the thesis

In real-time fraud detection, **latency and inference cost are first-class constraints**, not
afterthoughts. This project benchmarks a lean temporal model against a heavyweight Transformer
baseline and reports **quality _and_ efficiency side by side**.

### Results (to be filled after Phase 4)

| Model | Params | F1 | PR-AUC | Latency p50 | Latency p99 |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| **TCN (ours)** | — | — | — | — | — |
| Transformer (TranAD-like) | — | — | — | — | — |

> Headline to prove: comparable/better F1 than the Transformer with far fewer params and lower p99.

---

## Architecture

Real-time scoring pipeline on an **emulated AWS** stack (runs locally on LocalStack, **no AWS account, $0**):

```
 transactions ──► [Kinesis: tx-stream] ──► consumer (TCN scoring) ──► [Kinesis: alerts-stream]
   (producer)            ▲                        │
                         │                        └──► FastAPI /predict  (sync scoring + UI demo)
   datasets / model artifacts  ◄──► [S3 (LocalStack)]
                         orchestration: Airflow DAG   ·   experiment tracking: MLflow
```

## Quickstart

Tooling: [**uv**](https://docs.astral.sh/uv/) for the Python env, **Docker** for the local AWS
stack, and the **Terraform** CLI (used by `tflocal`).

```bash
# 0. set up the virtualenv and install dependencies (creates .venv + uv.lock)
uv sync                       # project + default `dev` group
uv sync --extra demo          # add the Streamlit demo UI
uv sync --group infra         # add tflocal/awslocal (provision LocalStack)
uv sync --group data          # add the Kaggle API (dataset download)

# 1. spin up the emulated AWS stack (LocalStack: Kinesis + S3) + MLflow + API
cp .env.example .env
docker compose up -d
uv run bash infra/init_localstack.sh   # tflocal apply -> really provisions the streams + bucket

# 2. download a public dataset and build transaction sequences
uv run python -m lean_fraud.data.download
uv run python -m lean_fraud.data.build_sequences

# 3. train the lean model (logs to MLflow)
uv run python -m lean_fraud.train --config configs/base.yaml

# 4. evaluate + efficiency benchmark (quality, params, latency p50/p99)
uv run python -m lean_fraud.evaluate --config configs/base.yaml
uv run python -m lean_fraud.benchmark --config configs/base.yaml

# 5. serve the scorer and run the real-time stream demo
uv run uvicorn lean_fraud.serve.api:app --host 0.0.0.0 --port 8000   # FastAPI on :8000
uv run python -m lean_fraud.streaming.producer   # replay transactions into Kinesis
uv run python -m lean_fraud.streaming.consumer   # score the stream and emit fraud alerts
```

Dev tasks: `uv run pytest -q` · `uv run ruff check src tests` · `uv run black src tests`.

Tear down the stack with `docker compose down -v`. Every entrypoint is a `python -m lean_fraud.<module>`
module, so it also runs without uv once the package is installed.

## Datasets (public)

Primary: **IEEE-CIS Fraud Detection** (~590K labelled transactions, ~3.5% fraud, rich anonymized
features). It has no explicit user id, so [build_sequences](src/lean_fraud/data/build_sequences.py)
derives a pseudo-user from `card1 + addr1 + P_emaildomain` and orders by `TransactionDT`. Candidate
alternatives: **IBM TabFormer** (true per-user sequences), **PaySim**.

**Access:** needs a free Kaggle token (`~/.kaggle/kaggle.json`) and a one-time acceptance of the
[competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules) — still possible although the
2019 contest is closed. Then `uv sync --group data && uv run python -m lean_fraud.data.download`.
Only the labelled train files are used; build_sequences makes its own strict time-based split (the
competition test set has no public labels). See [src/lean_fraud/data/download.py](src/lean_fraud/data/download.py).

**Pipeline** ([build_sequences](src/lean_fraud/data/build_sequences.py)): per-user, causal feature
engineering — `amount` (+ log), inter-transaction `Δt`, causal rolling spend (mean/count of *prior*
transactions), and the anonymized `C1–14` / `D1–15` blocks; a few low-cardinality categoricals
(`ProductCD`, `card4`, `card6`, `DeviceType`) integer-encoded and numeric features standardized — both
**fit on the train split only**. The output is a single time-sorted table
(`data/processed/ieee_cis.npz` + `meta.json`) tagged per row with `train`/`val`/`test`; fixed-length
sequence windows are built lazily per batch with `make_windows`, avoiding a multi-GB 3-D array.

## Design decisions

- **Why a TCN?** Causal dilated convolutions capture long transaction histories with a small,
  fast, parallelizable model — ideal for low-latency real-time scoring.
- **Imbalanced classes:** focal loss / class weighting; we report **PR-AUC** (not just ROC-AUC).
- **No temporal leakage:** a strict time-based split by `TransactionDT` (no future in train); causal
  windows plus train-only fitting of encoders/scalers keep every feature strictly backward-looking.
- **Efficiency is measured, not assumed:** params, model size, and p50/p99 latency are reported.

## On the "deployment" (honesty matters)

- The cloud here is **emulated locally with [LocalStack](https://localstack.cloud/)** (Kinesis + S3).
  It is a faithful **local environment**, **not** a real AWS deployment — and it is labeled as such
  everywhere.
- **The Terraform actually applies.** LocalStack emulates the AWS **control plane** (not just the data
  plane), so the [Terraform under `infra/terraform/`](infra/) is provisioned for real against LocalStack
  via [`tflocal`](https://github.com/localstack/terraform-local) (`init_localstack.sh` runs `tflocal apply`).
  The exact same `*.tf` would `terraform apply` to real AWS — it is **production-grade IaC, not a static
  artifact**. It is **not** applied in CI (to avoid any cloud cost).
- **Reproducible at $0.** The whole stack runs from `docker compose up` + `tflocal apply` with dummy
  credentials — no cloud account, no billing — so the IaC and the pipeline are **genuinely runnable**,
  not just described: a stronger, more honest demo than a deploy that can't actually run.

## Project layout

See `src/lean_fraud/` for the package (each module has a `python -m` entrypoint). Key folders: `src/` (code),
`infra/` (LocalStack init + Terraform IaC), `airflow/` (DAG), `configs/` (experiments), `tests/`.

## Author

Alejandro Torres — AI/ML Engineer focused on time-series anomaly detection and model efficiency,
including work on lightweight temporal models (TCN+HMM) that match a heavy Transformer (TranAD)
with ~6× fewer parameters.

## License

MIT
