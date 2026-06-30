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

### Results

Test split, same features for both models. Latency is single-transaction CPU inference.

| Model | Params | F1 | PR-AUC | Latency p50 | Latency p99 |
|---|---|---|---|---|---|
| **TCN (ours)** | **64,769** | **0.938** | **0.966** | 1.96 ms | 4.37 ms |
| Transformer (TranAD-like) | 399,105 | 0.807 | 0.851 | 1.62 ms | 3.22 ms |

> **Efficiency beats scale:** the TCN beats the Transformer on quality (**+0.12 PR-AUC, +0.13 F1**)
> with **6.2× fewer parameters**, and both models score in **under 5 ms p99** — far inside the 50 ms
> real-time budget. Scaling the baseline up does not buy accuracy here; the lean model wins outright.

<sub>A class-conditioned triple-PCA ablation was also run on both models; it did not improve the TCN
(PR-AUC 0.953) and only partly closed the Transformer's gap (0.883), so the raw-feature models above
are the headline.</sub>

### Key findings

- **The lean model wins outright.** The TCN does not merely *match* the heavier Transformer — it
  beats it on every quality metric (**PR-AUC 0.966 vs 0.851, F1 0.938 vs 0.807**) while using
  **6.2× fewer parameters**. This replays, in the financial domain, the efficiency result behind the
  author's FDI publication (TCN+HMM beating a TranAD Transformer at ~6× fewer params).
- **Scaling up the baseline bought nothing.** A 6× larger Transformer was *worse*, not better. For
  short, causally-ordered transaction sequences with strong engineered features, the right
  **inductive bias** (causal dilated convolutions) outperforms raw attention capacity — capacity is
  not the bottleneck, so spending parameters on it is wasted.
- **Latency is a non-issue for both.** Both models score a transaction in **under 5 ms p99 on CPU** —
  a ~10× margin under the 50 ms real-time budget, with **no GPU required for inference**. Efficiency
  here is therefore won on **parameter count and serving cost**, not on raw speed (the two latencies
  are comparable).
- **An honest negative result.** The triple-PCA feature ablation did *not* help the TCN — it already
  captures the discriminative structure on its own — and only partly helped the weaker Transformer.
  Reported rather than hidden, because the methodology, not a cherry-picked number, is the point.
- **Why this matters for production fraud.** A model that is 6× smaller is cheaper to serve, faster
  to retrain, and trivial to deploy on commodity CPU at high throughput — leaving ample headroom for
  stricter latency SLAs as transaction volume grows.

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

**Real-time scoring (the trained TCN, not a stub).** The sync API (`/predict`) and the stream
consumer share **one scoring core** ([scoring.py](src/lean_fraud/serve/scoring.py)) so they can
never drift apart. It takes a card's **raw** transaction history, rebuilds the exact training-time
features by reusing the ETL transforms (so there is **no train/serve skew**), standardizes with the
**train-fit scaler** from `meta.json`, runs the model, and decides with the **validation-tuned
threshold** saved by `evaluate` (not a hardcoded 0.5). The consumer keeps per-card history so the
causal rolling features reproduce training exactly; `latency_ms` in the `/predict` response is the
**measured** server-side inference time.

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

# 3. reproduce the results table: starts MLflow, then trains + evaluates + benchmarks
#    the full {tcn, transformer} x {raw, triple_pca} matrix and prints the table.
bash scripts/train_with_mlflow.sh                 # one command, browse runs at :5000

#    ...or drive a single cell by hand (train -> evaluate -> benchmark, all log to one MLflow run):
uv run python -m lean_fraud.train     --config configs/base.yaml   # train the lean model
uv run python -m lean_fraud.evaluate  --config configs/base.yaml   # test PR-AUC / F1 at val-tuned thr
uv run python -m lean_fraud.benchmark --config configs/base.yaml   # params + latency p50/p99

# 5. serve the scorer and run the real-time stream demo
uv run uvicorn lean_fraud.serve.api:app --host 0.0.0.0 --port 8000   # FastAPI on :8000
uv run python -m lean_fraud.streaming.producer   # replay transactions into Kinesis
uv run python -m lean_fraud.streaming.consumer   # score the stream and emit fraud alerts
```

Dev tasks: `uv run pytest -q` · `uv run ruff check src tests` · `uv run black src tests`.

Tear down the stack with `docker compose down -v`. Every entrypoint is a `python -m lean_fraud.<module>`
module, so it also runs without uv once the package is installed.

## Datasets (public)

Primary: **Sparkov** (`kartik2112/fraud-detection`) — ~1.85M synthetic credit-card transactions from
**999 cards**, ~0.5% fraud, with realistic per-card histories (median ~1,470 tx/card). Each row carries
a card number (`cc_num`, used as the per-user key), a `unix_time`, amount, merchant category,
cardholder + merchant geolocation, and an `is_fraud` label. Candidate alternatives: **IBM TabFormer**
(the dataset from the *Tabular Transformers* paper), **IEEE-CIS**.

**Access:** Sparkov is a public Kaggle *dataset* (not a competition), so it needs only a free Kaggle
API token — no competition-rules acceptance, no phone verification. Put `KAGGLE_API_TOKEN` in `.env`
(see [.env.example](.env.example)), then `uv sync --group data && uv run python -m lean_fraud.data.download`.
The two shipped CSVs (`fraudTrain.csv` + `fraudTest.csv`) are merged; build_sequences makes its **own**
strict time-based split. See [download.py](src/lean_fraud/data/download.py).

**Pipeline** ([build_sequences](src/lean_fraud/data/build_sequences.py)): per-card, causal feature
engineering — `amt` (+ log), inter-transaction `Δt`, causal rolling spend (mean/count of *prior*
transactions), cardholder↔merchant distance, and transaction hour / day-of-week; a few low-cardinality
categoricals (`category`, `gender`, `state`) integer-encoded and numeric features standardized — both
**fit on the train split only**. The output is a single time-sorted table
(`data/processed/sequences.npz` + `meta.json`) tagged per row with `train`/`val`/`test`; fixed-length
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
