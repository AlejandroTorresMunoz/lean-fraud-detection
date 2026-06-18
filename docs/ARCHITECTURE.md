# Architecture & Design

Design record for **lean-fraud-detection**. Captures the agreed approach, the rationale behind the
key choices, and what is implemented vs. planned. The thesis: a lean **TCN** matches/beats a heavy
**Transformer** at transaction fraud detection with ~5–10× fewer params and **sub-50 ms** scoring,
wrapped in production-style MLOps on a **free, fully local, AWS-emulated** stack (LocalStack).

## Design decisions (and why)

| Decision | Choice | Why |
|---|---|---|
| Cloud | **Emulated AWS via LocalStack** (not GCP) | LocalStack emulates the AWS **control plane**, so the Terraform actually applies (`tflocal`). Maps 1:1 to a real Revolut-style stack; matches the author's AWS/Terraform experience. |
| Queue | **Amazon SQS** (Kinesis retired) | The async inference pipeline is orchestrated by Airflow with "poll N → process → delete" semantics, which is exactly SQS (visibility timeout + DLQ = free retries). |
| History store | **PostgreSQL (container)** | Real SQL for the prediction history / dashboards. RDS is a LocalStack **Pro** feature, so on the free tier Postgres runs as its own container; an `aws_db_instance` stays as IaC artifact. |
| Experiment mgmt | **MLflow: tracking + Model Registry + auto-promotion** | Compare TCN vs Transformer vs baselines (quality **and** efficiency) and promote the best to `Production` automatically on PR-AUC. |
| Orchestration | **Airflow** (added in Phase 2) | Batch only (ingestion is manual scripts in Phase 1; real-time scoring is a service). Airflow is not used for long-running stream consumers. |
| Low-latency path | **FastAPI `/predict`** (synchronous) | The <50 ms thesis lives in the sync scorer; SQS handles the async/batch flow. |

> **Narrative note:** with SQS (a queue, not a stream) the wording shifts from "streams of
> transactions" to "a queue of transaction events". The latency/efficiency thesis is unchanged — it
> is demonstrated by the FastAPI scorer and the efficiency benchmark.

## System diagram

```
PHASE 1 — Data & Model (no Airflow; run as `python -m ...`)
  Kaggle ─► download ─► [EDA notebook] ─► build_sequences ─► train · evaluate · benchmark
                                                                  └──► MLflow (tracking + registry + auto-promote)
                                                                             │ "Production" model
PHASE 2 — Inference (Airflow + SQS + Postgres)                              ▼
  DAG A  feed_test_to_sqs   :  test split (S3) ─────────────────► [SQS: tx-queue]
  DAG B  consume_and_infer  :  [SQS: tx-queue] ─► load model (MLflow) ─► infer ─► [SQS: pred-queue]
  DAG C  collect_history    :  [SQS: pred-queue] ─► upsert ─► [Postgres: predictions_history]

  Real-time demo (service, NOT Airflow):  FastAPI /predict  (synchronous, <50 ms, same model)
```

Local stack (`docker-compose`): **LocalStack (SQS + S3)** · **MLflow** · **PostgreSQL** ·
(Phase 2) **Airflow** · **FastAPI**. All free, no AWS account.

## Phase 1 — Data & Model (no Airflow)

Plain `python -m lean_fraud.*` steps, two data commits then modelling:

1. **Download** ([data/download.py](../src/lean_fraud/data/download.py)) — IEEE-CIS labelled train
   files via the Kaggle API → `data/raw/`.
2. **EDA** — `notebooks/eda_ieee_cis.ipynb`: class imbalance, transactions-per-pseudo-user
   distribution (justifies `sequence_length=32`), missingness. Summary into the README.
3. **Build sequences** ([data/build_sequences.py](../src/lean_fraud/data/build_sequences.py)) —
   already implemented (see "Data pipeline" below).
4. **Train / evaluate / benchmark** — implement the stubs + a `SequenceDataset` (lazy windows via
   `make_windows`) + baselines (logreg/XGBoost) + the Transformer baseline, all logging to MLflow.

## Phase 2 — Inference (Airflow + SQS + Postgres)

Airflow is added to `docker-compose` (LocalExecutor). Three batch DAGs:

| DAG | Purpose | Reads | Writes |
|---|---|---|---|
| **A — `feed_test_to_sqs`** | Replay the test split as incoming events | `data/processed` (S3) | `tx-queue` (SQS) |
| **B — `consume_and_infer`** | Score queued transactions with the registered model | `tx-queue`, MLflow (Production model) | `pred-queue` (SQS) |
| **C — `collect_history`** | Aggregate model outputs into the audit history | `pred-queue` | `predictions_history` (Postgres) |

DAG tasks call the same `python -m lean_fraud.*` entrypoints (orchestration only; logic stays in the
package). Optional: chain DAGs with **Airflow Datasets** (data-aware scheduling).

## MLflow usage

- **Tracking** (server already in compose): each run logs **params** (from `configs/*.yaml`),
  **metrics** (F1, PR-AUC, precision, recall **+** param count, model size, p50/p99 latency), and
  **artifacts** (model weights, PR curve / confusion matrix, the `meta.json` scaler + feature order so
  serving reproduces preprocessing).
- **Model Registry**: best model registered as `lean-fraud-tcn` with `Staging`/`Production` stages.
- **Auto-promotion**: the training code promotes a new model to `Production` only if it beats the
  current champion on PR-AUC (MLflow Registry API — no Airflow needed).
- **Serving** (FastAPI + the inference DAGs) loads the `Production` model from the registry, replacing
  the current toy heuristic in [serve/api.py](../src/lean_fraud/serve/api.py).
- *(Option)* point MLflow's artifact store at the LocalStack S3 bucket for a fully AWS-emulated story.

## Data pipeline (IEEE-CIS)

No explicit user id → derive a **pseudo-user** (`card1 + addr1 + P_emaildomain`), order by
`TransactionDT`. Causal features (`amount`+log, inter-tx `Δt`, causal rolling spend, `C1–14`/`D1–15`),
a few low-cardinality categoricals integer-encoded — **encoders/scaler fit on train only**. Output is
one time-sorted table (`data/processed/ieee_cis.npz` + `meta.json`) tagged per row `train/val/test`
via a **strict time-based split**; fixed-length windows are built lazily with `make_windows` (no
multi-GB 3-D array). Only the labelled train files are used (the competition test set is unlabelled).

## SQS implication: real-time vs. batch

An SQS message is delivered to **one** consumer. So the real-time path (FastAPI) and the Airflow batch
path use **separate flows/queues** and never compete for the same messages.

## Honesty notes

- The cloud is **emulated locally** (LocalStack); not a real AWS deployment, labelled as such.
- The Terraform under `infra/terraform/` **actually applies** to LocalStack via `tflocal`; the same
  `*.tf` would target real AWS. Not run in CI (zero cost).
- **RDS is LocalStack Pro**; on the free tier Postgres is a plain container, and `aws_db_instance`
  remains IaC documentation rather than a `tflocal`-provisioned resource.

## Status

| Area | State |
|---|---|
| Data download + sequence build (IEEE-CIS) | ✅ implemented |
| Config validation tests | ✅ implemented |
| Pre-commit (ruff/black/file hooks) + CI (uv) | ✅ implemented |
| Pipeline validated on real data | ⏳ pending (needs Kaggle token) |
| EDA notebook | ⏳ pending |
| Train / evaluate / benchmark + MLflow | ⏳ stubs |
| Queue migration **Kinesis → SQS** | ⏳ pending (Phase 2; infra, streaming, `.env`, README) |
| Postgres + Airflow in `docker-compose` | ⏳ pending (Phase 2) |
| Airflow DAGs A/B/C | ⏳ pending (Phase 2; replaces the single skeleton DAG) |
| FastAPI / consumer load real model from MLflow | ⏳ pending |
