# Architecture & Design

Design record for **lean-fraud-detection**. Captures the agreed approach, the rationale behind the
key choices, and — kept scrupulously honest — **what is implemented vs. planned**. The thesis: a lean
**TCN** matches/beats a heavy **Transformer** at transaction fraud detection with ~5–10× fewer params
and **sub-50 ms** scoring, wrapped in production-style MLOps on a **free, fully local, AWS-emulated**
stack (LocalStack).

> **Read this first — implemented vs. planned.** The system that runs today is a **Kinesis**
> streaming pipeline (producer → consumer → alerts) + a synchronous **FastAPI** scorer + an **Airflow**
> batch training DAG + **MLflow** tracking + a **CloudWatch** alarm, all on **LocalStack**. A
> larger **SQS + Postgres** batch-inference topology is **sketched below as a future direction and is
> not built** — it is labelled as such everywhere so nothing here oversells the repo.

## Design decisions (and why)

| Decision | Choice | Status | Why |
|---|---|---|---|
| Cloud | **Emulated AWS via LocalStack** (not GCP) | ✅ implemented | LocalStack emulates the AWS **control plane**, so the Terraform actually applies (`tflocal`). Maps 1:1 to a real Revolut-style stack; matches the author's AWS/Terraform experience. |
| Streaming bus | **Amazon Kinesis** (`tx-stream` + `alerts-stream`) | ✅ implemented | Real-time producer → consumer (TCN scoring) → alerts, replayed from the raw Sparkov CSVs. Kinesis ≈ Pub/Sub in Revolut's GCP stack. |
| Low-latency path | **FastAPI `/predict`** (synchronous) | ✅ implemented | The <50 ms thesis lives in the sync scorer, sharing one `scoring.py` core with the consumer (no train/serve skew). |
| Fraud triage | **LangGraph ReAct agent** on flagged alerts | ✅ implemented | Cascade: the cheap TCN scores all traffic, an LLM agent investigates only the flagged ~0.5%. Local/`$0` (Ollama) or `mock` in CI. |
| Experiment mgmt | **MLflow tracking** | ✅ implemented | Compare TCN vs Transformer vs baselines on quality **and** efficiency; one run per experiment cell. Model Registry + auto-promotion remain planned (see below). |
| Orchestration | **Airflow** — one batch *training* DAG | ✅ implemented | `lean_fraud_pipeline` sequences download → build → train → evaluate → benchmark via the `python -m` entrypoints. Batch only; real-time scoring is a service, **not** on Airflow. |
| Monitoring | **CloudWatch** `FraudAlertRate` alarm | ✅ implemented | The consumer emits a windowed fraud-alert rate; a `tflocal`-provisioned alarm fires on a spike (attack/drift) — the only signal actionable without live labels. |
| Queue (future) | **Amazon SQS** for async batch inference | 🔭 future, not built | A "poll N → process → delete" batch flow maps cleanly to SQS (visibility timeout + DLQ = free retries). Would sit **alongside** Kinesis, not replace it. |
| History store (future) | **PostgreSQL** for a prediction-audit table | 🔭 future, not built | Real SQL for prediction history / dashboards. RDS is a LocalStack **Pro** feature, so on the free tier Postgres would run as its own container; `aws_db_instance` would stay an IaC artifact. |
| Registry (future) | **MLflow Model Registry + auto-promotion** | 🔭 future, not built | Register the best model and promote to `Production` on PR-AUC; serving would then load from the registry instead of local artifacts. |

## System diagram — what runs today

```
PHASE 1 — Data & Model (orchestrated by the Airflow batch DAG, or `python -m ...` by hand)
  Kaggle ─► download ─► [EDA notebook] ─► build_sequences ─► train · evaluate · benchmark
                                                                  └──► MLflow (tracking)

REAL-TIME — Scoring (services, on Kinesis; NOT Airflow)
  producer ─► [Kinesis: tx-stream] ─► consumer (TCN scoring) ─► [Kinesis: alerts-stream] ─► LLM triage agent
                                          │                                                  (flagged ~0.5% only)
                                          ├─► FastAPI /predict   (synchronous, <50 ms, same model + UI)
                                          └─► CloudWatch: FraudAlertRate ─► alarm (spike = attack/drift)
  model artifacts / datasets  ◄──►  [S3 (LocalStack)]
```

Local stack (`docker-compose`): **LocalStack (Kinesis + S3 + CloudWatch)** · **MLflow** · **FastAPI** ·
(optional profile) **Airflow**. All free, no AWS account.

## Phase 1 — Data & Model

Plain `python -m lean_fraud.*` steps (also wired as the Airflow `lean_fraud_pipeline` DAG):

1. **Download** ([data/download.py](../src/lean_fraud/data/download.py)) — the Sparkov CSVs via the
   Kaggle API (a public dataset, so just an API token — no competition rules) → `data/raw/`.
2. **EDA** — `notebooks/eda_sparkov.ipynb`: class imbalance, transactions-per-card distribution
   (justifies `sequence_length=32`), feature distributions.
3. **Build sequences** ([data/build_sequences.py](../src/lean_fraud/data/build_sequences.py)) — the
   modular ETL (see "Data pipeline" below).
4. **Train / evaluate / benchmark** — `SequenceDataset` (lazy windows via `make_windows`) + baselines
   + the Transformer baseline, all logging to MLflow.

## Orchestration — the Airflow batch DAG (implemented)

One DAG, [`airflow/dags/lean_fraud_pipeline.py`](../airflow/dags/lean_fraud_pipeline.py):

```
download ─► build_sequences ─► train ─► evaluate ─► benchmark
```

Each task is a `BashOperator` shelling out to a `python -m lean_fraud.<module>` entrypoint, so Airflow
only orchestrates (ordering, retries, UI) and all logic stays in the package. `schedule=None` (manual
trigger). Run it in the project venv or via the optional `docker compose --profile airflow up` — see
[airflow/README.md](../airflow/README.md). **Real-time scoring is deliberately not on Airflow** (it is
the long-running FastAPI service + stream consumer).

## Monitoring — CloudWatch fraud-rate-spike alarm (implemented)

With no ground-truth labels at scoring time, F1/PR-AUC can't be monitored live; latency is a settled
non-issue (~10× headroom). The one actionable live signal is the **alert rate**: the consumer emits a
windowed `FraudAlertRate` custom metric (`put_metric_data`, `LeanFraud` namespace), and a
`tflocal`-provisioned `aws_cloudwatch_metric_alarm` fires when it spikes above ~10× the base rate —
i.e. an attack or data drift. The metric push is **best-effort**: on the pinned LocalStack Community
image, CloudWatch metric ingestion and automatic alarm-state evaluation are limited (a modern
boto3 ↔ community wire-protocol skew can make `put_metric_data` a no-op, and you may need
`set-alarm-state` to see it fire) — the consumer swallows any such error and keeps scoring. The alarm
*definition* is the production-grade artifact and the same HCL alarms for real on AWS.

## Future direction — async batch inference (SQS + Postgres, NOT built)

Sketched for completeness; **none of this is implemented**. It would sit alongside the Kinesis
real-time path, adding Airflow + SQS + Postgres to `docker-compose`:

| DAG (future) | Purpose | Reads | Writes |
|---|---|---|---|
| **A — `feed_test_to_sqs`** | Replay the test split as incoming events | `data/processed` (S3) | `tx-queue` (SQS) |
| **B — `consume_and_infer`** | Score queued transactions with the registered model | `tx-queue`, MLflow | `pred-queue` (SQS) |
| **C — `collect_history`** | Aggregate model outputs into an audit history | `pred-queue` | `predictions_history` (Postgres) |

An SQS message is delivered to **one** consumer, so the real-time (FastAPI/Kinesis) and batch (SQS)
paths would use separate flows and never compete for the same messages.

## MLflow usage

- **Tracking** (implemented; server in compose): each run logs **params** (from `configs/*.yaml`),
  **metrics** (F1, PR-AUC, precision, recall **+** param count, model size, p50/p99 latency), and
  **artifacts** (model weights, the `meta.json` scaler + feature order so serving reproduces
  preprocessing).
- **Serving** today loads the trained model from **local artifacts** (`artifacts/<run>/best.pt` plus
  the `meta.json` scaler and the val-tuned threshold) through the shared
  [scoring.py](../src/lean_fraud/serve/scoring.py) core.
- **Model Registry + auto-promotion** (future): register the best model, promote to `Production` on
  PR-AUC, and switch serving to load from the registry — planned, not built.

## Data pipeline (Sparkov)

> **Full walkthrough:** [DATA_PIPELINE.md](DATA_PIPELINE.md). The pipeline is a modular ETL — a thin
> orchestrator (`build_sequences`) composing focused Transform stages under `data/transform/`
> (`features` · `split` · `encode` · `pca`), each unit-testable.

The card number (`cc_num`) is the per-user key; transactions are ordered by `unix_time`. Causal
features (`amt`+log, inter-tx `Δt`, causal rolling spend, cardholder↔merchant distance, hour /
day-of-week), a few low-cardinality categoricals (`category`, `gender`, `state`) integer-encoded —
**encoders/scaler fit on train only**. The two shipped CSVs (`fraudTrain`/`fraudTest`) are merged and
re-split with our **own** strict time-based split. Output is one time-sorted table
(`data/processed/sequences.npz` + `meta.json`) tagged per row `train/val/test`; fixed-length windows
are built lazily with `make_windows` (no multi-GB 3-D array). Validated end-to-end: ~1.85M tx, 999
cards, fraud ~0.5%.

## Honesty notes

- The cloud is **emulated locally** (LocalStack); not a real AWS deployment, labelled as such.
- The Terraform under `infra/terraform/` **actually applies** to LocalStack via `tflocal` (streams,
  bucket, CloudWatch alarm + SNS topic); the same `*.tf` would target real AWS. Not run in CI (zero
  cost).
- The SQS + Postgres + multi-DAG inference topology is a **documented future direction, not code**.
- **RDS is LocalStack Pro**; if the Postgres path is built, on the free tier it would be a plain
  container and `aws_db_instance` would remain IaC documentation.

## Status

| Area | State |
|---|---|
| Data ETL: download + modular `transform/` + `build_sequences` (Sparkov) | ✅ implemented |
| Config validation tests | ✅ implemented |
| Pre-commit (ruff/black/file hooks) + CI (uv) | ✅ implemented |
| Pipeline validated on real data | ✅ validated (~1.85M tx, 999 cards) |
| EDA notebook (`notebooks/eda_sparkov.ipynb`) | ✅ implemented |
| Train / evaluate / benchmark + MLflow tracking | ✅ implemented (TCN + Transformer, triple-PCA ablation) |
| Real-time serving: FastAPI + Kinesis consumer score the trained TCN | ✅ implemented (shared `scoring.py`, val-tuned threshold) |
| Fraud-triage LLM agent (LangGraph cascade) | ✅ implemented (Ollama / mock backends) |
| Airflow batch **training** DAG (download→build→train→eval→benchmark) | ✅ implemented |
| CloudWatch `FraudAlertRate` metric + spike alarm | ✅ implemented (metric in consumer, alarm in `tflocal`) |
| Containerization: `uv` image + api/producer/consumer compose services | ✅ implemented (one image, `stream` profile, bind-mounted model/data) |
| Async batch inference: SQS queues + Postgres history + DAGs A/B/C | 🔭 future (documented, not built) |
| MLflow Model Registry + auto-promotion + serve-from-registry | 🔭 future (documented, not built) |
