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
   (producer)            ▲                        │  │                       │
                         │                        │  └► CloudWatch:          └──► LLM triage agent
                         │                        │     FraudAlertRate ─► alarm    (flagged ~0.5% only)
                         │                        └──► FastAPI /predict
   datasets / model artifacts  ◄──► [S3 (LocalStack)]     (sync scoring + UI demo)
      batch training orchestration: Airflow DAG   ·   experiment tracking: MLflow
```

**Real-time scoring (the trained TCN, not a stub).** The sync API (`/predict`) and the stream
consumer share **one scoring core** ([scoring.py](src/lean_fraud/serve/scoring.py)) so they can
never drift apart. It takes a card's **raw** transaction history, rebuilds the exact training-time
features by reusing the ETL transforms (so there is **no train/serve skew**), standardizes with the
**train-fit scaler** from `meta.json`, runs the model, and decides with the **validation-tuned
threshold** saved by `evaluate` (not a hardcoded 0.5). The consumer keeps per-card history so the
causal rolling features reproduce training exactly; `latency_ms` in the `/predict` response is the
**measured** server-side inference time.

**One-call assessment — `GET /assess/{card_id}`.** The endpoint that ties it together: pass a card id,
and the API looks the card up in the data, scores its full history with the TCN, then runs the LLM
**triage agent** and returns **both** — the model verdict *and* the agent's decision + rationale:

```jsonc
// GET /assess/2291163933867244
{
  "card_id": "2291163933867244", "transactions_considered": 2201,
  "fraud_probability": 0.0075, "is_fraud": false, "latency_ms": 2.37,
  "triage": { "action": "review",
              "rationale": "Amount is low vs. the card's recent activity and the 0.0033 base rate
                            for the (travel, SC) segment; recent spend skews to groceries/dining...",
              "confidence": null }
}
```

The rationale is generated by the local model (grounded in the agent's tool calls); if Ollama is
unreachable it degrades to the deterministic threshold decision, so `/assess` always returns.

**The full API** (FastAPI — browse/try it at [`/docs`](http://localhost:8000/docs)):

| Method & path | What it returns |
|---|---|
| `GET /health` | liveness + whether a model is loaded |
| `POST /predict` | score an explicit raw transaction sequence (stateless) |
| `GET /assess/{card_id}` | TCN verdict **+** LLM triage decision & rationale (above) |
| `GET /cards?limit=` | list of known card ids (`?limit<=0` → all 999) |
| `GET /cards/{card_id}/transactions?limit=` | a card's raw transaction history (most recent `limit`) |
| `GET /cards/{card_id}/stream?limit=&rate_hz=` | **NDJSON** stream: each of the card's tx scored live |
| `GET /stream?limit=&rate_hz=` | **NDJSON** firehose: all cards scored in global time order |

```bash
curl localhost:8000/cards?limit=5
curl "localhost:8000/cards/2291163933867244/transactions?limit=3"
curl -N "localhost:8000/cards/2291163933867244/stream?limit=20&rate_hz=5"   # -N = don't buffer
curl -N "localhost:8000/stream?limit=200"                                    # firehose, time-ordered
```

The stream endpoints keep per-card history so the causal features match training exactly (same core
as the Kinesis consumer); `rate_hz` paces the feed for a live terminal view, `limit` bounds it.

## Fraud-triage agent (LLM cascade)

Scoring flags *what* looks fraudulent; a triage layer explains *why* and decides *what to do*. A
**cheap TCN scores every transaction**, and only the flagged ~0.5% is escalated to an **LLM agent**
that investigates the card and returns a structured decision — a cascade that keeps the LLM cost
marginal while mirroring a production fraud stack (a lightweight detector feeding an LLM analyst
assistant).

- **A real ReAct agent, orchestrated with LangChain + LangGraph** ([graph.py](src/lean_fraud/agent/graph.py),
  `create_agent`). It is given three read-only tools over the processed data — `get_card_profile`,
  `get_recent_transactions`, `get_population_fraud_rate` — reasons over the alert, calls the tools it
  needs, and emits a `Decision` (`block | review | allow` + rationale).
- **Local, $0, offline by default.** The LLM backend is pluggable by config (`agent.provider`): a
  local **Ollama** model (`qwen2.5:3b`/`7b` via `langchain-ollama`) for real runs, or a deterministic
  **mock** used in tests/CI that never touches the network.
- **A guardrail that makes small local models safe to rely on.** The decision is resolved in three
  tiers: (1) native structured output; (2) a separate `with_structured_output` extraction — small
  models reason and decide well but often fail to emit the schema in the same turn, so this recovers
  the decision from their prose; and (3) a **deterministic threshold fallback** on the TCN score.
  `triage()` therefore **always** returns a valid decision and never hangs on the model, with the
  reason/act loop capped by LangGraph's `recursion_limit`.

Try it against a random held-out transaction (needs [Ollama](https://ollama.com) running):

```bash
uv run python scripts/agent_demo.py                             # default local model (qwen2.5:3b)
uv run python scripts/agent_demo.py --model qwen2.5:7b --fraud-only
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
uv run python -m lean_fraud.streaming.consumer   # score the stream, triage alerts, emit decisions

# 6. (optional) triage one held-out alert with the local LLM agent (needs Ollama running)
uv run python scripts/agent_demo.py --fraud-only
```

Dev tasks: `uv run pytest -q` · `uv run ruff check src tests` · `uv run black src tests`.

Tear down the stack with `docker compose down -v`. Every entrypoint is a `python -m lean_fraud.<module>`
module, so it also runs without uv once the package is installed.

### One-command demo

Once data is built and a model is trained, [`scripts/demo.sh`](scripts/demo.sh) drives the whole
real-time path end to end — it brings up the stack, provisions it with `tflocal`, then streams a
bounded batch of transactions through the TCN scorer and prints the fraud alerts + triage decisions
live:

```bash
bash scripts/demo.sh                 # ~2000 tx at 200 tx/s, ready to screen-record
```

<!-- TODO(demo gif): record scripts/demo.sh and drop the GIF here, e.g.:
     ![real-time fraud demo](docs/demo.gif) -->

### Fully containerized (Docker Compose)

Everything runs from one `uv`-built image ([`Dockerfile`](Dockerfile)) — the FastAPI scorer, the
producer, and the consumer are the same image with different commands — plus a local **Ollama** that
backs the triage agent. Compose **profiles** keep the streaming demo opt-in:

```bash
cp .env.example .env
docker compose up -d                # LocalStack + MLflow + FastAPI (:8000) + Ollama; 1st run blocks on the ~2GB model pull
curl localhost:8000/assess/2291163933867244   # TCN verdict + the agent's decision & rationale
docker compose --profile stream up  # + provision streams/bucket, then the producer -> consumer demo
docker compose down -v              # tear everything down
```

The trained model and data are **not baked into the image** (they are git-ignored, produced by
training); the services read them from the bind-mounted `./artifacts` + `./data`, so build the model
once on the host (steps 2–4 above) before starting — the scorer returns `503` until a model is
present. **`GET /assess/{card_id}`** is the headline: the TCN scores the card and the agent triages it
against the `ollama` service. The API `depends_on` the one-shot model pull (`service_completed_
successfully`), so the first `up` blocks until the model is ready and `/assess` returns a real LLM
rationale as soon as the stack is up — no manual "wait for the download" step; it still falls back to a
threshold decision if Ollama becomes unreachable at runtime.
The `stream` profile adds a one-shot `init` service that provisions the streams/bucket via the AWS
CLI, then streams ≈20k tx so per-card history builds up and alerts fire; the stream consumer keeps
triage **off** (`decision=n/a`) to stay fast under load — per-card LLM triage is `/assess`'s job.

> **Pinned LocalStack:** the compose uses `localstack/localstack:3.8` (Community). Newer `latest`/4.x
> images require a `LOCALSTACK_AUTH_TOKEN` at startup, which would break the "$0, no account" promise.

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

See `src/lean_fraud/` for the package (each module has a `python -m` entrypoint). Key folders: `src/` (code,
incl. `agent/` — the LangGraph triage agent), `infra/` (LocalStack init + Terraform IaC, incl. the
CloudWatch fraud-rate-spike alarm), `airflow/` (the batch training DAG that sequences the `python -m`
steps — see [airflow/README.md](airflow/README.md)), `configs/` (experiments), `scripts/` (demos),
`tests/`. What runs today vs. what is a documented future direction (SQS + Postgres) is spelled out in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Author

Alejandro Torres — AI/ML Engineer focused on time-series anomaly detection and model efficiency,
including work on lightweight temporal models (TCN+HMM) that match a heavy Transformer (TranAD)
with ~6× fewer parameters.

## License

MIT
