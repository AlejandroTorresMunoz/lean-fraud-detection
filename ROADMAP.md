# Roadmap — lean-fraud-detection

Remaining work to complete the project, organized by branch/PR to keep changes small and reviewable.

State as of the `modelling` merge (PR #2, commit `a9b1300`).

---

## ✅ Done

- Scaffolding + emulated AWS infra (LocalStack + Terraform applied via `tflocal`), CI, docker-compose, tests.
- Full, validated data pipeline (Sparkov → causal features → strict time-based split, no leakage).
- **Modelling (core):** TCN + Transformer, train/evaluate/benchmark, experiment matrix,
  triple-PCA ablation, MLflow tracking grouped into a single run per cell.
- **README:** results table + conclusions (Key findings) + Quickstart.
- **PR #2 `modelling → main` merged.**

### Reference results (test split)

| Model | Params | F1 | PR-AUC | p50 | p99 |
|---|---|---|---|---|---|
| **TCN (ours)** | 64,769 | 0.938 | 0.966 | 1.96 ms | 4.37 ms |
| Transformer | 399,105 | 0.807 | 0.851 | 1.62 ms | 3.22 ms |

The TCN wins on quality with 6.2× fewer parameters. Both score under 5 ms p99.

---

## ✅ PR #3 — Real serving (branch `serving`) — merged

**Goal:** kill the toy scorer and serve the real TCN end to end. No dependencies (`best.pt` already exists).

- [x] Load the model at startup in `serve/api.py` via `load_checkpoint()` (startup, not per request).
- [x] Real `/predict`: build the sequence window from the payload, apply the **scaler from `meta.json`**
      (same normalization as training), run inference, return prob + real `latency_ms`.
- [x] Handle short input: pad when fewer than `seq_len` transactions arrive.
- [x] Same scorer in `streaming/consumer.py`: extract the scoring logic into a shared module
      (`scoring.py`) to avoid duplication.
- [x] Decision threshold: load the val-tuned threshold (saved by evaluate) instead of a hardcoded 0.5.
- [x] Tests: smoke test for `/predict` with the real model; test the consumer scoring a message.
- [x] Update README (drop any "toy" wording; `latency_ms` is now real).

---

## ✅ PR #4 — Fraud-triage agent (branch `agent`) — Phase 2 differentiator

**Goal:** a **local, $0** LLM layer on top of the alerts. **Cascade:** the cheap TCN scores all
traffic → the agent runs only on the flagged ~0.5%. Verified end to end against real Ollama
(`qwen2.5:3b` and `7b`) and in CI with the mock backend.

- [x] Design: input `AlertContext` (tx + score + card id) → output `Decision` (block/review/allow + rationale).
- [x] **Orchestration on LangChain + LangGraph** (`create_agent`, `recursion_limit`); no hand-rolled loop.
- [x] Pluggable backend by config (`agent.provider: ollama | mock`):
  - [x] `ollama` (default): local model via `langchain-ollama` `ChatOllama`; default `qwen2.5:3b`,
        `qwen2.5:7b` for stronger tool-calling.
  - [x] `mock`: deterministic LangChain fake chat model for tests/CI ($0, offline — never run Ollama).
  - [ ] (Deferred) `claude` backend via `langchain-anthropic` — consult the `claude-api` skill if added.
- [x] Three `@tool` functions over the processed data: `get_card_profile`, `get_recent_transactions`,
      `get_population_fraud_rate`; fraud-analyst system prompt.
- [x] Guardrails: `recursion_limit`, validate tool args (error observation, don't crash), **three-tier
      decision** — native structured output → `with_structured_output` extraction (small models reason
      but don't emit the schema in one turn) → deterministic threshold fallback. `triage()` never hangs.
- [x] Integration: the consumer runs the cascade (`AlertContext.from_alert` → `triage`) on flagged tx.
- [x] Demo: `scripts/agent_demo.py` triages a random held-out transaction against Ollama.
- [ ] (Optional) MCP server: expose `/predict` + the three tools as MCP tools.

---

## ✅ PR #5 — Polish & infrastructure (branch `polish`)

- [x] **Airflow DAG (real, light).** `airflow/dags/lean_fraud_pipeline.py` sequences
      download→build→train→evaluate→benchmark via `BashOperator`s shelling out to the `python -m`
      entrypoints (orchestration only; logic stays in the package). Added as an optional
      `docker compose --profile airflow` service + `airflow/README.md`; not installed by `uv sync`.
- [x] **`docs/ARCHITECTURE.md` aligned to reality.** The code uses **Kinesis** — the doc now describes
      the Kinesis streaming path + the real training DAG + the CloudWatch alarm as *implemented*, and
      clearly marks the **SQS + Postgres + multi-DAG** inference topology as a **future direction, not
      built**. (Decision: align the doc, not migrate — the migration is a separate large PR.)
- [x] **CloudWatch fraud-rate-spike alarm.** The consumer emits a windowed `FraudAlertRate` custom
      metric (`put_metric_data`, `LeanFraud` namespace); `main.tf` adds an `aws_cloudwatch_metric_alarm`
      (+ `aws_sns_topic` action) that fires when the rate exceeds ~10× baseline — the only signal
      actionable live (no labels at scoring time; latency is a settled non-issue). Provisioned for real
      via `tflocal`; README/ARCHITECTURE note LocalStack Community's limited auto-evaluation.
- [x] **Visual demo driver.** `scripts/demo.sh` drives the whole real-time path end to end (stack up →
      `tflocal` provision → stream a bounded batch → live alerts + triage), plus a `--limit`/`--rate-hz`
      flag on the producer. README has the command + a placeholder for the recorded GIF.
      - [ ] Record the actual GIF and drop it in the README (manual screen-recording step).
- [~] **Duplicate MLflow runs.** Root cause (the `run_id` grouping) is already fixed and merged; the
      leftover runs are stale, local-only data under the git-ignored `mlruns/` — pruned locally, nothing
      to commit. (Prune yours with `mlflow gc` / by deleting old run dirs.)

---

## ✅ PR #6 — Containerization / exportable bundle (branch `docker`)

**Goal:** the "clone & run" thesis — one command brings the whole demo up.

- [x] **Dockerfile on `uv`** — multi-stage build (`ghcr.io/astral-sh/uv` builder → slim runtime),
      deps resolved from `uv.lock` (`--frozen`, `--no-group dev`); one image runs the api, producer,
      and consumer. Added `.dockerignore` to keep the build context small.
- [x] **Model provisioning strategy** — the model isn't baked in (it's git-ignored, produced by
      training); services **bind-mount** the host-trained `./artifacts` + `./data` (read-only), so the
      image stays lean and reproducible. The scorer returns `503` until a model is present.
- [x] **`producer` + `consumer` as compose services** — same image, different commands, behind a
      `stream` profile, plus a one-shot `init` service that provisions the streams/bucket (AWS CLI).
- [x] **One-command up** — `docker compose up -d` (infra + scorer) and `docker compose --profile
      stream up` (full producer→consumer demo); documented in the README.

---

## Priority order

1. ~~**PR #3 serving**~~ ✅ merged
2. ~~**PR #4 agent**~~ ✅ done (differentiator)
3. ~~**PR #5 polish**~~ ✅ done
4. ~~**PR #6 containerization**~~ ✅ done — project complete
