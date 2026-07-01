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

## PR #3 — Real serving (branch `serving`) ⭐ next, highest technical value

**Goal:** kill the toy scorer and serve the real TCN end to end. No dependencies (`best.pt` already exists).

- [ ] Load the model at startup in `serve/api.py` via `load_checkpoint()` (startup, not per request).
- [ ] Real `/predict`: build the sequence window from the payload, apply the **scaler from `meta.json`**
      (same normalization as training), run inference, return prob + real `latency_ms`.
- [ ] Handle short input: pad when fewer than `seq_len` transactions arrive.
- [ ] Same scorer in `streaming/consumer.py`: extract the scoring logic into a shared module
      (`scoring.py`) to avoid duplication.
- [ ] Decision threshold: load the val-tuned threshold (saved by evaluate) instead of a hardcoded 0.5.
- [ ] Tests: smoke test for `/predict` with the real model; test the consumer scoring a message.
- [ ] Update README (drop any "toy" wording; `latency_ms` is now real).

---

## PR #4 — Fraud-triage agent (branch `agent`) — Phase 2 differentiator

**Goal:** a **local, $0** LLM layer on top of the alerts (maps to Revolut's Sherlock + AIR; showcases
the agentic/MCP experience from the CV). **Cascade:** the cheap TCN scores all traffic → the agent
runs only on the flagged ~0.5%.
**Dependency:** ideally after PR #3 (real alerts to triage).

- [ ] Design: input (tx + score + features + card context) → output (block/review/allow + rationale).
- [ ] **Orchestration on LangChain + LangGraph** (`create_react_agent`, `recursion_limit`); no
      hand-rolled loop.
- [ ] Pluggable backend by config (`agent.provider: ollama | mock`):
  - [ ] `ollama` (default): local model via `langchain-ollama` `ChatOllama(...).bind_tools([...])`
        (tool-calling + structured output); default `qwen2.5:3b` for the 6 GB GPU, `qwen2.5:7b` fallback.
  - [ ] `mock`: deterministic LangChain fake chat model for tests/CI ($0, offline — never run Ollama).
  - [ ] (Deferred) `claude` backend via `langchain-anthropic` — consult the `claude-api` skill if added.
- [ ] Three `@tool` functions over the processed data: `get_card_profile`, `get_recent_transactions`,
      `get_population_fraud_rate`; fraud-analyst system prompt.
- [ ] Guardrails: `recursion_limit`, validate tool args (error observation, don't crash), structured
      output (Pydantic via `with_structured_output`), deterministic threshold fallback.
- [ ] Integration: the agent consumes the alerts the consumer emits.
- [ ] (Optional) MCP server: expose `/predict` + the three tools as MCP tools.
- [ ] Demo + tests (mock the LLM in tests to avoid token spend / CI cost).

---

## PR #5 — Polish & infrastructure (branch `polish`)

- [ ] Clean up the duplicate MLflow runs from before the `run_id` fix.
- [ ] Airflow DAG: implement it for real (download→build→train→eval→benchmark) or trim the skeleton.
- [ ] `docs/ARCHITECTURE.md`: decide on the aspirational Kinesis→SQS+Postgres migration
      (implement it or align the doc with what exists).
- [ ] Visual demo: GIF/screenshot of `/predict` + the stream for the README (depends on PR #3).
- [ ] **CloudWatch fraud-rate-spike alarm** (depends on PR #3 — real alerts). CloudWatch is already
      enabled in LocalStack (`SERVICES=...,cloudwatch`) but unused.
      - Consumer emits a `FraudAlertRate` custom metric via boto3 `put_metric_data`.
      - `aws_cloudwatch_metric_alarm` in `main.tf` (+ optional `aws_sns_topic` as the action) that
        fires when the alert rate exceeds N× baseline over a window — detects attacks/drift, the only
        thing actionable live (no ground-truth labels at scoring time, so F1/PR-AUC can't be monitored
        in real time; latency is already a settled non-issue at ~10× headroom, so no latency alarm).
      - README: label honestly — the alarm is provisioned for real via `tflocal`, but LocalStack
        community's automatic alarm-state evaluation is limited (may need `set-alarm-state`).

---

## Priority order

1. **PR #3 serving** (high return, no dependencies)
2. **PR #4 agent** (differentiator, depends on #3)
3. **PR #5 polish** (once the rest is in)
