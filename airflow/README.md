# Airflow orchestration

One **batch** DAG, [`dags/lean_fraud_pipeline.py`](dags/lean_fraud_pipeline.py), that sequences the
offline model pipeline:

```
download ─► build_sequences ─► train ─► evaluate ─► benchmark
```

Each task is a `BashOperator` that shells out to a `python -m lean_fraud.<module>` entrypoint — the
**same commands** in the README's Quickstart. Airflow only orchestrates (ordering, retries, a UI); all
the ML logic stays in the package. `schedule=None`, so it runs on manual trigger.

> **Scope, honestly:** this is the *batch/training* path only. The **real-time scorer** is the FastAPI
> `/predict` service + the Kinesis stream consumer, which are long-running services and **not** on
> Airflow. The larger multi-DAG SQS+Postgres inference topology in
> [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) is a **future** design, not what runs today.

## Run it — locally, in the project venv (lightest)

The tasks call `python -m lean_fraud.*`, so run Airflow in an environment that already has the project
installed. The DAG defaults `LEAN_FRAUD_PYTHON` to the interpreter that parses it, so pointing Airflow
at this repo is enough:

```bash
pip install "apache-airflow==2.10.5"          # in the project's venv (or a dedicated one)
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=false
export LEAN_FRAUD_HOME="$PWD"
airflow standalone                             # prints a login + password, UI on :8080
```

Then trigger `lean_fraud_pipeline` from the UI (or `airflow dags trigger lean_fraud_pipeline`).

## Run it — via docker compose (optional profile)

Kept behind a profile so the default `docker compose up` stays lean:

```bash
docker compose --profile airflow up airflow     # UI on http://localhost:8080
```

The service mounts this repo at `/opt/lean-fraud` and installs it on first boot
(`_PIP_ADDITIONAL_REQUIREMENTS`), so the **first** start is slow (it pulls torch etc.). For a fast
iteration loop, prefer the local-venv route above.

> Airflow is **not** installed by `uv sync` (it is a heavy dep with its own constraints, and CI never
> needs it), so it is intentionally not in `pyproject.toml`. Install it as shown above only when you
> actually want to run the DAG.
