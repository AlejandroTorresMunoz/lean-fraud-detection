"""Offline training pipeline as an Airflow DAG: download -> build -> train -> evaluate -> benchmark.

**Orchestration only.** Each task shells out to a ``python -m lean_fraud.<module>`` entrypoint — the
exact same commands documented in the README and run by hand — so the ML logic stays in the package
and Airflow just sequences it (and gives retries + a UI). This is the *batch* path; the real-time
scorer is the FastAPI service + the stream consumer and is deliberately **not** on Airflow.

Manual trigger (``schedule=None``): the pipeline is run on demand, not on a clock.

Point the DAG at the project with env vars (all optional):

- ``LEAN_FRAUD_HOME``   — repo root containing ``src/lean_fraud`` (default ``/opt/lean-fraud``, the
  compose mount; set it to the repo path when running Airflow locally).
- ``LEAN_FRAUD_PYTHON`` — interpreter that has the project installed (default: the one parsing this
  DAG, so running Airflow inside the project's own venv "just works").
- ``LEAN_FRAUD_CONFIG`` — config passed to train/evaluate/benchmark (default ``configs/base.yaml``).

See ``airflow/README.md`` for how to run it.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

HOME = os.environ.get("LEAN_FRAUD_HOME", "/opt/lean-fraud")
PYTHON = os.environ.get("LEAN_FRAUD_PYTHON", sys.executable)
CONFIG = os.environ.get("LEAN_FRAUD_CONFIG", "configs/base.yaml")


def _cmd(module: str, args: str = "") -> str:
    """Build the shell command for a `python -m lean_fraud.<module>` step, run from the repo root."""
    return f"cd {HOME} && {PYTHON} -m lean_fraud.{module} {args}".strip()


with DAG(
    dag_id="lean_fraud_pipeline",
    description="Offline fraud pipeline: download -> build -> train -> evaluate -> benchmark.",
    schedule=None,  # manual trigger; not a periodic batch
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"retries": 1},
    tags=["lean-fraud", "training"],
) as dag:
    download = BashOperator(task_id="download", bash_command=_cmd("data.download"))
    build_sequences = BashOperator(
        task_id="build_sequences", bash_command=_cmd("data.build_sequences")
    )
    train = BashOperator(task_id="train", bash_command=_cmd("train", f"--config {CONFIG}"))
    evaluate = BashOperator(task_id="evaluate", bash_command=_cmd("evaluate", f"--config {CONFIG}"))
    benchmark = BashOperator(
        task_id="benchmark", bash_command=_cmd("benchmark", f"--config {CONFIG}")
    )

    download >> build_sequences >> train >> evaluate >> benchmark
