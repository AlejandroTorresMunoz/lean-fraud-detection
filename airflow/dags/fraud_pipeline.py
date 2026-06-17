"""Airflow DAG orchestrating the batch training pipeline (Airflow is part of Revolut's stack).

ingest -> preprocess/build sequences -> train -> evaluate -> benchmark -> register model.
This is a skeleton: tasks shell out to the same `python -m lean_fraud.*` entrypoints.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="fraud_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["fraud", "lean", "mlops"],
) as dag:
    download = BashOperator(task_id="download", bash_command="python -m lean_fraud.data.download")
    build = BashOperator(
        task_id="build_sequences", bash_command="python -m lean_fraud.data.build_sequences"
    )
    train = BashOperator(task_id="train", bash_command="python -m lean_fraud.train")
    evaluate = BashOperator(task_id="evaluate", bash_command="python -m lean_fraud.evaluate")
    benchmark = BashOperator(task_id="benchmark", bash_command="python -m lean_fraud.benchmark")

    download >> build >> train >> evaluate >> benchmark
