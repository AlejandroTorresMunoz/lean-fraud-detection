"""Thin MLflow wrapper, guarded so experiments never hard-depend on a tracking server.

`start_run(cfg["mlflow"], ...)` yields a handle exposing log_params / log_metrics / log_metric /
log_artifact / set_tags. If MLflow is disabled in config or not importable, it yields a no-op handle
with the same surface — so train/evaluate/benchmark stay identical whether or not tracking is on, and
CI (which never trains) is unaffected. The config's `tracking_uri` selects the backend — a local
sqlite file (`sqlite:///mlflow.db`, offline, no server) or the compose MLflow server; the deprecated
./mlruns file store is intentionally avoided.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


class _NoOpRun:
    """Same interface as the real handle; does nothing. Used when tracking is off/unavailable."""

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_metric(self, key: str, value: float, step: int | None = None) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def set_tags(self, tags: dict[str, Any]) -> None: ...


class _MlflowRun:
    def __init__(self, mlflow: Any) -> None:
        self._mlflow = mlflow

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)

    def set_tags(self, tags: dict[str, Any]) -> None:
        self._mlflow.set_tags(tags)


@contextmanager
def start_run(cfg_mlflow: dict | None, run_name: str | None = None) -> Iterator[Any]:
    """Context manager yielding a run handle (real MLflow or a no-op)."""
    cfg_mlflow = cfg_mlflow or {}
    if not cfg_mlflow.get("enabled", False):
        yield _NoOpRun()
        return
    try:
        import mlflow
    except ImportError:
        print("[tracking] mlflow not installed; continuing without tracking.")
        yield _NoOpRun()
        return

    if cfg_mlflow.get("tracking_uri"):
        mlflow.set_tracking_uri(cfg_mlflow["tracking_uri"])
    mlflow.set_experiment(cfg_mlflow.get("experiment", "lean-fraud"))
    with mlflow.start_run(run_name=run_name or cfg_mlflow.get("run_name")):
        yield _MlflowRun(mlflow)
