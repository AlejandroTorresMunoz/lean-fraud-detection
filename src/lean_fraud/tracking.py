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
from pathlib import Path
from typing import Any, Iterator

# train.py writes the MLflow run id here; evaluate/benchmark read it to RESUME the same run (they run
# as separate subprocesses, so without this each would open its own run and a single cell would
# scatter train/test/benchmark metrics across three runs).
RUN_ID_FILE = "mlflow_run_id.txt"


def save_run_id(artifacts_dir: str | Path, run_id: str | None) -> None:
    """Persist the active run id so sibling subprocesses can resume the same run (no-op if None)."""
    if not run_id:
        return
    d = Path(artifacts_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / RUN_ID_FILE).write_text(run_id, encoding="utf-8")


def load_run_id(artifacts_dir: str | Path) -> str | None:
    """Read the run id train.py persisted, or None if absent (tracking off / not trained yet)."""
    p = Path(artifacts_dir) / RUN_ID_FILE
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


class _NoOpRun:
    """Same interface as the real handle; does nothing. Used when tracking is off/unavailable."""

    run_id: str | None = None

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_metric(self, key: str, value: float, step: int | None = None) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def set_tags(self, tags: dict[str, Any]) -> None: ...


class _MlflowRun:
    def __init__(self, mlflow: Any) -> None:
        self._mlflow = mlflow

    @property
    def run_id(self) -> str | None:
        run = self._mlflow.active_run()
        return run.info.run_id if run else None

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
def start_run(
    cfg_mlflow: dict | None, run_name: str | None = None, run_id: str | None = None
) -> Iterator[Any]:
    """Context manager yielding a run handle (real MLflow or a no-op).

    Pass `run_id` to RESUME an existing run (used by evaluate/benchmark so their metrics land in the
    run train.py opened); omit it to start a fresh run named `run_name`.
    """
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

    uri = cfg_mlflow.get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(uri)
    # Reaching the backend can fail (server down, or a sqlite file locked by an open MLflow UI).
    # Degrade to a no-op rather than hang/crash the training run.
    try:
        mlflow.set_experiment(cfg_mlflow.get("experiment", "lean-fraud"))
        run_ctx = (
            mlflow.start_run(run_id=run_id)
            if run_id
            else mlflow.start_run(run_name=run_name or cfg_mlflow.get("run_name"))
        )
    except Exception as exc:
        print(f"[tracking] MLflow unavailable ({uri}): {exc}; continuing without tracking.")
        yield _NoOpRun()
        return
    with run_ctx:
        yield _MlflowRun(mlflow)
