"""Run the experiment matrix {tcn, transformer} x {raw, triple_pca} as ISOLATED subprocesses.

Each cell's build/train/evaluate/benchmark runs in its own `python -m` subprocess. This is good
sweep hygiene (no state leakage between runs) and, on Windows, a hard requirement: importing the
full module set (torch + scipy/sklearn + pandas) into ONE process and then running a Conv1d
corrupts the native stack and crashes ("Windows fatal exception: stack overflow"). The standalone
entrypoints each import only what they need, so they are stable. The orchestrator itself never
runs a model — it only spawns subprocesses and aggregates their JSON outputs.

Usage:
  python -m lean_fraud.experiments --config configs/base.yaml
  python -m lean_fraud.experiments --models tcn --features raw          # a single cell
  python -m lean_fraud.experiments --rebuild                            # force dataset rebuild
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from lean_fraud.config import load_config

MODELS = ("tcn", "transformer")
FEATURES = ("raw", "triple_pca")


def _processed_dir(base_processed: str, engineering: str) -> str:
    """raw uses the base processed dir; triple_pca gets a sibling `*_pca` dir."""
    return base_processed if engineering == "raw" else f"{base_processed.rstrip('/')}_pca"


def _write_cell_config(cfg: dict) -> str:
    """Dump a per-cell config to a temp YAML and return its path."""
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(cfg, fh, sort_keys=False)
    fh.close()
    return fh.name


def _run_module(module: str, cfg_path: str) -> int:
    """Run `python -u -m <module> --config <cfg_path>` as a subprocess, streaming its output live."""
    return subprocess.run(
        [sys.executable, "-u", "-m", module, "--config", cfg_path], check=False
    ).returncode


def _collect(cfg: dict, run_name: str) -> dict:
    """Read the test + efficiency metrics a cell's subprocesses wrote to the artifacts dir."""
    art = Path(cfg.get("artifacts", {}).get("dir", "artifacts")) / run_name
    test = json.loads((art / "test_metrics.json").read_text(encoding="utf-8"))
    bench = json.loads((art / "benchmark.json").read_text(encoding="utf-8"))
    return {
        "model": cfg["model"]["type"],
        "features": cfg["features"]["engineering"],
        "params": bench["n_params"],
        "pr_auc": test["pr_auc"],
        "f1": test["f1"],
        "p50_ms": bench["p50_ms"],
        "p99_ms": bench["p99_ms"],
    }


def run_matrix(
    cfg: dict, models: list[str], features_list: list[str], rebuild: bool = False
) -> list[dict]:
    """Build/train/evaluate/benchmark each (model, features) cell as subprocesses; one row each."""
    rows: list[dict] = []
    for engineering in features_list:
        processed_dir = _processed_dir(cfg["dataset"]["processed_dir"], engineering)

        # Build this feature variant's dataset once (if missing), in its own process.
        if rebuild or not (Path(processed_dir) / "sequences.npz").exists():
            build_cfg = copy.deepcopy(cfg)
            build_cfg["dataset"]["processed_dir"] = processed_dir
            build_cfg["features"]["engineering"] = engineering
            build_path = _write_cell_config(build_cfg)
            print(f"\n===== build dataset ({engineering}) =====", flush=True)
            rc = _run_module("lean_fraud.data.build_sequences", build_path)
            Path(build_path).unlink(missing_ok=True)
            if rc != 0:
                for model_type in models:
                    rows.append(
                        {"model": model_type, "features": engineering, "error": "build failed"}
                    )
                continue

        for model_type in models:
            run_name = f"{model_type}-{engineering}"
            run_cfg = copy.deepcopy(cfg)
            run_cfg["model"]["type"] = model_type
            run_cfg["features"]["engineering"] = engineering
            run_cfg["dataset"]["processed_dir"] = processed_dir
            run_cfg["mlflow"]["run_name"] = run_name
            cfg_path = _write_cell_config(run_cfg)
            print(f"\n===== {run_name} =====", flush=True)
            try:
                for module in ("lean_fraud.train", "lean_fraud.evaluate", "lean_fraud.benchmark"):
                    if _run_module(module, cfg_path) != 0:
                        raise RuntimeError(f"{module} exited non-zero")
                rows.append(_collect(run_cfg, run_name))
            except Exception as exc:  # keep the sweep alive if a cell fails
                print(f"[experiments] {run_name} FAILED: {exc}", flush=True)
                rows.append({"model": model_type, "features": engineering, "error": str(exc)})
            finally:
                Path(cfg_path).unlink(missing_ok=True)
    _print_summary(rows)
    return rows


def _print_summary(rows: list[dict]) -> None:
    print("\n===== results =====", flush=True)
    print(
        f"{'model':<12} {'features':<11} {'params':>8} {'PR-AUC':>8} "
        f"{'F1':>7} {'p50ms':>7} {'p99ms':>7}",
        flush=True,
    )
    for r in rows:
        if "error" in r:
            print(f"{r['model']:<12} {r['features']:<11}  ERROR: {r['error']}", flush=True)
        else:
            print(
                f"{r['model']:<12} {r['features']:<11} {r['params']:>8} {r['pr_auc']:>8.4f} "
                f"{r['f1']:>7.4f} {r['p50_ms']:>7.3f} {r['p99_ms']:>7.3f}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model x features experiment matrix.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument(
        "--rebuild", action="store_true", help="force rebuild of processed datasets"
    )
    args = parser.parse_args()
    run_matrix(load_config(args.config), args.models, args.features, args.rebuild)


if __name__ == "__main__":
    main()
