"""Efficiency benchmark — the differentiator of this repo.

Measures, per model: parameter count, model size (MB), and inference latency p50/p99 plus
throughput. This is what fills the README results table alongside quality metrics.

Usage: python -m lean_fraud.benchmark --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lean_fraud.config import load_config
from lean_fraud.models import build_model, load_checkpoint
from lean_fraud.tracking import load_run_id, start_run


def measure_latency(
    model: torch.nn.Module, sample: torch.Tensor, n: int = 1000
) -> dict[str, float]:
    """Return p50/p99 latency (ms) for single-sample inference on CPU."""
    model.eval()
    times: list[float] = []
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(sample)
        for _ in range(n):
            t0 = time.perf_counter()
            model(sample)
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {"p50_ms": float(np.percentile(arr, 50)), "p99_ms": float(np.percentile(arr, 99))}


def model_size_mb(model: torch.nn.Module) -> float:
    """On-disk weight footprint in MB (sum of parameter byte sizes)."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def benchmark(cfg: dict, ckpt_path: str | None = None, n_iters: int = 1000) -> dict:
    """Param count + size + single-sample p50/p99 latency for the configured model.

    Latency and size are weight-independent, so this runs on the trained checkpoint when present,
    else on a freshly built model (random weights) — useful before training too.
    """
    features_tag = cfg.get("features", {}).get("engineering", "raw")
    run_name = cfg["mlflow"].get("run_name") or f"{cfg['model']['type']}-{features_tag}"
    artifacts_dir = Path(cfg.get("artifacts", {}).get("dir", "artifacts")) / run_name

    ckpt_path = ckpt_path or str(artifacts_dir / "best.pt")
    if Path(ckpt_path).exists():
        model, meta = load_checkpoint(ckpt_path)
        n_features, seq_len = meta["n_features"], meta["seq_len"]
        source = "checkpoint"
    else:
        meta_json = json.loads((Path(cfg["dataset"]["processed_dir"]) / "meta.json").read_text())
        n_features = meta_json["n_features"]
        seq_len = cfg["dataset"].get("sequence_length", meta_json.get("sequence_length", 32))
        model = build_model(cfg["model"], n_features)
        source = "untrained (random weights)"

    model.eval()
    sample = torch.randn(1, seq_len, n_features)  # one transaction window — real-time scoring shape
    result = {
        "model_type": cfg["model"]["type"],
        "n_params": model.count_parameters(),
        "size_mb": round(model_size_mb(model), 4),
        **measure_latency(model, sample, n=n_iters),
        "source": source,
    }
    print(
        f"[benchmark] {run_name} ({source})  params={result['n_params']}  "
        f"size={result['size_mb']:.3f}MB  p50={result['p50_ms']:.3f}ms  p99={result['p99_ms']:.3f}ms"
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with start_run(cfg.get("mlflow"), run_name, run_id=load_run_id(artifacts_dir)) as run:
        run.log_metrics({k: v for k, v in result.items() if isinstance(v, (int, float))})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model efficiency.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default=None, help="override checkpoint path")
    parser.add_argument("--iters", type=int, default=1000, help="latency measurement iterations")
    args = parser.parse_args()
    benchmark(load_config(args.config), args.checkpoint, args.iters)


if __name__ == "__main__":
    main()
