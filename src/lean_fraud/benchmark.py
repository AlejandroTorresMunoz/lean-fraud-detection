"""Efficiency benchmark — the differentiator of this repo.

Measures, per model: parameter count, model size (MB), and inference latency p50/p99 plus
throughput. This is what fills the README results table alongside quality metrics.

Usage: python -m lean_fraud.benchmark --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def measure_latency(model: torch.nn.Module, sample: torch.Tensor, n: int = 1000) -> dict[str, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model efficiency.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.parse_args()
    # TODO: load each trained model, run measure_latency, log params/size, emit results table.
    print("[benchmark] TODO: params + size + p50/p99 latency for TCN vs Transformer vs baselines.")


if __name__ == "__main__":
    main()
