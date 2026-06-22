"""Evaluate a trained model on the held-out test split.

Loads the best checkpoint from the LOCAL artifacts dir (durable — not the ephemeral LocalStack S3),
scores val and test, picks the F1-optimal threshold ON VALIDATION, and reports test PR-AUC (the
headline for imbalanced fraud) plus precision/recall/F1 and the confusion matrix at that threshold.

Usage: python -m lean_fraud.evaluate --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lean_fraud.config import load_config
from lean_fraud.data.dataset import SequenceDataset, load_processed
from lean_fraud.metrics import best_f1_threshold, classification_metrics
from lean_fraud.models import load_checkpoint
from lean_fraud.tracking import start_run
from lean_fraud.train import predict_scores


def _run_name(cfg: dict) -> str:
    features_tag = cfg.get("features", {}).get("engineering", "raw")
    return cfg["mlflow"].get("run_name") or f"{cfg['model']['type']}-{features_tag}"


def evaluate(cfg: dict, ckpt_path: str | None = None) -> dict:
    """Score the test split and return the metrics dict."""
    run_name = _run_name(cfg)
    artifacts_dir = Path(cfg.get("artifacts", {}).get("dir", "artifacts")) / run_name
    ckpt_path = ckpt_path or str(artifacts_dir / "best.pt")
    if not Path(ckpt_path).exists():
        raise SystemExit(f"No checkpoint at {ckpt_path}. Train first: python -m lean_fraud.train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(ckpt_path)
    model.to(device)

    data = load_processed(cfg["dataset"]["processed_dir"])
    seq_len = meta.get("seq_len", cfg["dataset"].get("sequence_length", 32))
    batch = cfg["training"].get("batch_size", 512)
    val_ds = SequenceDataset(data["X"], data["y"], data["user"], data["split"], "val", seq_len)
    test_ds = SequenceDataset(data["X"], data["y"], data["user"], data["split"], "test", seq_len)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False)

    # Threshold chosen on val, applied to test (never tuned on test).
    val_scores, val_labels = predict_scores(model, val_loader, device)
    threshold = best_f1_threshold(val_labels, val_scores)
    test_scores, test_labels = predict_scores(model, test_loader, device)
    metrics = classification_metrics(test_labels, test_scores, threshold=threshold)

    print(
        f"[evaluate] {run_name}  PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
        f"F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
        f"@thr={threshold:.4f}"
    )
    print(
        f"[evaluate] confusion  TP={metrics['tp']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TN={metrics['tn']}"
    )

    (artifacts_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    with start_run(cfg.get("mlflow"), run_name) as run:
        run.log_metrics({f"test_{k}": v for k, v in metrics.items() if isinstance(v, float)})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default=None, help="override checkpoint path")
    args = parser.parse_args()
    evaluate(load_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
