"""Train a sequence model (TCN / Transformer) and log to MLflow.

Loads the processed table, builds causal per-card windows lazily (SequenceDataset), trains with
focal loss for the ~0.5% fraud imbalance, early-stops on validation PR-AUC, and saves the best
checkpoint to the artifacts dir (later: the emulated S3 bucket). Everything is config-driven so the
same entrypoint runs the raw-feature and triple-PCA experiments for an apples-to-apples MLflow
comparison.

Usage: python -m lean_fraud.train --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lean_fraud.config import load_config
from lean_fraud.data.dataset import SequenceDataset, load_processed
from lean_fraud.losses import build_loss
from lean_fraud.metrics import classification_metrics
from lean_fraud.models import build_model
from lean_fraud.tracking import save_run_id, start_run


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict_scores(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader, returning (scores in [0,1], labels) as numpy arrays."""
    model.eval()
    scores, labels = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def train(cfg: dict) -> dict:
    """Train one model from a config dict; return the best validation metrics."""
    _set_seed(cfg.get("seed", 42))
    tr_cfg = cfg["training"]
    device = _resolve_device(tr_cfg.get("device", "auto"))

    data = load_processed(cfg["dataset"]["processed_dir"])
    seq_len = cfg["dataset"].get("sequence_length", 32)
    n_features = data["X"].shape[1]

    train_ds = SequenceDataset(data["X"], data["y"], data["user"], data["split"], "train", seq_len)
    val_ds = SequenceDataset(data["X"], data["y"], data["user"], data["split"], "val", seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg.get("batch_size", 512),
        shuffle=True,
        num_workers=tr_cfg.get("num_workers", 0),
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=tr_cfg.get("batch_size", 512), shuffle=False)

    model = build_model(cfg["model"], n_features).to(device)
    loss_fn = build_loss(tr_cfg)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tr_cfg.get("lr", 1e-3), weight_decay=tr_cfg.get("weight_decay", 0.0)
    )

    n_params = model.count_parameters()
    features_tag = cfg.get("features", {}).get("engineering", "raw")
    run_name = cfg["mlflow"].get("run_name") or f"{cfg['model']['type']}-{features_tag}"
    print(
        f"[train] model={cfg['model']['type']} params={n_params} device={device} run={run_name}",
        flush=True,
    )

    artifacts_dir = Path(cfg.get("artifacts", {}).get("dir", "artifacts")) / run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifacts_dir / "best.pt"

    max_batches = tr_cfg.get("max_train_batches")
    n_epochs = tr_cfg.get("epochs", 8)
    log_every_epochs = max(tr_cfg.get("log_every_epochs", 1), 1)
    log_every_batches = tr_cfg.get("log_every_batches", 0)  # within-epoch heartbeat (0 = off)
    total_batches = (
        len(train_loader) if max_batches is None else min(max_batches, len(train_loader))
    )
    patience = tr_cfg.get("early_stopping_patience", 3)
    best_pr_auc, best_metrics, best_state, no_improve = -1.0, {}, None, 0
    print(f"[train] starting: {n_epochs} epochs x {total_batches} batches/epoch", flush=True)

    with start_run(cfg.get("mlflow"), run_name) as run:
        # Record the run id so evaluate/benchmark (separate subprocesses) resume THIS run.
        save_run_id(artifacts_dir, run.run_id)
        run.set_tags({"model_type": cfg["model"]["type"], "features": features_tag})
        run.log_params(
            {
                "model_type": cfg["model"]["type"],
                "n_features": n_features,
                "seq_len": seq_len,
                "n_params": n_params,
                "loss": tr_cfg.get("loss", "focal"),
                "lr": tr_cfg.get("lr", 1e-3),
                "batch_size": tr_cfg.get("batch_size", 512),
                "epochs": tr_cfg.get("epochs", 8),
            }
        )

        for epoch in range(n_epochs):
            model.train()
            running, n_seen = 0.0, 0
            for b, (xb, yb) in enumerate(train_loader):
                if max_batches is not None and b >= max_batches:
                    break
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(yb)
                n_seen += len(yb)
                if log_every_batches and (b + 1) % log_every_batches == 0:
                    print(
                        f"[train] epoch {epoch:>2}  batch {b + 1}/{total_batches}  "
                        f"loss={running / max(n_seen, 1):.4f}",
                        flush=True,
                    )

            train_loss = running / max(n_seen, 1)
            scores, labels = predict_scores(model, val_loader, device)
            val = classification_metrics(labels, scores)
            # Validation/MLflow run every epoch (early stopping needs it); only the console
            # summary is throttled to every `log_every_epochs` epochs (and the last one).
            if epoch % log_every_epochs == 0 or epoch == n_epochs - 1:
                print(
                    f"[train] epoch {epoch:>2}  loss={train_loss:.4f}  "
                    f"val_pr_auc={val['pr_auc']:.4f}  val_roc_auc={val['roc_auc']:.4f}",
                    flush=True,
                )
            run.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_pr_auc": val["pr_auc"],
                    "val_roc_auc": val["roc_auc"],
                },
                step=epoch,
            )

            if val["pr_auc"] > best_pr_auc:
                best_pr_auc, best_metrics, no_improve = val["pr_auc"], val, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(
                        f"[train] early stop at epoch {epoch} (no val PR-AUC gain for {patience})"
                    )
                    break

        if best_state is not None:
            torch.save(
                {
                    "state_dict": best_state,
                    "model": cfg["model"],
                    "n_features": n_features,
                    "seq_len": seq_len,
                    "features": features_tag,
                },
                ckpt_path,
            )
            (artifacts_dir / "val_metrics.json").write_text(
                json.dumps({"n_params": n_params, **best_metrics}, indent=2), encoding="utf-8"
            )
            run.log_metrics(
                {f"best_val_{k}": v for k, v in best_metrics.items() if isinstance(v, float)}
            )
            run.log_artifact(str(ckpt_path))
            print(f"[train] best val PR-AUC={best_pr_auc:.4f}  saved {ckpt_path}")

    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fraud model.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], help="override training.device"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["training"]["device"] = args.device
    train(cfg)


if __name__ == "__main__":
    main()
