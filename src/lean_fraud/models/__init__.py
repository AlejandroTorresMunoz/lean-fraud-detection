"""Models: lean TCN (ours), heavy Transformer baseline, and tabular baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lean_fraud.models.tcn import TCNClassifier
from lean_fraud.models.transformer import TransformerClassifier

NEURAL_TYPES = ("tcn", "transformer")


def build_model(cfg_model: dict[str, Any], n_features: int) -> torch.nn.Module:
    """Construct a sequence model from the `model` config block.

    Only the neural sequence models (tcn, transformer) are built here; the tabular baselines
    (logreg, xgboost) follow a different, non-sequence training path.
    """
    mtype = cfg_model["type"]
    if mtype == "tcn":
        p = cfg_model.get("tcn", {})
        return TCNClassifier(
            n_features=n_features,
            channels=p.get("channels", [64, 64, 64]),
            kernel_size=p.get("kernel_size", 3),
            dropout=p.get("dropout", 0.1),
        )
    if mtype == "transformer":
        p = cfg_model.get("transformer", {})
        return TransformerClassifier(
            n_features=n_features,
            d_model=p.get("d_model", 128),
            n_heads=p.get("n_heads", 4),
            n_layers=p.get("n_layers", 3),
            dim_feedforward=p.get("dim_feedforward", 256),
            dropout=p.get("dropout", 0.1),
        )
    raise ValueError(
        f"build_model only handles {NEURAL_TYPES}; got {mtype!r} "
        "(logreg/xgboost train via the tabular baseline path)."
    )


def load_checkpoint(path: str | Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild a trained model from a checkpoint saved by train.py.

    Returns (model in eval mode, meta) where meta carries n_features/seq_len/features so
    evaluate and benchmark can shape inputs identically to training.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], ckpt["n_features"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {k: ckpt[k] for k in ("n_features", "seq_len", "features") if k in ckpt}
    return model, meta
