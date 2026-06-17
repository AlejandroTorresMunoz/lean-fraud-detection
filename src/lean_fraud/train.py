"""Train a model and log to MLflow.

Usage: python -m lean_fraud.train --config configs/base.yaml
"""

from __future__ import annotations

import argparse

from lean_fraud.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fraud model.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    print(f"[train] model={cfg['model']['type']} epochs={cfg['training']['epochs']}")
    # TODO:
    #  - load processed sequences
    #  - build model (TCNClassifier / TransformerClassifier / baselines)
    #  - focal/weighted loss, early stopping
    #  - mlflow.log_params / log_metrics / log_artifact(model)
    #  - upload artifact to the (emulated) S3 bucket


if __name__ == "__main__":
    main()
