"""Evaluate quality metrics on the test split.

Reports F1, precision, recall, PR-AUC and the confusion matrix (PR-AUC is the headline for
imbalanced fraud data, not ROC-AUC).

Usage: python -m lean_fraud.evaluate --config configs/base.yaml
"""

from __future__ import annotations

import argparse

from lean_fraud.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    _ = load_config(args.config)

    # TODO: load model + test set, compute f1/precision/recall/pr_auc, write to results table.
    print("[evaluate] TODO: compute F1 / PR-AUC / precision / recall.")


if __name__ == "__main__":
    main()
