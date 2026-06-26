"""Quality metrics for imbalanced fraud detection.

PR-AUC (average precision) is the headline — it reflects performance on the rare positive class,
unlike ROC-AUC which is optimistic under heavy imbalance. Threshold metrics (precision/recall/F1)
are reported at a chosen operating point.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision). The headline metric."""
    return float(average_precision_score(y_true, scores))


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximizing F1 on the precision-recall curve.

    Chosen on validation and then applied to test, so the operating point is never tuned on the
    test set. PR-AUC stays the threshold-free headline; this only fixes where precision/recall/F1
    are read off.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have one more point than thresholds; align to the threshold array.
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    return float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5


def classification_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """PR-AUC, ROC-AUC and precision/recall/F1 + confusion counts at `threshold`."""
    y_pred = (scores >= threshold).astype(np.int8)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {
        "pr_auc": pr_auc(y_true, scores),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
