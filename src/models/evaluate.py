from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def optimize_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, beta: float = 0.5
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Last point from precision_recall_curve has no threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    scores = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-12)
    best_idx = int(np.nanargmax(scores))

    return float(thresholds[best_idx]), float(scores[best_idx])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f_beta_0_5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }
