"""Evaluation metrics and decision-threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def best_threshold(y_true, y_prob, min_recall: float | None = None, grid: np.ndarray | None = None) -> float:
    """Threshold that maximises churn-class F1 while keeping recall >= min_recall.

    Chosen on validation data only. If no threshold reaches min_recall, the
    plain F1-optimal threshold is returned.
    """
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    preds = [(y_prob >= t).astype(int) for t in grid]
    f1 = np.array([f1_score(y_true, p, zero_division=0) for p in preds])
    if min_recall is not None:
        recall = np.array([recall_score(y_true, p, zero_division=0) for p in preds])
        if (recall >= min_recall).any():
            f1 = np.where(recall >= min_recall, f1, -1.0)
    return float(grid[int(np.argmax(f1))])
