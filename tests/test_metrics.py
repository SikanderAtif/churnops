import numpy as np

from src.metrics import best_threshold, classification_metrics


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert m["f1"] == 1.0 and m["roc_auc"] == 1.0 and m["fn"] == 0


def test_best_threshold_finds_separating_point():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.35, 0.4])
    assert 0.3 < best_threshold(y, p) <= 0.35


def test_best_threshold_respects_min_recall():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.3, 0.8, 0.85, 0.9])
    t = best_threshold(y, p, min_recall=1.0)
    assert classification_metrics(y, p, t)["recall"] == 1.0
