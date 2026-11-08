"""Validation-only threshold selection helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def choose_threshold_for_recall(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    min_recall: float = 0.60,
) -> dict[str, float | str]:
    """Maximize precision among validation thresholds meeting a recall floor."""
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be greater than 0 and at most 1.")
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return {"threshold": 0.5, "precision": float(precision[0]), "recall": float(recall[0]), "selection": "single-score fallback"}

    threshold_precision = precision[:-1]
    threshold_recall = recall[:-1]
    eligible = np.flatnonzero(threshold_recall >= min_recall)
    if eligible.size:
        best_precision = threshold_precision[eligible].max()
        best = eligible[threshold_precision[eligible] == best_precision]
        index = best[np.argmax(thresholds[best])]
        selection = f"highest precision subject to recall >= {min_recall:.2f}"
    else:
        best_recall = threshold_recall.max()
        best = np.flatnonzero(threshold_recall == best_recall)
        index = best[np.argmax(threshold_precision[best])]
        selection = "maximum achievable recall; configured floor was unattainable"

    return {
        "threshold": float(thresholds[index]),
        "precision": float(threshold_precision[index]),
        "recall": float(threshold_recall[index]),
        "selection": selection,
    }

