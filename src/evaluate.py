import numpy as np
from sklearn.metrics import confusion_matrix


def compute_metrics(y_true, y_proba, threshold=0.5):
    """Calculates classification metrics at a specific threshold."""

    # If proba >= 0.5, predict 1 (Defaulter), otherwise 0 (Safe)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Out of ALL actual defaulters (TP + FN), what % did we catch (TP)?
    recall = tp / (tp + fn)

    # Out of everyone we called a defaulter (TP + FP), how many actually were (TP)?
    precision = tp / (tp + fp)

    return {
        'threshold': threshold,
        'defaulters_caught': tp,
        'defaulters_missed': fn,
        'recall': recall,
        'precision': precision
    }


def find_optimal_threshold(y_true, y_proba, cost_fn=10.0, cost_fp=1.0):
    """
    Simulate 100 decision thresholds to find the cutoff that
    minimizes total financial loss (10:1 default cost ratio)
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_cost = float('inf')
    optimal_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Business loss formula
        total_cost = (fn * cost_fn) + (fp * cost_fp)

        if total_cost < best_cost:
            best_cost = total_cost
            optimal_thresh = thresh

    print(f"\n[Business Optimizer] Optimal Decision Threshold: {optimal_thresh:.3f}")
    return optimal_thresh