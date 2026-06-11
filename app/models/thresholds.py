"""Threshold selection strategies."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from app.config import CONFIG
from app.utils.logging import logger


def find_max_f1_threshold(y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
    """Find threshold that maximizes F1 score.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.

    Returns:
        Optimal threshold for max F1.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    logger.info(f"Max F1 threshold: {best_threshold:.3f} (F1={f1_scores[best_idx]:.4f})")
    return float(best_threshold)


def find_threshold_for_recall(y_true: pd.Series, y_pred_proba: np.ndarray, target_recall: float = 0.75) -> float:
    """Find threshold that achieves target recall.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        target_recall: Target recall value.

    Returns:
        Threshold achieving target recall.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)

    for t in sorted(thresholds, reverse=True):
        y_pred = (y_pred_proba >= t).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        if recall >= target_recall:
            logger.info(f"Threshold for recall>={target_recall}: {t:.3f} (recall={recall:.4f})")
            return float(t)

    logger.warning(f"Could not achieve recall>={target_recall}, returning 0.1")
    return 0.1


def find_threshold_for_precision(y_true: pd.Series, y_pred_proba: np.ndarray, target_precision: float = 0.60) -> float:
    """Find threshold that achieves target precision.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        target_precision: Target precision value.

    Returns:
        Threshold achieving target precision.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)

    for t in sorted(thresholds):
        y_pred = (y_pred_proba >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        if precision >= target_precision:
            logger.info(f"Threshold for precision>={target_precision}: {t:.3f} (precision={precision:.4f})")
            return float(t)

    logger.warning(f"Could not achieve precision>={target_precision}, returning 0.9")
    return 0.9


def find_business_cost_threshold(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> float:
    """Find threshold that minimizes business cost.

    Business cost:
    - False negative (missed default): cost_fn = 5
    - False positive (rejected good client): cost_fp = 1

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        cost_fn: Cost of false negative.
        cost_fp: Cost of false positive.

    Returns:
        Threshold with minimum business cost.
    """
    from sklearn.metrics import confusion_matrix

    thresholds = np.arange(0.05, 0.95, 0.01)
    costs = []

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_cost = fn * cost_fn + fp * cost_fp
        costs.append(total_cost)

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]

    logger.info(f"Business cost threshold: {best_threshold:.3f} (cost={costs[best_idx]:.1f})")
    return float(best_threshold)


def select_all_thresholds(y_true: pd.Series, y_pred_proba: np.ndarray, config=None) -> dict:
    """Select thresholds using all strategies.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        config: App configuration.

    Returns:
        Dictionary with all threshold strategies.
    """
    if config is None:
        config = CONFIG

    cost_fn = config.threshold.fn_cost
    cost_fp = config.threshold.fp_cost

    thresholds = {
        "max_f1": find_max_f1_threshold(y_true, y_pred_proba),
        "target_recall_075": find_threshold_for_recall(y_true, y_pred_proba, config.threshold.min_recall),
        "target_precision_060": find_threshold_for_precision(y_true, y_pred_proba, config.threshold.min_precision),
        "business_cost": find_business_cost_threshold(y_true, y_pred_proba, cost_fn, cost_fp),
    }

    # Select recommended threshold (business cost)
    thresholds["recommended"] = thresholds["business_cost"]

    return thresholds
