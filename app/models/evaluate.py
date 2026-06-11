"""Model evaluation utilities."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.utils.logging import logger


def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model and return comprehensive metrics.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        threshold: Classification threshold.

    Returns:
        Dictionary with evaluation metrics.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Core metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Brier score
    brier = brier_score_loss(y_test, y_pred_proba)

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "brier_score": float(brier),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "threshold": float(threshold),
    }

    logger.info(f"Evaluation Metrics (threshold={threshold:.3f}):")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  PR-AUC:  {pr_auc:.4f}")
    logger.info(f"  F1:      {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:  {recall:.4f}")
    logger.info(f"  Brier:   {brier:.4f}")

    return metrics


def get_calibration_data(
    model: object,
    X_test: np.ndarray,
    y_test: pd.Series,
    n_bins: int = 10,
) -> dict:
    """Get calibration curve data.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        n_bins: Number of bins for calibration.

    Returns:
        Dictionary with calibration data.
    """
    from sklearn.calibration import calibration_curve

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_pred_proba, n_bins=n_bins
    )

    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
    }


def compare_models(results: dict) -> pd.DataFrame:
    """Compare all trained models.

    Args:
        results: Training results dictionary.

    Returns:
        DataFrame with model comparison metrics.
    """
    X_test = results["X_test_processed"]
    y_test = results["y_test"]

    comparison = []
    for name, model in results["models"].items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "model": name,
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "pr_auc": average_precision_score(y_test, y_pred_proba),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "brier_score": brier_score_loss(y_test, y_pred_proba),
        }
        comparison.append(metrics)

    df = pd.DataFrame(comparison)
    df = df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return df
