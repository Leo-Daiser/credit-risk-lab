"""Model calibration utilities."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def compute_calibration_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration curve.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        Dictionary with calibration data.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )

    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
    }


def compute_calibration_metrics(y_true: pd.Series, y_pred_proba: np.ndarray) -> dict:
    """Compute calibration-related metrics.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.

    Returns:
        Dictionary with calibration metrics.
    """
    from sklearn.metrics import brier_score_loss, log_loss

    brier = brier_score_loss(y_true, y_pred_proba)
    logloss = log_loss(y_true, y_pred_proba)

    return {
        "brier_score": float(brier),
        "log_loss": float(logloss),
    }
