"""Plotting utilities."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from app.utils.io import ensure_dir


def plot_roc_curve(
    model: object,
    X_test: np.ndarray,
    y_test: pd.Series,
    output_path: str = "artifacts/plots/roc_curve.png",
) -> Path:
    """Plot ROC curve and save to file.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        output_path: Output file path.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    ax.grid(True, alpha=0.3)

    path = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pr_curve(
    model: object,
    X_test: np.ndarray,
    y_test: pd.Series,
    output_path: str = "artifacts/plots/pr_curve.png",
) -> Path:
    """Plot Precision-Recall curve and save to file.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        output_path: Output file path.

    Returns:
        Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("Precision-Recall Curve")
    ax.grid(True, alpha=0.3)

    path = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(
    model: object,
    X_test: np.ndarray,
    y_test: pd.Series,
    threshold: float = 0.5,
    output_path: str = "artifacts/plots/confusion_matrix.png",
) -> Path:
    """Plot confusion matrix and save to file.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        threshold: Classification threshold.
        output_path: Output file path.

    Returns:
        Path to saved plot.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})")

    path = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_calibration_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    output_path: str = "artifacts/plots/calibration_curve.png",
) -> Path:
    """Plot calibration curve and save to file.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        output_path: Output file path.

    Returns:
        Path to saved plot.
    """
    from sklearn.calibration import calibration_curve

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    output_path: str = "artifacts/plots/feature_importance.png",
) -> Path:
    """Plot feature importance and save to file.

    Args:
        importance_df: DataFrame with feature importances.
        top_n: Number of top features to plot.
        output_path: Output file path.

    Returns:
        Path to saved plot.
    """
    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_features["importance_mean"].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"].values)
    ax.set_xlabel("Importance (mean)")
    ax.set_title(f"Top {top_n} Feature Importances (Permutation)")
    ax.invert_yaxis()

    # Add error bars
    ax.errorbar(
        top_features["importance_mean"].values,
        range(len(top_features)),
        xerr=top_features["importance_std"].values,
        fmt="none",
        color="black",
        capsize=3,
    )

    path = ensure_dir(Path(output_path).parent) / Path(output_path).name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
