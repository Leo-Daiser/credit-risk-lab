"""Tests for model evaluation."""
import numpy as np
import pytest

from app.data.synthetic_generator import generate_synthetic_data
from app.models.evaluate import evaluate_model
from app.models.train import prepare_data, train_models


class TestEvaluate:
    """Tests for model evaluation module."""

    def test_evaluate_returns_roc_auc(self):
        """Test that evaluate returns ROC-AUC."""
        df = generate_synthetic_data(n_rows=200)
        results = train_models(df)

        model = results["models"]["logistic_regression"]
        metrics = evaluate_model(model, results["X_test_processed"], results["y_test"])

        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_evaluate_returns_pr_auc(self):
        """Test that evaluate returns PR-AUC."""
        df = generate_synthetic_data(n_rows=200)
        results = train_models(df)

        model = results["models"]["logistic_regression"]
        metrics = evaluate_model(model, results["X_test_processed"], results["y_test"])

        assert "pr_auc" in metrics
        assert 0 <= metrics["pr_auc"] <= 1

    def test_evaluate_returns_all_metrics(self):
        """Test that evaluate returns all required metrics."""
        df = generate_synthetic_data(n_rows=200)
        results = train_models(df)

        model = results["models"]["logistic_regression"]
        metrics = evaluate_model(model, results["X_test_processed"], results["y_test"])

        required_metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall", "brier_score", "confusion_matrix"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
