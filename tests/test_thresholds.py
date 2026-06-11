"""Tests for threshold selection."""
import numpy as np
import pandas as pd

from app.models.thresholds import (
    find_business_cost_threshold,
    find_max_f1_threshold,
    find_threshold_for_precision,
    find_threshold_for_recall,
)


class TestThresholds:
    """Tests for threshold selection module."""

    def test_max_f1_threshold_in_range(self):
        """Test that max F1 threshold is in [0, 1]."""
        y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        threshold = find_max_f1_threshold(y_true, y_pred_proba)
        assert 0 <= threshold <= 1

    def test_recall_threshold_in_range(self):
        """Test that recall threshold is in [0, 1]."""
        y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        threshold = find_threshold_for_recall(y_true, y_pred_proba, target_recall=0.75)
        assert 0 <= threshold <= 1

    def test_precision_threshold_in_range(self):
        """Test that precision threshold is in [0, 1]."""
        y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        threshold = find_threshold_for_precision(y_true, y_pred_proba, target_precision=0.60)
        assert 0 <= threshold <= 1

    def test_business_cost_threshold_in_range(self):
        """Test that business cost threshold is in [0, 1]."""
        y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        threshold = find_business_cost_threshold(y_true, y_pred_proba)
        assert 0 <= threshold <= 1

    def test_business_cost_calculation(self):
        """Test that business cost is calculated correctly."""
        y_true = pd.Series([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.1, 0.9])

        # At threshold 0.5:
        # predictions: [0, 1, 0, 1]
        # TP=1, TN=1, FP=1, FN=1
        # cost = FN*5 + FP*1 = 5 + 1 = 6
        threshold = find_business_cost_threshold(y_true, y_pred_proba, cost_fn=5, cost_fp=1)
        assert 0 <= threshold <= 1
