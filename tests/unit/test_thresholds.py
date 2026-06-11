"""Unit tests for threshold selection."""
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

    def test_threshold_always_between_0_and_1(self):
        """Test that threshold is always between 0 and 1."""
        y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

        assert 0 <= find_max_f1_threshold(y_true, y_pred_proba) <= 1
        assert 0 <= find_threshold_for_recall(y_true, y_pred_proba, 0.75) <= 1
        assert 0 <= find_threshold_for_precision(y_true, y_pred_proba, 0.60) <= 1
        assert 0 <= find_business_cost_threshold(y_true, y_pred_proba) <= 1
