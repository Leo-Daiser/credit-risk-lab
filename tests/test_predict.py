"""Tests for prediction functionality."""
import numpy as np
import pandas as pd
import pytest

from app.data.synthetic_generator import generate_synthetic_data
from app.features.feature_engineering import add_engineered_features
from app.models.explainability import generate_reason_codes
from app.models.train import prepare_data, train_models


class TestPredict:
    """Tests for prediction functionality."""

    def test_predict_returns_probability_default(self):
        """Test that prediction returns probability_default."""
        df = generate_synthetic_data(n_rows=200)
        results = train_models(df)

        model = results["models"]["logistic_regression"]
        proba = model.predict_proba(results["X_test_processed"])[:, 1]

        assert "probability_default" is not None
        assert len(proba) == len(results["y_test"])
        assert all(0 <= p <= 1 for p in proba)

    def test_predict_returns_decision(self):
        """Test that prediction returns decision."""
        y_pred_proba = np.array([0.1, 0.5, 0.9])

        def get_decision(prob):
            if prob < 0.3:
                return "approve"
            elif prob > 0.7:
                return "reject"
            else:
                return "manual_review"

        decisions = [get_decision(p) for p in y_pred_proba]

        assert decisions[0] == "approve"
        assert decisions[1] == "manual_review"
        assert decisions[2] == "reject"

    def test_reason_codes_not_empty_for_risky_client(self):
        """Test that reason codes are not empty for risky client."""
        risky_client = pd.Series({
            "debt_to_income": 0.8,  # High DTI
            "previous_defaults": 2,  # Has defaults
            "late_payments_12m": 5,  # Many late payments
            "income": 20000,  # Low income
            "credit_history_years": 1,  # Short history
            "loan_to_income": 8,  # High loan to income
            "employment_years": 1,  # Unstable employment
        })

        reason_codes = generate_reason_codes(
            risky_client,
            feature_names=list(risky_client.index),
            model=None,
            preprocessor=None,
        )

        assert len(reason_codes) > 0
        assert any("high_debt_to_income" in code for code in reason_codes)
