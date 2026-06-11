"""Tests for feature engineering."""
import numpy as np
import pandas as pd
import pytest

from app.features.feature_engineering import add_engineered_features, get_engineered_feature_names


class TestFeatureEngineering:
    """Tests for feature engineering module."""

    def test_adds_required_features(self):
        """Test that all required engineered features are added."""
        from app.data.synthetic_generator import generate_synthetic_data
        df = generate_synthetic_data(n_rows=100)
        df_engineered = add_engineered_features(df)

        required_features = get_engineered_feature_names()
        for feat in required_features:
            assert feat in df_engineered.columns, f"Missing feature: {feat}"

    def test_loan_to_income_calculation(self):
        """Test loan_to_income = loan_amount / income."""
        df = pd.DataFrame({
            "loan_amount": [10000, 20000],
            "income": [50000, 100000],
            "credit_history_years": [5, 10],
            "open_credit_lines": [3, 5],
            "late_payments_12m": [1, 2],
            "debt_to_income": [0.3, 0.5],
        })
        df_result = add_engineered_features(df)

        assert np.isclose(df_result["loan_to_income"].iloc[0], 10000 / 50000)
        assert np.isclose(df_result["loan_to_income"].iloc[1], 20000 / 100000)

    def test_risk_bucket_dti_values(self):
        """Test that risk_bucket_dti has correct categories."""
        from app.data.synthetic_generator import generate_synthetic_data
        df = generate_synthetic_data(n_rows=100)
        df = add_engineered_features(df)

        valid_buckets = {"low", "medium", "high", "extreme"}
        assert set(df["risk_bucket_dti"].unique()).issubset(valid_buckets)

    def test_handles_zero_division(self):
        """Test that division by zero is handled gracefully."""
        df = pd.DataFrame({
            "loan_amount": [10000],
            "income": [0],  # Zero income
            "credit_history_years": [0],  # Zero years
            "open_credit_lines": [3],
            "late_payments_12m": [1],
            "debt_to_income": [0.3],
        })
        df_result = add_engineered_features(df)

        # Should not have NaN or inf
        assert not df_result["loan_to_income"].isna().any()
        assert not np.isinf(df_result["loan_to_income"]).any()
