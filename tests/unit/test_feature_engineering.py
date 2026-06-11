"""Unit tests for feature engineering."""
import numpy as np
import pandas as pd

from app.features.feature_engineering import add_engineered_features, get_engineered_feature_names


class TestFeatureEngineering:
    """Tests for feature engineering module."""

    def test_creates_all_required_features(self):
        """Test that all required engineered features are created."""
        from app.data.synthetic_generator import generate_synthetic_data
        df = generate_synthetic_data(n_rows=100, random_state=42)
        df_engineered = add_engineered_features(df)

        required_features = get_engineered_feature_names()
        for feat in required_features:
            assert feat in df_engineered.columns, f"Missing feature: {feat}"

    def test_avoids_division_by_zero(self):
        """Test that feature engineering avoids division by zero."""
        df = pd.DataFrame({
            "loan_amount": [10000],
            "income": [0],  # Zero income
            "credit_history_years": [0],  # Zero years
            "open_credit_lines": [3],
            "late_payments_12m": [1],
            "debt_to_income": [0.3],
            "employment_years": [0],
            "previous_defaults": [0],
            "savings_balance": [0],
            "checking_balance": [0],
            "purpose": ["car"],
        })
        df_result = add_engineered_features(df)

        # Should not have inf values
        numeric_cols = df_result.select_dtypes(include=[np.number]).columns
        assert not np.isinf(df_result[numeric_cols]).any().any()

    def test_handles_missing_values(self):
        """Test that feature engineering handles missing values."""
        from app.data.synthetic_generator import generate_synthetic_data
        df = generate_synthetic_data(n_rows=100, random_state=42)
        df.loc[0, "income"] = np.nan
        df.loc[1, "credit_history_years"] = np.nan
        df_engineered = add_engineered_features(df)

        # Should not crash and should have same number of rows
        assert len(df_engineered) == 100

    def test_preserves_target_column(self):
        """Test that target column is preserved."""
        from app.data.synthetic_generator import generate_synthetic_data
        df = generate_synthetic_data(n_rows=100, random_state=42)
        df_engineered = add_engineered_features(df)
        assert "target" in df_engineered.columns
