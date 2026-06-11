"""Unit tests for synthetic data generator."""
import pandas as pd
import pytest

from app.data.synthetic_generator import generate_synthetic_data


class TestSyntheticGenerator:
    """Tests for synthetic data generator."""

    def test_creates_exact_requested_rows(self):
        """Test that generator creates exact requested row count."""
        for n in [100, 500, 1000]:
            df = generate_synthetic_data(n_rows=n, random_state=42)
            assert len(df) == n

    def test_is_deterministic_with_fixed_seed(self):
        """Test that same seed produces same data."""
        df1 = generate_synthetic_data(n_rows=100, random_state=42)
        df2 = generate_synthetic_data(n_rows=100, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_target_has_both_classes(self):
        """Test that target contains both classes (0 and 1)."""
        df = generate_synthetic_data(n_rows=1000, random_state=42)
        assert df["target"].nunique() == 2
        assert 0 in df["target"].values
        assert 1 in df["target"].values

    def test_target_reacts_directionally_to_risk_factors(self):
        """Test that target reacts directionally to risk factors."""
        # Generate base data
        df = generate_synthetic_data(n_rows=2000, random_state=42)

        # Compare high DTI vs low DTI groups
        high_dti = df[df["debt_to_income"] > 0.5]
        low_dti = df[df["debt_to_income"] < 0.2]

        # High DTI group should have higher default rate
        if len(high_dti) > 0 and len(low_dti) > 0:
            assert high_dti["target"].mean() >= low_dti["target"].mean()

    def test_has_all_required_columns(self):
        """Test that all required columns are present."""
        df = generate_synthetic_data(n_rows=100, random_state=42)
        required_columns = [
            "applicant_id", "age", "income", "employment_years", "loan_amount",
            "loan_term_months", "interest_rate", "debt_to_income",
            "credit_history_years", "previous_defaults", "open_credit_lines",
            "late_payments_12m", "home_ownership", "education",
            "purpose", "region", "application_channel", "has_cosigner",
            "savings_balance", "checking_balance", "target",
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_categorical_columns_have_valid_values(self):
        """Test that categorical columns have expected values."""
        df = generate_synthetic_data(n_rows=100, random_state=42)
        assert set(df["home_ownership"].unique()).issubset({"rent", "mortgage", "own", "other"})
        assert set(df["education"].unique()).issubset({"school", "college", "bachelor", "master", "phd"})
        assert set(df["purpose"].unique()).issubset({"car", "education", "business", "home", "medical", "other"})

    def test_numeric_columns_have_positive_values(self):
        """Test that numeric columns have reasonable positive values."""
        df = generate_synthetic_data(n_rows=100, random_state=42, missing_rate=0)
        assert (df["age"] > 0).all()
        assert (df["income"] > 0).all()
        assert (df["loan_amount"] > 0).all()
