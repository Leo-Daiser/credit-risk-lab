"""Unit tests for explainability."""
import pandas as pd

from app.models.explainability import generate_reason_codes


class TestExplainability:
    """Tests for explainability module."""

    def test_reason_codes_non_empty_for_high_risk(self):
        """Test that reason codes are non-empty for high-risk applicant."""
        risky_client = pd.Series({
            "debt_to_income": 0.8,  # High DTI
            "previous_defaults": 2,  # Has defaults
            "late_payments_12m": 5,  # Many late payments
            "income": 20000,  # Low income
            "credit_history_years": 1,  # Short history
            "loan_to_income": 8,  # High loan to income
            "employment_years": 1,  # Unstable employment
        })

        reason_codes = generate_reason_codes(risky_client, list(risky_client.index))
        assert len(reason_codes) > 0

    def test_reason_codes_fewer_for_low_risk(self):
        """Test that reason codes are fewer for low-risk applicant."""
        low_risk_client = pd.Series({
            "debt_to_income": 0.1,  # Low DTI
            "previous_defaults": 0,  # No defaults
            "late_payments_12m": 0,  # No late payments
            "income": 100000,  # High income
            "credit_history_years": 15,  # Long history
            "loan_to_income": 2,  # Low loan to income
            "employment_years": 10,  # Stable employment
        })

        reason_codes = generate_reason_codes(low_risk_client, list(low_risk_client.index))
        assert len(reason_codes) < 5

    def test_reason_codes_are_deterministic(self):
        """Test that reason codes are deterministic."""
        client = pd.Series({
            "debt_to_income": 0.5,
            "previous_defaults": 1,
            "late_payments_12m": 2,
            "income": 40000,
            "credit_history_years": 3,
            "loan_to_income": 4,
            "employment_years": 2,
        })

        codes1 = generate_reason_codes(client, list(client.index))
        codes2 = generate_reason_codes(client, list(client.index))
        assert codes1 == codes2
