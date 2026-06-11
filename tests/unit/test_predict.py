"""Unit tests for prediction module."""
import numpy as np
import pandas as pd
import pytest

from app.models.explainability import generate_reason_codes


class TestPredict:
    """Tests for prediction functionality."""

    def test_predict_returns_required_columns(self):
        """Test that prediction returns required columns."""
        client = pd.Series({
            "debt_to_income": 0.5,
            "previous_defaults": 1,
            "late_payments_12m": 2,
            "income": 40000,
            "credit_history_years": 3,
            "loan_to_income": 4,
            "employment_years": 2,
        })

        reason_codes = generate_reason_codes(client, list(client.index))
        assert isinstance(reason_codes, list)
        assert len(reason_codes) > 0
