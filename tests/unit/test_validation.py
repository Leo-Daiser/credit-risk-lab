"""Unit tests for data validation."""
import pandas as pd
import pytest

from app.data.validation import format_validation_report, validate_dataset


class TestDataValidation:
    """Tests for data validation module."""

    def test_catches_missing_target(self):
        """Test that validation catches missing target column."""
        from app.config import get_config
        cfg = get_config()
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        results = validate_dataset(df, config=cfg)
        assert results["valid"] is False
        assert any("target" in err.lower() for err in results["errors"])

    def test_catches_impossible_age(self):
        """Test that validation catches impossible age values."""
        from app.config import get_config
        from app.data.synthetic_generator import generate_synthetic_data
        cfg = get_config()
        df = generate_synthetic_data(n_rows=100)
        df.loc[0, "age"] = 10  # Below minimum age of 18
        results = validate_dataset(df, config=cfg)
        assert any("age" in w.lower() for w in results["warnings"])

    def test_catches_duplicate_applicant_id(self):
        """Test that validation catches duplicate applicant_id."""
        from app.config import get_config
        from app.data.synthetic_generator import generate_synthetic_data
        cfg = get_config()
        df = generate_synthetic_data(n_rows=100)
        df.loc[1, "applicant_id"] = df.loc[0, "applicant_id"]  # Duplicate ID
        results = validate_dataset(df, config=cfg)
        assert not results["valid"]
        assert any("duplicate" in err.lower() for err in results["errors"])

    def test_validates_correct_dataset(self):
        """Test that validation passes for valid dataset."""
        from app.config import get_config
        from app.data.synthetic_generator import generate_synthetic_data
        cfg = get_config()
        df = generate_synthetic_data(n_rows=100)
        results = validate_dataset(df, config=cfg)
        assert results["valid"] is True

    def test_format_validation_report(self):
        """Test that validation report is formatted correctly."""
        results = {"valid": True, "errors": [], "warnings": [], "stats": {}}
        report = format_validation_report(results)
        assert "VALID" in report
        assert "# Data Validation Report" in report
