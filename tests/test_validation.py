"""Tests for data validation."""
import pandas as pd

from app.data.validation import format_validation_report, validate_dataset


class TestDataValidation:
    """Tests for data validation module."""

    def test_validates_correct_dataset(self):
        """Test that validation passes for valid dataset."""
        from app.data.synthetic_generator import generate_synthetic_data
        df = generate_synthetic_data(n_rows=100)
        results = validate_dataset(df)
        assert results["valid"] is True

    def test_catches_missing_target(self):
        """Test that validation catches missing target column."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        results = validate_dataset(df)
        assert results["valid"] is False
        assert any("target" in err.lower() for err in results["errors"])

    def test_detects_class_imbalance(self):
        """Test that validation detects severe class imbalance."""
        from app.data.synthetic_generator import generate_synthetic_data
        df = generate_synthetic_data(n_rows=100)
        # Create severe imbalance
        df.loc[df.index[:90], "target"] = 0
        df.loc[df.index[90:], "target"] = 1
        results = validate_dataset(df)
        # Should have warning about imbalance
        assert len(results["warnings"]) > 0 or results["stats"]["class_distribution"][0] > 9 * results["stats"]["class_distribution"][1]

    def test_format_validation_report(self):
        """Test that validation report is formatted correctly."""
        results = {"valid": True, "errors": [], "warnings": [], "stats": {}}
        report = format_validation_report(results)
        assert "VALID" in report
        assert "=== Dataset Validation Report ===" in report
