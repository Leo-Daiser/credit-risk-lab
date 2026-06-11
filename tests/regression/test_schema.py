"""Regression tests for schema stability."""
import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestRegression:
    """Regression tests for schema stability."""

    def test_metrics_json_schema_stable(self):
        """Test that metrics.json schema stays stable."""
        metrics_path = PROJECT_ROOT / "artifacts" / "metrics" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            # Check required keys exist
            assert "roc_auc" in metrics
            assert "pr_auc" in metrics
            assert "f1" in metrics
            assert "precision" in metrics
            assert "recall" in metrics

    def test_predictions_csv_schema_stable(self):
        """Test that predictions.csv schema stays stable."""
        predictions_path = PROJECT_ROOT / "artifacts" / "predictions" / "predictions.csv"
        if predictions_path.exists():
            df = pd.read_csv(predictions_path)
            assert "applicant_id" in df.columns
            assert "probability_default" in df.columns
            assert "decision" in df.columns
            assert "reason_codes" in df.columns

    def test_report_files_are_non_empty(self):
        """Test that report files are non-empty."""
        report_dir = PROJECT_ROOT / "artifacts" / "reports"
        if report_dir.exists():
            for report_file in report_dir.glob("*.md"):
                assert report_file.stat().st_size > 0
            for report_file in report_dir.glob("*.html"):
                assert report_file.stat().st_size > 0
