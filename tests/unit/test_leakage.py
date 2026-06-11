"""Unit tests for leakage detection."""
import numpy as np
import pandas as pd

from app.data.leakage import (
    detect_leakage_columns,
    detect_perfect_separation,
    detect_suspicious_target_correlation,
    generate_leakage_report,
)


class TestLeakageDetection:
    """Tests for leakage detection module."""

    def test_finds_column_named_target_leak(self):
        """Test that detector finds column named target_leak."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "target_leak": [0, 1, 0, 1, 0],
            "target": [0, 1, 0, 1, 0],
        })
        suspicious = detect_leakage_columns(df, "target")
        assert "target_leak" in suspicious

    def test_finds_near_perfect_correlation(self):
        """Test that detector finds near-perfect correlation."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "target": [0, 0, 1, 1, 1],
        })
        # Create near-perfect correlation
        df["leaky_feature"] = df["target"] * 100 + np.random.normal(0, 0.01, 5)
        suspicious = detect_suspicious_target_correlation(df, "target", threshold=0.95)
        assert len(suspicious) > 0

    def test_finds_perfect_separation(self):
        """Test that detector finds perfect target separation."""
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "C"],
            "target": [0, 0, 1, 1, 0],
        })
        # Category A always has target=0
        suspicious = detect_perfect_separation(df, "target")
        assert "category" in suspicious

    def test_generate_leakage_report(self):
        """Test that leakage report is generated correctly."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        })
        report = generate_leakage_report(df, "target")
        assert "has_leakage" in report
        assert "suspicious_columns" in report
