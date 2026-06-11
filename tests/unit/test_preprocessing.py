"""Unit tests for preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from app.features.preprocessing import build_preprocessing_pipeline


class TestPreprocessing:
    """Tests for preprocessing pipeline."""

    def test_handles_missing_values(self):
        """Test that preprocessing handles missing values."""
        df = pd.DataFrame({
            "num1": [1.0, np.nan, 3.0, 4.0, 5.0],
            "num2": [10.0, 20.0, np.nan, 40.0, 50.0],
            "cat1": ["a", "b", "a", None, "b"],
        })

        numeric_cols = ["num1", "num2"]
        categorical_cols = ["cat1"]

        pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols)
        X = df[numeric_cols + categorical_cols]
        result = pipeline.fit_transform(X)

        assert result is not None
        assert result.shape[0] == 5
        assert not np.isnan(result).any()

    def test_handles_unknown_categories(self):
        """Test that preprocessing handles unknown categories."""
        df_train = pd.DataFrame({
            "cat1": ["a", "b", "a", "b"],
        })
        df_test = pd.DataFrame({
            "cat1": ["a", "c", "d"],  # Unknown categories
        })

        pipeline = build_preprocessing_pipeline([], ["cat1"])
        pipeline.fit(df_train)
        result = pipeline.transform(df_test)

        # Should not crash
        assert result.shape[0] == 3

    def test_transformed_matrix_has_expected_row_count(self):
        """Test that transformed matrix has expected row count."""
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "cat1": ["a", "b", "a"],
        })

        pipeline = build_preprocessing_pipeline(["num1"], ["cat1"])
        result = pipeline.fit_transform(df)

        assert result.shape[0] == 3

    def test_target_not_included_in_features(self):
        """Test that target is not included in features."""
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0],
            "cat1": ["a", "b", "a"],
            "target": [0, 1, 0],
        })

        # Target should be excluded before calling this
        features = ["num1", "cat1"]
        pipeline = build_preprocessing_pipeline(["num1"], ["cat1"])
        result = pipeline.fit_transform(df[features])

        # Result should not have target column
        assert result.shape[1] == 3  # num1 + 2 one-hot encoded cat1
