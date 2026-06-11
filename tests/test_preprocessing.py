"""Tests for preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from app.features.preprocessing import build_preprocessing_pipeline


class TestPreprocessing:
    """Tests for preprocessing pipeline."""

    def test_preprocessing_handles_missing_values(self):
        """Test that preprocessing does not fail on missing values."""
        df = pd.DataFrame({
            "num1": [1.0, np.nan, 3.0, 4.0, 5.0],
            "num2": [10.0, 20.0, np.nan, 40.0, 50.0],
            "cat1": ["a", "b", "a", None, "b"],
            "target": [0, 1, 0, 1, 0],
        })

        numeric_cols = ["num1", "num2"]
        categorical_cols = ["cat1"]

        pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols)
        X = df[numeric_cols + categorical_cols]
        result = pipeline.fit_transform(X)

        assert result is not None
        assert result.shape[0] == 5
        assert not np.isnan(result).any()

    def test_preprocessing_scales_numeric(self):
        """Test that numeric features are scaled."""
        df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num2": [100.0, 200.0, 300.0, 400.0, 500.0],
            "target": [0, 1, 0, 1, 0],
        })

        numeric_cols = ["num1", "num2"]
        categorical_cols = []

        pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols, scale_numeric=True)
        X = df[numeric_cols]
        result = pipeline.fit_transform(X)

        # After scaling, mean should be close to 0
        assert abs(result[:, 0].mean()) < 0.1
        assert abs(result[:, 1].mean()) < 0.1

    def test_preprocessing_encodes_categorical(self):
        """Test that categorical features are one-hot encoded."""
        df = pd.DataFrame({
            "cat1": ["a", "b", "a", "b", "a"],
            "target": [0, 1, 0, 1, 0],
        })

        numeric_cols = []
        categorical_cols = ["cat1"]

        pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols)
        X = df[categorical_cols]
        result = pipeline.fit_transform(X)

        # One-hot encoding should create 2 columns for 2 categories
        assert result.shape[1] == 2
