"""Unit tests for model registry."""
import pytest

from app.models.registry import (
    get_available_models,
    get_model_specs,
    get_search_space,
)


class TestModelRegistry:
    """Tests for model registry."""

    def test_returns_required_models(self):
        """Test that registry returns required models."""
        models = get_available_models()
        assert "logistic_regression" in models
        assert "random_forest" in models
        assert "gradient_boosting" in models

    def test_get_search_space(self):
        """Test that search space is returned."""
        space = get_search_space("logistic_regression")
        assert "C" in space
        assert "penalty" in space

    def test_fast_mode_search_space(self):
        """Test that fast mode search space is minimal."""
        space = get_search_space("logistic_regression", fast_mode=True)
        assert len(space["C"]) == 1
