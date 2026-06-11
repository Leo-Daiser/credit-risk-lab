"""Tests for model training."""

from app.data.synthetic_generator import generate_synthetic_data
from app.models.train import prepare_data, train_models


class TestTrain:
    """Tests for model training module."""

    def test_logistic_regression_trains(self):
        """Test that Logistic Regression can be trained."""
        from app.models.baseline import get_logistic_regression
        from app.features.preprocessing import build_preprocessing_pipeline

        df = generate_synthetic_data(n_rows=200)
        X, y, numeric_cols, categorical_cols = prepare_data(df)

        pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols)
        X_processed = pipeline.fit_transform(X)

        model = get_logistic_regression()
        model.fit(X_processed, y)

        assert hasattr(model, "coef_")
        assert model.coef_.shape[1] == X_processed.shape[1]

    def test_train_models_returns_results(self):
        """Test that train_models returns complete results."""
        df = generate_synthetic_data(n_rows=200)
        results = train_models(df)

        assert "preprocessor" in results
        assert "models" in results
        assert "X_test_processed" in results
        assert "y_test" in results
        assert len(results["models"]) >= 3  # At least LR, RF, GB

    def test_models_have_predict_proba(self):
        """Test that all trained models have predict_proba."""
        df = generate_synthetic_data(n_rows=200)
        results = train_models(df)

        for name, model in results["models"].items():
            assert hasattr(model, "predict_proba"), f"{name} missing predict_proba"
            proba = model.predict_proba(results["X_test_processed"])
            assert proba.shape[1] == 2, f"{name} predict_proba should have 2 columns"
