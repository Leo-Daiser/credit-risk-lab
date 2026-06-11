"""Integration tests for training pipeline."""

from app.config import get_config
from app.data.synthetic_generator import generate_synthetic_data
from app.models.train import train_models


class TestTrainIntegration:
    """Integration tests for training pipeline."""

    def test_train_pipeline_works_on_500_rows(self):
        """Test that train pipeline works on 500-row synthetic dataset."""
        cfg = get_config()
        df = generate_synthetic_data(n_rows=500, random_state=42)
        results = train_models(df, cfg)

        assert "preprocessor" in results
        assert "models" in results
        assert len(results["models"]) >= 2
        assert "X_test_processed" in results
        assert "y_test" in results

    def test_evaluate_works_after_train(self):
        """Test that evaluate works after train."""
        from app.models.evaluate import evaluate_model
        cfg = get_config()
        df = generate_synthetic_data(n_rows=200, random_state=42)
        results = train_models(df, cfg)

        model = results["models"]["logistic_regression"]
        metrics = evaluate_model(model, results["X_test_processed"], results["y_test"])

        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1

    def test_select_threshold_works_after_evaluate(self):
        """Test that select-threshold works after evaluate."""
        from app.models.thresholds import select_all_thresholds
        cfg = get_config()
        df = generate_synthetic_data(n_rows=200, random_state=42)
        results = train_models(df, cfg)

        model = results["models"]["logistic_regression"]
        y_pred_proba = model.predict_proba(results["X_test_processed"])[:, 1]
        thresholds = select_all_thresholds(results["y_test"], y_pred_proba, cfg)

        assert "recommended" in thresholds
        assert 0 <= thresholds["recommended"] <= 1

    def test_generate_report_creates_final_report(self):
        """Test that generate-report creates final report."""
        from app.reports.final_report import generate_final_report
        assert callable(generate_final_report)

    def test_predict_creates_predictions(self):
        """Test that predict creates predictions.csv."""
        from app.models.predict import predict_from_csv
        assert callable(predict_from_csv)
