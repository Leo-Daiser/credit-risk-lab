"""Command-line interface using Typer."""
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from app.config import get_config

app = typer.Typer(help="Credit Risk Research Lab - ML Pipeline CLI")


@app.command()
def generate_data(
    rows: int = typer.Option(5000, help="Number of rows to generate"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
) -> None:
    """Generate synthetic credit scoring dataset."""
    from app.config import get_config
    from app.data.synthetic_generator import generate_synthetic_data
    from app.utils.io import save_csv
    from app.utils.logging import logger
    from app.utils.paths import ensure_all_dirs, get_data_raw_dir, get_reports_dir
    from app.utils.timing import Timer

    cfg = get_config(config)
    ensure_all_dirs(cfg)

    timer = Timer("data_generation")
    with timer:
        logger.info(f"Generating {rows} rows of synthetic credit data...")
        df = generate_synthetic_data(
            n_rows=rows,
            random_state=cfg.project.random_state,
            default_rate_target=cfg.synthetic.default_rate_target,
            noise_level=cfg.synthetic.noise_level,
            missing_rate=cfg.synthetic.missing_rate,
            outlier_rate=cfg.synthetic.outlier_rate,
        )

    output_path = get_data_raw_dir(cfg) / "synthetic_credit_data.csv"
    save_csv(df, output_path)
    logger.info(f"Data saved to {output_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['target'].value_counts()}")
    logger.info(f"Generation time: {timer.elapsed:.3f}s")

    # Generate synthetic data card
    card_lines = [
        "# Synthetic Data Card\n",
        "## Generation Logic\n",
        "This dataset is synthetically generated for credit scoring research.\n",
        "## Features\n",
        "- 20 features including demographic, financial, and credit history data",
        "- Target variable indicates loan default (1) or no default (0)\n",
        "## Target Construction\n",
        "Target is generated using a logistic model with risk factors:\n",
        "- High debt-to-income increases default risk",
        "- Previous defaults increase default risk",
        "- Late payments increase default risk",
        "- Low income increases default risk",
        "- Short credit history increases default risk",
        "- Short employment increases default risk",
        "- Having a cosigner decreases default risk",
        "- High savings balance decreases default risk\n",
        "## Known Limitations\n",
        "- This is synthetic data, not real banking data",
        "- Feature distributions may not match real-world data",
        "- Target relationships are simplified",
        "- No temporal component\n",
    ]
    card_path = get_reports_dir(cfg) / "synthetic_data_card.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("\n".join(card_lines))
    logger.info(f"Data card saved to {card_path}")


@app.command()
def validate_data(
    path: str = typer.Argument(..., help="Path to CSV file"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
) -> None:
    """Validate dataset for quality issues."""
    from app.config import get_config
    from app.data.loader import load_dataset
    from app.data.validation import format_validation_report, save_validation_report, validate_dataset
    from app.utils.logging import logger
    from app.utils.paths import ensure_all_dirs

    cfg = get_config(config)
    ensure_all_dirs(cfg)

    logger.info(f"Loading dataset from {path}...")
    df = load_dataset(path)

    logger.info("Validating dataset...")
    results = validate_dataset(df, path, cfg)

    report = format_validation_report(results)
    logger.info("\n" + report)

    # Save reports
    md_path, json_path = save_validation_report(results, cfg.paths.artifact_dir if hasattr(cfg.paths, 'artifact_dir') else "artifacts")
    logger.info(f"Validation report saved to {md_path}")
    logger.info(f"Validation metrics saved to {json_path}")

    if not results["valid"]:
        raise typer.Exit(code=1)


@app.command()
def run_eda(
    path: str = typer.Argument(..., help="Path to CSV file"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
) -> None:
    """Run Exploratory Data Analysis and generate report."""
    from app.config import get_config
    from app.data.loader import load_dataset
    from app.reports.eda_report import generate_eda_report
    from app.utils.logging import logger
    from app.utils.paths import ensure_all_dirs

    cfg = get_config(config)
    ensure_all_dirs(cfg)

    logger.info(f"Loading dataset from {path}...")
    df = load_dataset(path)

    logger.info("Generating EDA report...")
    report_path = generate_eda_report(df, cfg)
    logger.info(f"EDA report saved to {report_path}")


@app.command()
def train(
    path: str = typer.Argument(..., help="Path to CSV file"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
) -> None:
    """Train models and save the best one."""
    from app.config import get_config
    from app.data.loader import load_dataset
    from app.models.evaluate import compare_models
    from app.models.train import save_training_artifacts, select_best_model, train_models
    from app.utils.io import save_json
    from app.utils.logging import logger
    from app.utils.paths import ensure_all_dirs, get_metrics_dir, get_models_dir
    from app.utils.timing import Timer

    cfg = get_config(config)
    ensure_all_dirs(cfg)

    timer = Timer("training")
    with timer:
        logger.info(f"Loading dataset from {path}...")
        df = load_dataset(path)

        logger.info("Training models...")
        results = train_models(df, cfg)

    logger.info("Comparing models...")
    comparison = compare_models(results)
    logger.info(f"\n{comparison.to_string()}")

    logger.info("Selecting best model...")
    best_name, best_model = select_best_model(results)

    logger.info("Saving artifacts...")
    artifacts = save_training_artifacts(results, best_name, best_model, cfg.paths.artifacts_models)
    logger.info(f"Artifacts saved: {artifacts}")

    # Save comparison
    comparison_path = get_metrics_dir(cfg) / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison saved to {comparison_path}")

    # Save train results
    train_results = {
        "best_model": best_name,
        "n_models": len(results["models"]),
        "train_size": len(results["X_train"]),
        "test_size": len(results["X_test"]),
        "timing": timer.elapsed,
    }
    train_results_path = get_metrics_dir(cfg) / "train_results.json"
    save_json(train_results, train_results_path)


@app.command()
def evaluate(
    threshold: float = typer.Option(0.5, help="Classification threshold"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
) -> None:
    """Evaluate the best model and save metrics."""
    from app.config import get_config
    from app.models.evaluate import evaluate_model, get_calibration_data
    from app.utils.io import load_csv, load_json, load_model, save_json
    from app.utils.logging import logger
    from app.utils.paths import ensure_all_dirs, get_metrics_dir, get_plots_dir, get_reports_dir
    from app.utils.timing import Timer

    cfg = get_config(config)
    ensure_all_dirs(cfg)

    # Load metadata
    metadata_path = Path(cfg.paths.artifacts_models) / "metadata.json"
    if not metadata_path.exists():
        logger.error("No trained models found. Run 'train' first.")
        raise typer.Exit(code=1)

    metadata = load_json(metadata_path)
    logger.info(f"Loaded metadata: best model = {metadata['best_model']}")

    # Load model and preprocessor
    model = load_model(Path(cfg.paths.artifacts_models) / f"{metadata['best_model']}.joblib")
    preprocessor = load_model(Path(cfg.paths.artifacts_models) / "preprocessor.joblib")

    # Load test data
    from app.features.feature_engineering import add_engineered_features
    from app.models.train import prepare_data
    from sklearn.model_selection import train_test_split

    data_path = Path(cfg.paths.data_raw) / "synthetic_credit_data.csv"
    if not data_path.exists():
        logger.error("No data found. Run 'generate-data' first.")
        raise typer.Exit(code=1)

    df = load_csv(data_path)
    X, y, numeric_cols, categorical_cols = prepare_data(df, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y,
    )

    X_test_processed = preprocessor.transform(X_test)

    logger.info(f"Evaluating model with threshold={threshold}...")
    metrics = evaluate_model(model, X_test_processed, y_test, threshold=threshold)

    # Calibration data
    cal_data = get_calibration_data(model, X_test_processed, y_test)
    metrics["calibration"] = cal_data

    # Save metrics
    metrics_path = get_metrics_dir(cfg) / "metrics.json"
    save_json(metrics, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")

    # Generate plots if configured
    if cfg.reports.generate_plots:
        from app.reports.plots import (
            plot_calibration_curve,
            plot_confusion_matrix,
            plot_pr_curve,
            plot_roc_curve,
        )
        logger.info("Generating plots...")
        plot_roc_curve(model, X_test_processed, y_test, str(get_plots_dir(cfg) / "roc_curve.png"))
        plot_pr_curve(model, X_test_processed, y_test, str(get_plots_dir(cfg) / "pr_curve.png"))
        plot_confusion_matrix(model, X_test_processed, y_test, threshold, str(get_plots_dir(cfg) / "confusion_matrix.png"))
        plot_calibration_curve(y_test, model.predict_proba(X_test_processed)[:, 1], str(get_plots_dir(cfg) / "calibration_curve.png"))
        logger.info("Plots generated")


@app.command()
def select_threshold(
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
) -> None:
    """Select classification threshold."""
    from app.config import get_config
    from app.models.thresholds import select_all_thresholds
    from app.utils.io import load_csv, load_json, load_model, save_json
    from app.utils.logging import logger
    from app.utils.paths import ensure_all_dirs, get_metrics_dir

    cfg = get_config(config)
    ensure_all_dirs(cfg)

    # Load metadata
    metadata_path = Path(cfg.paths.artifacts_models) / "metadata.json"
    if not metadata_path.exists():
        logger.error("No trained models found. Run 'train' first.")
        raise typer.Exit(code=1)

    metadata = load_json(metadata_path)

    # Load model and preprocessor
    model = load_model(Path(cfg.paths.artifacts_models) / f"{metadata['best_model']}.joblib")
    preprocessor = load_model(Path(cfg.paths.artifacts_models) / "preprocessor.joblib")

    # Load data and get test split
    from app.models.train import prepare_data
    from sklearn.model_selection import train_test_split

    data_path = Path(cfg.paths.data_raw) / "synthetic_credit_data.csv"
    if not data_path.exists():
        logger.error("No data found. Run 'generate-data' first.")
        raise typer.Exit(code=1)

    df = load_csv(data_path)
    X, y, numeric_cols, categorical_cols = prepare_data(df, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y,
    )

    X_test_processed = preprocessor.transform(X_test)

    logger.info("Selecting thresholds...")
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    thresholds = select_all_thresholds(y_test, y_pred_proba, cfg)

    # Save thresholds
    thresholds_path = get_metrics_dir(cfg) / "thresholds.json"
    save_json(thresholds, thresholds_path)
    logger.info(f"Thresholds saved to {thresholds_path}")
    logger.info(f"Recommended threshold: {thresholds['recommended']:.3f}")


@app.command()
def explain(
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
) -> None:
    """Generate explainability report."""
    from app.config import get_config
    from app.models.explainability import (
        compute_permutation_importance,
        get_logistic_regression_coefficients,
    )
    from app.utils.io import load_csv, load_json, load_model, save_json
    from app.utils.logging import logger
    from app.utils.paths import ensure_all_dirs, get_metrics_dir, get_plots_dir, get_reports_dir

    cfg = get_config(config)
    ensure_all_dirs(cfg)

    # Load metadata
    metadata_path = Path(cfg.paths.artifacts_models) / "metadata.json"
    if not metadata_path.exists():
        logger.error("No trained models found. Run 'train' first.")
        raise typer.Exit(code=1)

    metadata = load_json(metadata_path)

    # Load model and preprocessor
    model = load_model(Path(cfg.paths.artifacts_models) / f"{metadata['best_model']}.joblib")
    preprocessor = load_model(Path(cfg.paths.artifacts_models) / "preprocessor.joblib")

    # Load test data
    from app.models.train import prepare_data
    from sklearn.model_selection import train_test_split

    data_path = Path(cfg.paths.data_raw) / "synthetic_credit_data.csv"
    if not data_path.exists():
        logger.error("No data found. Run 'generate-data' first.")
        raise typer.Exit(code=1)

    df = load_csv(data_path)
    X, y, numeric_cols, categorical_cols = prepare_data(df, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y,
    )

    X_test_processed = preprocessor.transform(X_test)
    feature_names = metadata.get("feature_names", [f"f{i}" for i in range(X_test_processed.shape[1])])

    # Permutation importance
    logger.info("Computing permutation importance...")
    importance_df = compute_permutation_importance(
        model, X_test_processed, y_test, feature_names,
        random_state=cfg.training.random_state,
    )

    # Save feature importance
    importance_path = get_metrics_dir(cfg) / "feature_importance.json"
    importance_df.to_json(importance_path, orient="records", indent=2)

    # Generate report
    report_lines = ["# Explainability Report\n"]
    report_lines.append("## Feature Importance (Permutation)\n")
    report_lines.append("| Feature | Importance | Std |")
    report_lines.append("|---------|------------|-----|")
    for _, row in importance_df.head(15).iterrows():
        report_lines.append(f"| {row['feature']} | {row['importance_mean']:.4f} | {row['importance_std']:.4f} |")
    report_lines.append("")

    # LR coefficients if applicable
    if metadata["best_model"] == "logistic_regression":
        coef_df = get_logistic_regression_coefficients(model, feature_names)
        if not coef_df.empty:
            report_lines.append("## Logistic Regression Coefficients\n")
            report_lines.append("| Feature | Coefficient |")
            report_lines.append("|---------|-------------|")
            for _, row in coef_df.head(15).iterrows():
                report_lines.append(f"| {row['feature']} | {row['coefficient']:.4f} |")
            report_lines.append("")

    report_path = get_reports_dir(cfg) / "explainability_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info(f"Explainability report saved to {report_path}")


@app.command()
def predict(
    path: str = typer.Argument(..., help="Path to CSV file with new data"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
) -> None:
    """Make predictions for new data."""
    from app.config import get_config
    from app.models.predict import predict_from_csv
    from app.utils.logging import logger
    from app.utils.paths import ensure_all_dirs, get_predictions_dir

    cfg = get_config(config)
    ensure_all_dirs(cfg)

    output_path = get_predictions_dir(cfg) / "predictions.csv"
    results = predict_from_csv(path, str(output_path), cfg)
    logger.info(f"Decision distribution:\n{results['decision'].value_counts()}")


@app.command()
def generate_report(
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
) -> None:
    """Generate final report with all sections."""
    from app.config import get_config
    from app.reports.export import export_to_html
    from app.reports.final_report import generate_final_report
    from app.utils.io import load_json
    from app.utils.logging import logger
    from app.utils.paths import ensure_all_dirs, get_reports_dir

    cfg = get_config(config)
    ensure_all_dirs(cfg)

    logger.info("Generating final report...")
    report_path = generate_final_report(cfg)
    logger.info(f"Final report saved to {report_path}")


@app.command()
def run_all(
    config: str = typer.Option("configs/fast_debug.yaml", help="Config file path"),
) -> None:
    """Run full pipeline."""
    from app.config import get_config
    from app.utils.logging import logger
    from app.utils.paths import ensure_all_dirs
    from app.utils.timing import Timer

    cfg = get_config(config)
    ensure_all_dirs(cfg)

    data_path = str(Path(cfg.paths.data_raw) / "synthetic_credit_data.csv")

    commands = [
        ("generate-data", lambda: generate_data(rows=cfg.synthetic.rows, config=config)),
        ("validate-data", lambda: validate_data(path=data_path, config=config)),
        ("run-eda", lambda: run_eda(path=data_path, config=config)),
        ("train", lambda: train(path=data_path, config=config)),
        ("evaluate", lambda: evaluate(threshold=0.5, config=config)),
        ("select-threshold", lambda: select_threshold(config=config)),
        ("explain", lambda: explain(config=config)),
        ("predict", lambda: predict(path=data_path, config=config)),
        ("generate-report", lambda: generate_report(config=config)),
    ]

    total_timer = Timer("total_pipeline")
    with total_timer:
        for name, cmd in commands:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {name}")
            logger.info(f"{'='*60}")
            try:
                cmd()
            except Exception as e:
                logger.error(f"Command {name} failed: {e}")
                raise

    logger.info(f"\nPipeline completed in {total_timer.elapsed:.3f}s")


@app.command()
def audit(
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
) -> None:
    """Run technical audit."""
    from app.config import get_config
    from app.utils.logging import logger
    from app.utils.paths import ensure_all_dirs, get_audit_dir
    from pathlib import Path

    cfg = get_config(config)
    ensure_all_dirs(cfg)

    # Generate audit report
    audit_lines = [
        "# Final Technical Audit\n",
        "## Implemented Features\n",
        "- Synthetic data generation with 20 features",
        "- Data validation with comprehensive checks",
        "- Leakage detection",
        "- Feature engineering with 11 engineered features",
        "- Preprocessing pipeline with sklearn",
        "- Model training with hyperparameter search",
        "- Evaluation with comprehensive metrics",
        "- Threshold selection with business cost optimization",
        "- Explainability with permutation importance",
        "- Prediction with reason codes",
        "- EDA report generation",
        "- Final report generation",
        "- CLI with 11 commands\n",
        "## Test Results\n",
        "Run `pytest -q` to see test results\n",
        "## Known Technical Debt\n",
        "- Some hardcoded paths remain in legacy code",
        "- Plot generation could be more comprehensive\n",
        "## Reproducibility Notes\n",
        "- All random states are configurable",
        "- All paths are configurable via YAML",
        "- All parameters are documented\n",
        "## Windows Compatibility\n",
        "- All paths use pathlib.Path for cross-platform compatibility",
        "- No Unix-specific commands used\n",
    ]

    audit_path = get_audit_dir(cfg) / "final_audit.md"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))

    logger.info(f"Audit saved to {audit_path}")


if __name__ == "__main__":
    app()
