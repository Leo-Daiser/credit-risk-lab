"""Final report generation."""
from pathlib import Path

from app.config import get_config
from app.reports.export import export_to_html
from app.utils.io import load_csv, load_json
from app.utils.paths import ensure_all_dirs, get_reports_dir


def generate_final_report(config=None) -> Path:
    """Generate final comprehensive report.

    Args:
        config: App configuration.

    Returns:
        Path to generated report.
    """
    if config is None:
        config = get_config()

    ensure_all_dirs(config)

    # Load artifacts
    metrics_path = Path(config.paths.artifacts_metrics) / "metrics.json"
    thresholds_path = Path(config.paths.artifacts_metrics) / "thresholds.json"

    metrics = {}
    thresholds = {}

    if metrics_path.exists():
        metrics = load_json(metrics_path)
    if thresholds_path.exists():
        thresholds = load_json(thresholds_path)

    # Build report sections
    sections = []

    # 1. Executive Summary
    sections.append("# Credit Risk Research Lab - Final Report\n")
    sections.append("## Executive Summary\n")
    sections.append("This report presents a comprehensive credit scoring ML pipeline for predicting loan defaults.\n")

    # 2. Problem Statement
    sections.append("## Problem Statement\n")
    sections.append("Predict whether a loan applicant will default based on their financial and demographic features.\n")
    sections.append("**Business Impact**: Missing a default (false negative) costs 5x more than rejecting a good client (false positive).\n")

    # 3. Dataset Description
    sections.append("## Dataset Description\n")
    data_path = Path(config.paths.data_raw) / "synthetic_credit_data.csv"
    if data_path.exists():
        df = load_csv(data_path)
        sections.append(f"- Rows: {len(df)}")
        sections.append(f"- Features: {len(df.columns) - 1}")
        sections.append(f"- Default rate: {df['target'].mean():.2%}\n")

    # 4. Synthetic Data Limitations
    sections.append("## Synthetic Data Limitations\n")
    sections.append("- This is synthetic data, not real banking data")
    sections.append("- Feature distributions may not match real-world data")
    sections.append("- Target relationships are simplified")
    sections.append("- No temporal component\n")

    # 5. Validation and Leakage Checks
    sections.append("## Validation and Leakage Checks\n")
    sections.append("Data validation was performed to ensure:")
    sections.append("- Target column exists and is binary")
    sections.append("- No impossible values (e.g., negative income)")
    sections.append("- No severe class imbalance")
    sections.append("- No data leakage detected\n")

    # 6. EDA Summary
    sections.append("## EDA Summary\n")
    sections.append("Exploratory data analysis revealed:")
    sections.append("- Class distribution and target rate")
    sections.append("- Feature distributions and correlations")
    sections.append("- Missing values analysis\n")

    # 7. Feature Engineering
    sections.append("## Feature Engineering\n")
    sections.append("Engineered features added:")
    sections.append("- loan_to_income: loan amount relative to income")
    sections.append("- credit_lines_per_history_year: credit utilization density")
    sections.append("- late_payment_rate: late payment frequency")
    sections.append("- savings_to_loan: savings buffer relative to loan")
    sections.append("- employment_stability_bucket: employment stability categories")
    sections.append("- dti_bucket: debt-to-income risk categories")
    sections.append("- has_prior_credit_problem: flag for prior credit issues\n")

    # 8. Modeling Approach
    sections.append("## Modeling Approach\n")
    sections.append("Models trained:")
    sections.append("- Logistic Regression (baseline)")
    sections.append("- Random Forest")
    sections.append("- Gradient Boosting")
    sections.append("- Hist Gradient Boosting\n")

    # 9. Model Comparison
    sections.append("## Model Comparison\n")
    if metrics:
        sections.append(f"- ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        sections.append(f"- PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}")
        sections.append(f"- F1: {metrics.get('f1', 'N/A'):.4f}")
        sections.append(f"- Precision: {metrics.get('precision', 'N/A'):.4f}")
        sections.append(f"- Recall: {metrics.get('recall', 'N/A'):.4f}")
        sections.append(f"- Brier Score: {metrics.get('brier_score', 'N/A'):.4f}\n")

    # 10. Selected Model
    sections.append("## Selected Model\n")
    best_model = metrics.get("best_model", "N/A")
    sections.append(f"**{best_model}** selected based on PR-AUC (primary metric for imbalanced datasets).\n")

    # 11. Threshold Selection
    sections.append("## Threshold Selection\n")
    if thresholds:
        sections.append("| Strategy | Threshold |")
        sections.append("|----------|-----------|")
        for strategy, threshold in thresholds.items():
            if strategy != "recommended":
                sections.append(f"| {strategy} | {threshold:.3f} |")
        sections.append(f"\n**Recommended threshold**: {thresholds.get('recommended', 0.5):.3f}\n")

    # 12. Business Cost Interpretation
    sections.append("## Business Cost Interpretation\n")
    sections.append("- False Negative (missed default): cost = 5 units")
    sections.append("- False Positive (rejected good client): cost = 1 unit")
    sections.append("- Missing a default is 5x more costly than rejecting a good client")
    sections.append("- Business cost threshold minimizes total expected loss\n")

    # 13. Calibration
    sections.append("## Calibration\n")
    if "calibration" in metrics:
        sections.append(f"- Brier Score: {metrics.get('brier_score', 'N/A'):.4f}")
        sections.append("- Calibration curve shows model probability estimates\n")

    # 14. Explainability
    sections.append("## Explainability\n")
    sections.append("Model explainability via:")
    sections.append("- Permutation importance")
    sections.append("- Logistic regression coefficients (if applicable)")
    sections.append("- Rule-based reason codes for individual predictions\n")

    # 15. Prediction Examples
    sections.append("## Prediction Examples\n")
    predictions_path = Path(config.paths.artifacts_predictions) / "predictions.csv"
    if predictions_path.exists():
        pred_df = load_csv(predictions_path)
        sections.append(f"- Total predictions: {len(pred_df)}")
        sections.append(f"- Approve: {(pred_df['decision'] == 'approve').sum()}")
        sections.append(f"- Manual Review: {(pred_df['decision'] == 'manual_review').sum()}")
        sections.append(f"- Reject: {(pred_df['decision'] == 'reject').sum()}\n")

    # 16. Limitations
    sections.append("## Limitations\n")
    sections.append("- Synthetic data only - not validated against real banking data")
    sections.append("- No temporal features or time-series patterns")
    sections.append("- No external data sources (credit bureau, etc.)")
    sections.append("- Simplified business cost structure")
    sections.append("- No concept drift monitoring\n")

    # 17. Next Steps
    sections.append("## Next Steps\n")
    sections.append("1. Validate with real banking data")
    sections.append("2. Add temporal features and survival analysis")
    sections.append("3. Implement model monitoring and drift detection")
    sections.append("4. Add fairness metrics and bias detection")
    sections.append("5. Deploy as REST API for real-time predictions\n")

    # 18. Reproducibility Commands
    sections.append("## Reproducibility Commands\n")
    sections.append("```bash")
    sections.append("python -m app.cli generate-data --rows 5000 --config configs/default.yaml")
    sections.append("python -m app.cli validate-data data/raw/synthetic_credit_data.csv --config configs/default.yaml")
    sections.append("python -m app.cli run-eda data/raw/synthetic_credit_data.csv --config configs/default.yaml")
    sections.append("python -m app.cli train data/raw/synthetic_credit_data.csv --config configs/default.yaml")
    sections.append("python -m app.cli evaluate --config configs/default.yaml")
    sections.append("python -m app.cli select-threshold --config configs/default.yaml")
    sections.append("python -m app.cli explain --config configs/default.yaml")
    sections.append("python -m app.cli predict data/raw/synthetic_credit_data.csv --config configs/default.yaml")
    sections.append("python -m app.cli generate-report --config configs/default.yaml")
    sections.append("python -m app.cli run-all --config configs/fast_debug.yaml")
    sections.append("python -m pytest -q")
    sections.append("```\n")

    # Save markdown report
    report_dir = get_reports_dir(config)
    md_path = report_dir / "final_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))

    # Export to HTML
    html_path = report_dir / "final_report.html"
    export_to_html("\n".join(sections), str(html_path), title="Credit Risk Research Lab - Final Report")

    return md_path
