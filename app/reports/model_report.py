"""Model report generation."""
from pathlib import Path

import pandas as pd

from app.utils.io import ensure_dir


def generate_model_report(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    metrics: dict,
    thresholds: dict,
    output_dir: str = "artifacts/reports",
) -> Path:
    """Generate model comparison report in Markdown format.

    Args:
        comparison_df: Model comparison DataFrame.
        best_model_name: Name of selected best model.
        metrics: Evaluation metrics.
        thresholds: Selected thresholds.
        output_dir: Output directory.

    Returns:
        Path to saved report.
    """
    lines = ["# Model Training Report\n"]

    # Model comparison
    lines.append("## Model Comparison\n")
    lines.append("| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Brier Score |")
    lines.append("|-------|---------|--------|-----|-----------|--------|-------------|")
    for _, row in comparison_df.iterrows():
        marker = " **" if row["model"] == best_model_name else ""
        lines.append(
            f"| {row['model']}{marker} | {row['roc_auc']:.4f}{marker} | "
            f"{row['pr_auc']:.4f}{marker} | {row['f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['brier_score']:.4f} |"
        )
    lines.append("")

    # Selected model
    lines.append("## Selected Model\n")
    lines.append(f"**{best_model_name}** selected based on PR-AUC (primary metric for imbalanced datasets).\n")

    # Metrics explanation
    lines.append("## Metrics Interpretation\n")
    lines.append(f"- **ROC-AUC ({metrics['roc_auc']:.4f})**: Measures ability to distinguish between classes. "
                 f"1.0 = perfect, 0.5 = random.")
    lines.append(f"- **PR-AUC ({metrics['pr_auc']:.4f})**: Area under Precision-Recall curve. "
                 f"More informative than ROC-AUC for imbalanced datasets.")
    lines.append(f"- **F1 ({metrics['f1']:.4f})**: Harmonic mean of precision and recall. "
                 f"Balances false positives and false negatives.")
    lines.append(f"- **Precision ({metrics['precision']:.4f})**: Proportion of predicted defaults that were actual defaults.")
    lines.append(f"- **Recall ({metrics['recall']:.4f})**: Proportion of actual defaults that were correctly identified.")
    lines.append(f"- **Brier Score ({metrics['brier_score']:.4f})**: Measures prediction accuracy. Lower is better.")
    lines.append("")

    # Why accuracy is not the main metric
    lines.append("## Why Accuracy is Not the Main Metric\n")
    lines.append("In credit risk modeling, the target variable is typically imbalanced "
                 "(most clients don't default). With 90% non-defaults, a model that predicts "
                 "'no default' for everyone achieves 90% accuracy but is completely useless.\n")
    lines.append("PR-AUC is preferred because:")
    lines.append("- It focuses on the minority class (defaults)")
    lines.append("- It doesn't benefit from predicting the majority class")
    lines.append("- It directly measures the trade-off between precision and recall")
    lines.append("")

    # Threshold selection
    lines.append("## Threshold Selection\n")
    lines.append("| Strategy | Threshold |")
    lines.append("|----------|-----------|")
    for strategy, threshold in thresholds.items():
        if strategy != "recommended":
            lines.append(f"| {strategy} | {threshold:.3f} |")
    lines.append("")

    lines.append(f"**Recommended threshold**: {thresholds.get('recommended', 0.5):.3f} (business cost minimization)\n")

    lines.append("### Business Cost Rationale\n")
    lines.append("- False negative (missed default): cost = 5 units")
    lines.append("- False positive (rejected good client): cost = 1 unit")
    lines.append("- Missing a default is 5x more costly than rejecting a good client")
    lines.append("- Business cost threshold minimizes total expected loss")
    lines.append("")

    # Write report
    report_path = ensure_dir(output_dir) / "model_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path
