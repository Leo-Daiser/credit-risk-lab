"""EDA report generation."""
from pathlib import Path

import pandas as pd

from app.config import get_config
from app.utils.io import ensure_dir
from app.utils.paths import get_reports_dir


def generate_eda_report(df: pd.DataFrame, config=None) -> Path:
    """Generate EDA report in Markdown format.

    Args:
        df: Input DataFrame.
        config: App configuration.

    Returns:
        Path to saved report.
    """
    if config is None:
        config = get_config()

    target = config.features.target

    lines = ["# Exploratory Data Analysis Report\n"]

    # Dataset overview
    lines.append("## Dataset Overview\n")
    lines.append(f"- **Rows**: {len(df)}")
    lines.append(f"- **Columns**: {len(df.columns)}")
    lines.append(f"- **Target column**: {target}")
    lines.append("")

    # Class distribution
    lines.append("## Class Distribution\n")
    target_counts = df[target].value_counts()
    target_pct = (target_counts / len(df) * 100).round(2)
    lines.append("| Class | Count | Percentage |")
    lines.append("|-------|-------|------------|")
    for cls in sorted(target_counts.index):
        lines.append(f"| {cls} | {target_counts[cls]} | {target_pct[cls]}% |")
    lines.append("")

    # Missing values
    lines.append("## Missing Values\n")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        lines.append("| Column | Missing Count | Percentage |")
        lines.append("|--------|---------------|------------|")
        for col in missing.index:
            pct = (missing[col] / len(df) * 100).round(2)
            lines.append(f"| {col} | {missing[col]} | {pct}% |")
    else:
        lines.append("No missing values found.")
    lines.append("")

    # Numeric features
    lines.append("## Numeric Features Summary\n")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    lines.append("| Feature | Mean | Std | Min | Max |")
    lines.append("|---------|------|-----|-----|-----|")
    for col in numeric_cols:
        lines.append(f"| {col} | {df[col].mean():.2f} | {df[col].std():.2f} | {df[col].min():.2f} | {df[col].max():.2f} |")
    lines.append("")

    # Categorical features
    lines.append("## Categorical Features Summary\n")
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in categorical_cols:
        lines.append(f"### {col}\n")
        value_counts = df[col].value_counts()
        lines.append("| Value | Count | Percentage |")
        lines.append("|-------|-------|------------|")
        for val in value_counts.index:
            count = value_counts[val]
            pct = (count / len(df) * 100).round(2)
            lines.append(f"| {val} | {count} | {pct}% |")
        lines.append("")

    # Target rate by categories
    lines.append("## Target Rate by Categories\n")
    for col in categorical_cols:
        lines.append(f"### {col}\n")
        target_rate = df.groupby(col)[target].mean().round(4)
        lines.append("| Value | Default Rate |")
        lines.append("|-------|--------------|")
        for val in target_rate.index:
            lines.append(f"| {val} | {target_rate[val]:.4f} |")
        lines.append("")

    # Correlation with target
    lines.append("## Feature Correlation with Target\n")
    correlations = df[numeric_cols + [target]].corr()[target].drop(target).sort_values(ascending=False)
    lines.append("| Feature | Correlation |")
    lines.append("|---------|-------------|")
    for feat in correlations.index:
        lines.append(f"| {feat} | {correlations[feat]:.4f} |")
    lines.append("")

    # Write report
    report_dir = get_reports_dir(config)
    report_path = report_dir / "eda_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path
