"""Data validation utilities."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import get_config
from app.constants import (
    MAX_AGE,
    MAX_DEBT_TO_INCOME,
    MIN_AGE,
    MIN_CREDIT_HISTORY_YEARS,
    MIN_DEBT_TO_INCOME,
    MIN_EMPLOYMENT_YEARS,
    MIN_INCOME,
    MIN_LOAN_AMOUNT,
)
from app.utils.io import ensure_dir, save_json
from app.utils.logging import logger


def validate_dataset(df: pd.DataFrame, path: str | Path = "", config=None) -> dict:
    """Validate dataset for quality issues.

    Checks:
    - Target column presence
    - Target is binary
    - Missing values
    - Column types
    - Value ranges
    - Impossible values
    - Class balance
    - Duplicate rows
    - Duplicate applicant_id
    - Suspicious leakage features
    - Constant columns
    """
    if config is None:
        config = get_config()

    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": [],
        "stats": {},
    }

    # Check target presence
    target = config.data.target_column
    if target not in df.columns:
        results["valid"] = False
        results["errors"].append(f"Target column '{target}' not found")
        return results

    # Check target is binary
    unique_targets = df[target].dropna().unique()
    if not set(unique_targets).issubset({0, 1}):
        results["valid"] = False
        results["errors"].append(f"Target is not binary. Unique values: {unique_targets}")

    # Check duplicate rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        dup_pct = n_duplicates / len(df) * 100
        results["warnings"].append(f"Found {n_duplicates} duplicate rows ({dup_pct:.1f}%)")
        results["stats"]["n_duplicates"] = int(n_duplicates)

    # Check duplicate applicant_id
    id_col = config.data.id_column
    if id_col in df.columns:
        n_dup_ids = df[id_col].duplicated().sum()
        if n_dup_ids > 0:
            results["errors"].append(f"Found {n_dup_ids} duplicate {id_col}")
            results["valid"] = False
        results["stats"]["n_unique_ids"] = int(df[id_col].nunique())

    # Check missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    results["stats"]["missing_values"] = {k: int(v) for k, v in missing[missing > 0].items()}
    results["stats"]["missing_pct"] = {k: float(v) for k, v in missing_pct[missing_pct > 0].items()}

    if missing.any():
        cols_with_missing = missing[missing > 0].index.tolist()
        max_missing = missing_pct.max()
        if max_missing > config.validation.max_missing_share * 100:
            results["errors"].append(f"Too many missing values in {cols_with_missing}")
            results["valid"] = False
        else:
            results["warnings"].append(f"Columns with missing values: {cols_with_missing}")

    # Check column types
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    results["stats"]["numeric_columns"] = numeric_cols
    results["stats"]["categorical_columns"] = categorical_cols

    # Check impossible values
    impossible_checks = {
        "age": (MIN_AGE, MAX_AGE),
        "income": (MIN_INCOME, None),
        "loan_amount": (MIN_LOAN_AMOUNT, None),
        "employment_years": (MIN_EMPLOYMENT_YEARS, None),
        "credit_history_years": (MIN_CREDIT_HISTORY_YEARS, None),
        "debt_to_income": (MIN_DEBT_TO_INCOME, MAX_DEBT_TO_INCOME),
    }

    for col, (min_val, max_val) in impossible_checks.items():
        if col in df.columns:
            col_data = df[col].dropna()
            if min_val is not None and (col_data < min_val).any():
                n_violations = (col_data < min_val).sum()
                results["warnings"].append(f"{col}: {n_violations} values below {min_val}")
            if max_val is not None and (col_data > max_val).any():
                n_violations = (col_data > max_val).sum()
                results["warnings"].append(f"{col}: {n_violations} values above {max_val}")

    # Check value ranges
    for col in numeric_cols:
        if col == target:
            continue
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        results["stats"][f"{col}_range"] = {"min": col_min, "max": col_max}

    # Check class balance
    target_counts = df[target].value_counts()
    target_pct = (target_counts / len(df) * 100).round(2)
    results["stats"]["class_distribution"] = target_counts.to_dict()
    results["stats"]["class_percentage"] = target_pct.to_dict()

    imbalance_ratio = target_counts.max() / target_counts.min() if target_counts.min() > 0 else float("inf")
    if imbalance_ratio > 10:
        results["warnings"].append(f"Severe class imbalance: ratio={imbalance_ratio:.1f}")

    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        results["warnings"].append(f"Constant columns: {constant_cols}")

    # Check for high cardinality
    for col in categorical_cols:
        if col == target:
            continue
        n_unique = df[col].nunique()
        if n_unique > 50:
            results["warnings"].append(f"High cardinality in {col}: {n_unique} unique values")

    # Check for leakage (features that perfectly predict target)
    leakage_candidates = []
    for col in numeric_cols:
        if col == target:
            continue
        if df[col].nunique() > 1 and df.groupby(col)[target].nunique().max() == 1:
            for val in df[col].unique():
                subset = df[df[col] == val]
                if len(subset) > 0 and subset[target].nunique() == 1:
                    leakage_candidates.append(col)
                    break

    if leakage_candidates:
        results["warnings"].append(f"Potential leakage features: {leakage_candidates}")
        results["stats"]["leakage_candidates"] = leakage_candidates

    results["stats"]["n_rows"] = len(df)
    results["stats"]["n_columns"] = len(df.columns)

    return results


def format_validation_report(results: dict) -> str:
    """Format validation results as markdown report.

    Args:
        results: Validation results dictionary.

    Returns:
        Formatted markdown string.
    """
    lines = ["# Data Validation Report\n"]

    if results["valid"]:
        lines.append("## Status: VALID\n")
    else:
        lines.append("## Status: INVALID\n")

    if results["errors"]:
        lines.append("## Errors\n")
        for err in results["errors"]:
            lines.append(f"- **ERROR**: {err}")
        lines.append("")

    if results["warnings"]:
        lines.append("## Warnings\n")
        for warn in results["warnings"]:
            lines.append(f"- {warn}")
        lines.append("")

    stats = results["stats"]
    if "class_distribution" in stats:
        lines.append("## Class Distribution\n")
        lines.append("| Class | Count | Percentage |")
        lines.append("|-------|-------|------------|")
        for cls in sorted(stats["class_distribution"].keys()):
            count = stats["class_distribution"][cls]
            pct = stats["class_percentage"][cls]
            lines.append(f"| {cls} | {count} | {pct}% |")
        lines.append("")

    if "missing_values" in stats and stats["missing_values"]:
        lines.append("## Missing Values\n")
        lines.append("| Column | Missing Count | Percentage |")
        lines.append("|--------|---------------|------------|")
        for col, count in stats["missing_values"].items():
            pct = stats["missing_pct"][col]
            lines.append(f"| {col} | {count} | {pct}% |")
        lines.append("")

    lines.append("## Summary\n")
    lines.append(f"- Rows: {stats.get('n_rows', 'N/A')}")
    lines.append(f"- Columns: {stats.get('n_columns', 'N/A')}")
    lines.append(f"- Duplicates: {stats.get('n_duplicates', 0)}")
    lines.append("")

    return "\n".join(lines)


def save_validation_report(results: dict, output_dir: str = "artifacts") -> tuple[Path, Path]:
    """Save validation report as markdown and JSON.

    Args:
        results: Validation results dictionary.
        output_dir: Output directory.

    Returns:
        Tuple of (markdown_path, json_path).
    """
    report_dir = Path(output_dir) / "reports"
    metrics_dir = Path(output_dir) / "metrics"

    ensure_dir(report_dir)
    ensure_dir(metrics_dir)

    # Save markdown report
    md_report = format_validation_report(results)
    md_path = report_dir / "data_validation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    # Save JSON metrics
    json_path = metrics_dir / "data_validation.json"
    save_json(results, json_path)

    return md_path, json_path
