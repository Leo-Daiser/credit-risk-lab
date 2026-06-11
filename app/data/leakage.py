"""Leakage detection utilities."""
import numpy as np
import pandas as pd

from app.constants import LEAKAGE_KEYWORDS


def detect_leakage_columns(df: pd.DataFrame, target_col: str = "target") -> list[str]:
    """Detect columns with names suggesting leakage.

    Args:
        df: Input DataFrame.
        target_col: Target column name.

    Returns:
        List of suspicious column names.
    """
    suspicious = []
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower()
        for keyword in LEAKAGE_KEYWORDS:
            if keyword in col_lower:
                suspicious.append(col)
                break
    return suspicious


def detect_suspicious_target_correlation(
    df: pd.DataFrame,
    target_col: str = "target",
    threshold: float = 0.95,
) -> list[dict]:
    """Detect numeric columns with suspiciously high correlation with target.

    Args:
        df: Input DataFrame.
        target_col: Target column name.
        threshold: Correlation threshold.

    Returns:
        List of dicts with column name and correlation.
    """
    if target_col not in df.columns:
        return []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    suspicious = []
    for col in numeric_cols:
        if df[col].nunique() < 2:
            continue
        corr = abs(df[col].corr(df[target_col]))
        if corr > threshold:
            suspicious.append({"column": col, "correlation": float(corr)})

    return suspicious


def detect_perfect_separation(
    df: pd.DataFrame,
    target_col: str = "target",
) -> list[str]:
    """Detect categorical columns that perfectly separate target.

    Args:
        df: Input DataFrame.
        target_col: Target column name.

    Returns:
        List of suspicious column names.
    """
    if target_col not in df.columns:
        return []

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    suspicious = []

    for col in categorical_cols:
        if col == target_col:
            continue
        for val in df[col].unique():
            subset = df[df[col] == val]
            if len(subset) > 0 and subset[target_col].nunique() == 1:
                suspicious.append(col)
                break

    return suspicious


def generate_leakage_report(df: pd.DataFrame, target_col: str = "target") -> dict:
    """Generate comprehensive leakage report.

    Args:
        df: Input DataFrame.
        target_col: Target column name.

    Returns:
        Dictionary with leakage analysis results.
    """
    report = {
        "suspicious_columns": detect_leakage_columns(df, target_col),
        "high_correlation": detect_suspicious_target_correlation(df, target_col),
        "perfect_separation": detect_perfect_separation(df, target_col),
    }

    report["has_leakage"] = any([
        report["suspicious_columns"],
        report["high_correlation"],
        report["perfect_separation"],
    ])

    return report


def format_leakage_report(report: dict) -> str:
    """Format leakage report as markdown.

    Args:
        report: Leakage report dictionary.

    Returns:
        Formatted markdown string.
    """
    lines = ["# Data Leakage Report\n"]

    if report["has_leakage"]:
        lines.append("**WARNING: Potential data leakage detected!**\n")
    else:
        lines.append("**No significant data leakage detected.**\n")

    if report["suspicious_columns"]:
        lines.append("## Suspicious Column Names\n")
        for col in report["suspicious_columns"]:
            lines.append(f"- `{col}`")
        lines.append("")

    if report["high_correlation"]:
        lines.append("## High Correlation with Target\n")
        for item in report["high_correlation"]:
            lines.append(f"- `{item['column']}`: r = {item['correlation']:.4f}")
        lines.append("")

    if report["perfect_separation"]:
        lines.append("## Perfect Target Separation\n")
        for col in report["perfect_separation"]:
            lines.append(f"- `{col}`")
        lines.append("")

    return "\n".join(lines)
