"""Data loading utilities."""
from pathlib import Path

import pandas as pd

from app.utils.io import load_csv


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load dataset from CSV file.

    Args:
        path: Path to CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty or cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = load_csv(path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return df


def get_feature_columns(df: pd.DataFrame, target: str = "target") -> tuple[list[str], list[str]]:
    """Separate features and target column names.

    Args:
        df: Input DataFrame.
        target: Name of target column.

    Returns:
        Tuple of (feature_columns, target_column).
    """
    feature_cols = [col for col in df.columns if col != target]
    return feature_cols, target


def split_features_types(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical based on config.

    Args:
        df: Input DataFrame.
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names.

    Returns:
        Tuple of (actual_numeric, actual_categorical) columns present in df.
    """
    actual_numeric = [col for col in numeric_cols if col in df.columns]
    actual_categorical = [col for col in categorical_cols if col in df.columns]
    return actual_numeric, actual_categorical
