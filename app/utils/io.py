"""Utility functions for I/O operations."""
import json
from pathlib import Path

import joblib
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save DataFrame to CSV."""
    p = ensure_dir(Path(path).parent) / Path(path).name
    df.to_csv(p, index=False)
    return p


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV to DataFrame."""
    return pd.read_csv(path)


def save_json(data: dict, path: str | Path) -> Path:
    """Save dictionary to JSON."""
    p = ensure_dir(Path(path).parent) / Path(path).name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return p


def load_json(path: str | Path) -> dict:
    """Load JSON to dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model(model: object, path: str | Path) -> Path:
    """Save model using joblib."""
    p = ensure_dir(Path(path).parent) / Path(path).name
    joblib.dump(model, p)
    return p


def load_model(path: str | Path) -> object:
    """Load model using joblib."""
    return joblib.load(path)
