"""Synthetic credit data generator with realistic feature-target relationships."""
import numpy as np
import pandas as pd

from app.utils.random import get_random_state


def generate_synthetic_data(
    n_rows: int = 5000,
    random_state: int | None = None,
    default_rate_target: float = 0.35,
    noise_level: float = 0.05,
    missing_rate: float = 0.02,
    outlier_rate: float = 0.01,
) -> pd.DataFrame:
    """Generate synthetic credit scoring dataset with logical target dependency.

    Target (default) depends on:
    - Higher risk with high debt_to_income
    - Higher risk with previous_defaults
    - Higher risk with late_payments_12m
    - Higher risk with low income
    - Higher risk with short employment
    - Higher risk with short credit history
    - Higher risk with high interest_rate
    - Higher risk with rent home ownership
    - Lower risk with has_cosigner
    - Lower risk with high savings_balance
    - Lower risk with own home ownership
    """
    seed = get_random_state(random_state)
    rng = np.random.RandomState(seed)

    n = n_rows

    data = {
        "applicant_id": [f"APP_{i:06d}" for i in range(n)],
        "age": rng.randint(21, 70, size=n),
        "income": rng.lognormal(mean=10.5, sigma=0.5, size=n).clip(15000, 300000),
        "employment_years": rng.exponential(scale=5, size=n).clip(0, 40),
        "loan_amount": rng.lognormal(mean=10, sigma=0.8, size=n).clip(1000, 200000),
        "loan_term_months": rng.choice([12, 24, 36, 48, 60, 72, 84], size=n),
        "interest_rate": rng.uniform(3.0, 25.0, size=n),
        "debt_to_income": rng.beta(a=2, b=5, size=n),
        "credit_history_years": rng.exponential(scale=8, size=n).clip(0, 40),
        "previous_defaults": rng.poisson(lam=0.3, size=n).clip(0, 5),
        "open_credit_lines": rng.poisson(lam=3, size=n).clip(0, 15),
        "late_payments_12m": rng.poisson(lam=0.5, size=n).clip(0, 10),
        "home_ownership": rng.choice(["rent", "mortgage", "own", "other"], size=n, p=[0.35, 0.40, 0.20, 0.05]),
        "education": rng.choice(["school", "college", "bachelor", "master", "phd"], size=n, p=[0.15, 0.25, 0.35, 0.20, 0.05]),
        "purpose": rng.choice(["car", "education", "business", "home", "medical", "other"], size=n, p=[0.20, 0.15, 0.20, 0.25, 0.10, 0.10]),
        "region": rng.choice(["north", "south", "east", "west", "central"], size=n),
        "application_channel": rng.choice(["online", "branch", "partner", "referral"], size=n, p=[0.40, 0.30, 0.20, 0.10]),
        "has_cosigner": rng.choice([True, False], size=n, p=[0.25, 0.75]),
        "savings_balance": rng.lognormal(mean=9, sigma=1.5, size=n).clip(0, 500000),
        "checking_balance": rng.lognormal(mean=8, sigma=1.2, size=n).clip(0, 200000),
    }

    df = pd.DataFrame(data)

    # Calculate risk score based on features
    risk_score = (
        0.25 * _normalize(df["debt_to_income"]) +
        0.20 * _normalize(df["previous_defaults"].astype(float)) +
        0.15 * _normalize(df["late_payments_12m"].astype(float)) +
        0.15 * (1 - _normalize(df["income"])) +
        0.10 * (1 - _normalize(df["credit_history_years"])) +
        0.08 * (1 - _normalize(df["employment_years"])) +
        0.05 * _normalize(df["interest_rate"]) +
        0.02 * (df["home_ownership"] == "rent").astype(float) -
        0.05 * df["has_cosigner"].astype(float) -
        0.03 * _normalize(df["savings_balance"])
    )

    # Add noise
    noise = rng.normal(0, noise_level, size=n)
    risk_score = risk_score + noise

    # Convert to probability using sigmoid, calibrated to target default rate
    prob_default = _sigmoid(risk_score * 4 - 1.5)

    # Calibrate to target default rate
    current_rate = prob_default.mean()
    if current_rate > 0:
        adjustment = default_rate_target / current_rate
        prob_default = np.clip(prob_default * adjustment, 0.01, 0.99)

    # Sample target
    df["target"] = rng.binomial(1, prob_default)

    # Add controlled missing values
    if missing_rate > 0:
        for col in ["income", "employment_years", "savings_balance", "checking_balance"]:
            mask = rng.random(n) < missing_rate
            df.loc[mask, col] = np.nan

    # Add controlled outliers
    if outlier_rate > 0:
        n_outliers = max(1, int(n * outlier_rate))
        outlier_idx = rng.choice(n, size=n_outliers, replace=False)
        df.loc[outlier_idx, "income"] = df["income"].max() * 3

    return df


def _normalize(series: pd.Series) -> pd.Series:
    """Normalize series to [0, 1] range."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))
