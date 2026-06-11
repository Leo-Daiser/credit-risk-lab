"""Feature engineering utilities."""
import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataset.

    Features added:
    - loan_to_income: loan_amount / income
    - credit_lines_per_history_year: open_credit_lines / max(credit_history_years, 1)
    - late_payment_rate: late_payments_12m / max(open_credit_lines, 1)
    - savings_to_loan: savings_balance / loan_amount
    - checking_to_loan: checking_balance / loan_amount
    - employment_stability_bucket: categorical bucket for employment_years
    - dti_bucket: categorical bucket for debt_to_income
    - income_bucket: categorical bucket for income
    - credit_history_bucket: categorical bucket for credit_history_years
    - high_risk_purpose_flag: 1 if purpose in [business, medical]
    - has_prior_credit_problem: 1 if previous_defaults > 0 or late_payments_12m > 0
    """
    df = df.copy()

    # Safe division function
    def safe_divide(num, den):
        return np.where(den != 0, num / den, 0.0)

    # Loan to income ratio
    if "loan_amount" in df.columns and "income" in df.columns:
        df["loan_to_income"] = safe_divide(df["loan_amount"].fillna(0), df["income"].fillna(1))

    # Credit lines per year of credit history
    if "open_credit_lines" in df.columns and "credit_history_years" in df.columns:
        credit_history = df["credit_history_years"].fillna(0).clip(lower=1)
        df["credit_lines_per_history_year"] = safe_divide(df["open_credit_lines"].fillna(0), credit_history)

    # Late payment rate
    if "late_payments_12m" in df.columns and "open_credit_lines" in df.columns:
        open_lines = df["open_credit_lines"].fillna(0).clip(lower=1)
        df["late_payment_rate"] = safe_divide(df["late_payments_12m"].fillna(0), open_lines)

    # Savings to loan ratio
    if "savings_balance" in df.columns and "loan_amount" in df.columns:
        df["savings_to_loan"] = safe_divide(df["savings_balance"].fillna(0), df["loan_amount"].fillna(1))

    # Checking to loan ratio
    if "checking_balance" in df.columns and "loan_amount" in df.columns:
        df["checking_to_loan"] = safe_divide(df["checking_balance"].fillna(0), df["loan_amount"].fillna(1))

    # Employment stability bucket
    if "employment_years" in df.columns:
        df["employment_stability_bucket"] = pd.cut(
            df["employment_years"].fillna(0),
            bins=[-1, 1, 3, 7, 15, 100],
            labels=["unstable", "low", "medium", "stable", "very_stable"],
            include_lowest=True,
        )

    # DTI bucket
    if "debt_to_income" in df.columns:
        df["dti_bucket"] = pd.cut(
            df["debt_to_income"].fillna(0),
            bins=[0, 0.2, 0.4, 0.6, 1.0],
            labels=["low", "medium", "high", "extreme"],
            include_lowest=True,
        )

    # Income bucket
    if "income" in df.columns:
        df["income_bucket"] = pd.cut(
            df["income"].fillna(0),
            bins=[0, 30000, 60000, 100000, 200000, 500000],
            labels=["very_low", "low", "medium", "high", "very_high"],
            include_lowest=True,
        )

    # Credit history bucket
    if "credit_history_years" in df.columns:
        df["credit_history_bucket"] = pd.cut(
            df["credit_history_years"].fillna(0),
            bins=[-1, 1, 3, 7, 15, 100],
            labels=["none", "short", "medium", "long", "very_long"],
            include_lowest=True,
        )

    # High risk purpose flag
    if "purpose" in df.columns:
        high_risk_purposes = ["business", "medical"]
        df["high_risk_purpose_flag"] = df["purpose"].isin(high_risk_purposes).astype(int)

    # Has prior credit problem
    if "previous_defaults" in df.columns and "late_payments_12m" in df.columns:
        df["has_prior_credit_problem"] = (
            (df["previous_defaults"].fillna(0) > 0) | (df["late_payments_12m"].fillna(0) > 0)
        ).astype(int)

    return df


def get_engineered_feature_names() -> list[str]:
    """Return names of engineered features."""
    return [
        "loan_to_income",
        "credit_lines_per_history_year",
        "late_payment_rate",
        "savings_to_loan",
        "checking_to_loan",
        "employment_stability_bucket",
        "dti_bucket",
        "income_bucket",
        "credit_history_bucket",
        "high_risk_purpose_flag",
        "has_prior_credit_problem",
    ]
