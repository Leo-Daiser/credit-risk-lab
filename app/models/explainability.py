"""Model explainability utilities."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from app.constants import REASON_THRESHOLDS
from app.utils.logging import logger


def compute_permutation_importance(
    model: object,
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        feature_names: Feature names.
        n_repeats: Number of repeats.
        random_state: Random state.

    Returns:
        DataFrame with feature importances.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="average_precision",
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return importance_df


def get_logistic_regression_coefficients(
    model: object,
    feature_names: list[str],
) -> pd.DataFrame:
    """Get Logistic Regression coefficients.

    Args:
        model: Trained LogisticRegression model.
        feature_names: Feature names.

    Returns:
        DataFrame with coefficients.
    """
    if not hasattr(model, "coef_"):
        logger.warning("Model does not have coef_ attribute")
        return pd.DataFrame()

    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        "feature": feature_names[:len(coefs)],
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    return coef_df


def generate_reason_codes(
    row: pd.Series,
    feature_names: list[str],
    model: object = None,
    preprocessor: object = None,
    top_n: int = 5,
) -> list[str]:
    """Generate reason codes for a single prediction.

    Reason codes explain why a client was flagged as high risk.

    Args:
        row: Single row of features.
        feature_names: Feature names.
        model: Trained model (optional, not used in rule-based approach).
        preprocessor: Fitted preprocessor (optional, not used).
        top_n: Number of top reasons to return.

    Returns:
        List of reason code strings.
    """
    reasons = []

    # Check high DTI
    if "debt_to_income" in row.index and pd.notna(row["debt_to_income"]):
        if row["debt_to_income"] > REASON_THRESHOLDS["high_debt_to_income"]:
            reasons.append("high_debt_to_income")

    # Check previous defaults
    if "previous_defaults" in row.index and pd.notna(row["previous_defaults"]):
        if row["previous_defaults"] > 0:
            reasons.append("previous_defaults")

    # Check late payments
    if "late_payments_12m" in row.index and pd.notna(row["late_payments_12m"]):
        if row["late_payments_12m"] > REASON_THRESHOLDS["recent_late_payments"]:
            reasons.append("recent_late_payments")

    # Check low income
    if "income" in row.index and pd.notna(row["income"]):
        if row["income"] < REASON_THRESHOLDS["low_income"]:
            reasons.append("low_income")

    # Check short credit history
    if "credit_history_years" in row.index and pd.notna(row["credit_history_years"]):
        if row["credit_history_years"] < REASON_THRESHOLDS["short_credit_history"]:
            reasons.append("short_credit_history")

    # Check high loan to income
    if "loan_to_income" in row.index and pd.notna(row["loan_to_income"]):
        if row["loan_to_income"] > REASON_THRESHOLDS["high_loan_to_income"]:
            reasons.append("high_loan_to_income")

    # Check unstable employment
    if "employment_years" in row.index and pd.notna(row["employment_years"]):
        if row["employment_years"] < REASON_THRESHOLDS["short_employment"]:
            reasons.append("short_employment")

    # Check high interest rate
    if "interest_rate" in row.index and pd.notna(row["interest_rate"]):
        if row["interest_rate"] > REASON_THRESHOLDS["high_interest_rate"]:
            reasons.append("high_interest_rate")

    # Check low savings buffer
    if "savings_balance" in row.index and pd.notna(row["savings_balance"]):
        if row["savings_balance"] < REASON_THRESHOLDS["low_savings_buffer"]:
            reasons.append("low_savings_buffer")

    # Check high risk purpose
    if "purpose" in row.index and pd.notna(row["purpose"]):
        if row["purpose"] in REASON_THRESHOLDS["high_risk_purpose"]:
            reasons.append("high_risk_purpose")

    return reasons[:top_n] if reasons else ["low_risk_profile"]
