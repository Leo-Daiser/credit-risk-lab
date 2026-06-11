"""Baseline model definitions."""
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.utils.random import get_random_state


def get_logistic_regression(random_state: int | None = None) -> LogisticRegression:
    """Get Logistic Regression baseline model."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=get_random_state(random_state),
    )


def get_random_forest(random_state: int | None = None) -> RandomForestClassifier:
    """Get Random Forest model."""
    return RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        random_state=get_random_state(random_state),
    )


def get_gradient_boosting(random_state: int | None = None) -> GradientBoostingClassifier:
    """Get Gradient Boosting model."""
    return GradientBoostingClassifier(
        n_estimators=200,
        random_state=get_random_state(random_state),
    )


def get_xgboost(random_state: int | None = None) -> Any | None:
    """Get XGBoost model if available."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200,
            random_state=get_random_state(random_state),
            eval_metric="logloss",
            use_label_encoder=False,
        )
    except ImportError:
        return None


def get_lightgbm(random_state: int | None = None) -> Any | None:
    """Get LightGBM model if available."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200,
            random_state=get_random_state(random_state),
            verbose=-1,
        )
    except ImportError:
        return None


def get_all_models(random_state: int | None = None) -> dict[str, Any]:
    """Get all available models.

    Returns:
        Dictionary of model_name -> model_instance.
    """
    models = {
        "logistic_regression": get_logistic_regression(random_state),
        "random_forest": get_random_forest(random_state),
        "gradient_boosting": get_gradient_boosting(random_state),
    }

    # Add optional models if available
    xgb = get_xgboost(random_state)
    if xgb is not None:
        models["xgboost"] = xgb

    lgbm = get_lightgbm(random_state)
    if lgbm is not None:
        models["lightgbm"] = lgbm

    return models
