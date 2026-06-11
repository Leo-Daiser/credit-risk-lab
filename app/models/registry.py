"""Model registry with hyperparameter search spaces."""

from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.config import get_config
from app.features.preprocessing import build_preprocessing_pipeline


MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}

MODEL_PARAMS = {
    "logistic_regression": {
        "class_weight": ["balanced"],
        "max_iter": [5000],
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["saga"],
    },
    "random_forest": {
        "class_weight": ["balanced"],
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "hist_gradient_boosting": {
        "max_iter": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [1, 5, 10],
    },
}

# Minimal search spaces for fast mode
FAST_MODEL_PARAMS = {
    "logistic_regression": {
        "class_weight": ["balanced"],
        "max_iter": [5000],
        "C": [1.0],
        "penalty": ["l2"],
        "solver": ["saga"],
    },
    "random_forest": {
        "class_weight": ["balanced"],
        "n_estimators": [100],
        "max_depth": [10],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
    },
    "gradient_boosting": {
        "n_estimators": [100],
        "learning_rate": [0.1],
        "max_depth": [3],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
    },
    "hist_gradient_boosting": {
        "max_iter": [100],
        "learning_rate": [0.1],
        "max_depth": [3],
        "min_samples_leaf": [5],
    },
}


def get_model_specs(config=None) -> dict[str, dict]:
    """Get model specifications from config.

    Args:
        config: App configuration.

    Returns:
        Dictionary of model_name -> specs.
    """
    if config is None:
        config = get_config()

    specs = {}
    for name in config.training.models:
        if name in MODEL_CLASSES:
            specs[name] = {
                "class": MODEL_CLASSES[name],
                "default_params": config.models.__dict__.get(name, {}),
            }
    return specs


def build_model_pipeline(
    model_name: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config=None,
    use_scaling: bool = True,
) -> Pipeline:
    """Build a complete model pipeline with preprocessing.

    Args:
        model_name: Name of the model.
        numeric_cols: Numeric feature columns.
        categorical_cols: Categorical feature columns.
        config: App configuration.
        use_scaling: Whether to use StandardScaler.

    Returns:
        Pipeline with preprocessing and model.
    """
    if config is None:
        config = get_config()

    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}")

    # Build preprocessor
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols, scale_numeric=use_scaling)

    # Get model class and default params
    model_class = MODEL_CLASSES[model_name]
    default_params = config.models.__dict__.get(model_name, {})

    if model_name in ["logistic_regression", "random_forest"]:
        default_params["random_state"] = config.training.random_state

    model = model_class(**default_params)

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def get_search_space(model_name: str, fast_mode: bool = False) -> dict[str, list]:
    """Get hyperparameter search space for a model.

    Args:
        model_name: Name of the model.
        fast_mode: Use minimal search space.

    Returns:
        Dictionary of parameter_name -> list of values.
    """
    if fast_mode:
        return FAST_MODEL_PARAMS.get(model_name, {})
    return MODEL_PARAMS.get(model_name, {})


def get_available_models() -> list[str]:
    """Get list of available model names."""
    return list(MODEL_CLASSES.keys())
