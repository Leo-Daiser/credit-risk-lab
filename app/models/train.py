"""Model training utilities."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import get_config
from app.features.feature_engineering import add_engineered_features
from app.features.preprocessing import build_preprocessing_pipeline, get_feature_names_after_preprocessing
from app.models.baseline import get_all_models
from app.utils.io import save_model
from app.utils.logging import logger


def prepare_data(df: pd.DataFrame, config=None) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Prepare data for training.

    Args:
        df: Input DataFrame.
        config: App configuration.

    Returns:
        Tuple of (X, y, numeric_cols, categorical_cols).
    """
    if config is None:
        config = get_config()

    target = config.features.target
    y = df[target]
    X = df.drop(columns=[target])

    # Add engineered features
    X = add_engineered_features(X)

    # Identify column types
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return X, y, numeric_cols, categorical_cols


def train_models(df: pd.DataFrame, config=None) -> dict:
    """Train all models and return results.

    Args:
        df: Input DataFrame.
        config: App configuration.

    Returns:
        Dictionary with trained models, metrics, and preprocessing info.
    """
    if config is None:
        config = get_config()

    seed = config.training.random_state

    # Prepare data
    X, y, numeric_cols, categorical_cols = prepare_data(df, config)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.training.test_size,
        random_state=seed,
        stratify=y,
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        numeric_cols,
        categorical_cols,
        scale_numeric=config.preprocessing.scaling,
    )

    # Get feature names before fitting
    all_numeric = numeric_cols.copy()
    all_categorical = categorical_cols.copy()

    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    feature_names = get_feature_names_after_preprocessing(preprocessor, all_numeric, all_categorical)

    # Train all models
    models = get_all_models(seed)
    results = {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "models": {},
    }

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_processed, y_train)
        results["models"][name] = model
        logger.info(f"  {name} trained successfully")

    return results


def select_best_model(results: dict) -> tuple[str, object]:
    """Select best model based on PR-AUC.

    Args:
        results: Training results dictionary.

    Returns:
        Tuple of (model_name, model_instance).
    """
    from sklearn.metrics import average_precision_score

    best_score = -1
    best_name = ""

    X_test = results["X_test_processed"]
    y_test = results["y_test"]

    for name, model in results["models"].items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, y_pred_proba)
        logger.info(f"  {name} PR-AUC: {pr_auc:.4f}")

        if pr_auc > best_score:
            best_score = pr_auc
            best_name = name

    logger.info(f"Best model: {best_name} (PR-AUC: {best_score:.4f})")
    return best_name, results["models"][best_name]


def save_training_artifacts(results: dict, best_name: str, best_model: object, output_dir: str = "artifacts") -> dict:
    """Save training artifacts (models, preprocessor, metadata).

    Args:
        results: Training results dictionary.
        best_name: Name of best model.
        best_model: Best model instance.
        output_dir: Output directory.

    Returns:
        Dictionary with saved artifact paths.
    """
    from app.utils.io import save_json

    output_path = Path(output_dir)
    models_dir = output_path
    models_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # Save preprocessor
    preprocessor_path = models_dir / "preprocessor.joblib"
    save_model(results["preprocessor"], preprocessor_path)
    artifacts["preprocessor"] = str(preprocessor_path)

    # Save best model
    model_path = models_dir / f"{best_name}.joblib"
    save_model(best_model, model_path)
    artifacts["best_model"] = str(model_path)

    # Save all models
    for name, model in results["models"].items():
        model_path = models_dir / f"{name}.joblib"
        save_model(model, model_path)
        artifacts[f"model_{name}"] = str(model_path)

    # Save metadata
    metadata = {
        "best_model": best_name,
        "feature_names": results["feature_names"],
        "numeric_cols": results["numeric_cols"],
        "categorical_cols": results["categorical_cols"],
        "n_features": len(results["feature_names"]),
        "train_size": len(results["X_train"]),
        "test_size": len(results["X_test"]),
    }
    metadata_path = models_dir / "metadata.json"
    save_json(metadata, metadata_path)
    artifacts["metadata"] = str(metadata_path)

    return artifacts
