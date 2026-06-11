"""Prediction module."""
import pandas as pd

from app.config import get_config
from app.features.feature_engineering import add_engineered_features
from app.models.explainability import generate_reason_codes
from app.utils.io import load_csv, load_json, load_model
from app.utils.logging import logger


def predict_from_csv(
    input_path: str,
    output_path: str,
    config=None,
) -> pd.DataFrame:
    """Make predictions for new data from CSV.

    Args:
        input_path: Path to input CSV.
        output_path: Path to output CSV.
        config: App configuration.

    Returns:
        DataFrame with predictions.
    """
    if config is None:
        config = get_config()

    # Load model artifacts
    models_dir = config.paths.artifacts_models
    metadata_path = f"{models_dir}/metadata.json"

    metadata = load_json(metadata_path)
    model = load_model(f"{models_dir}/{metadata['best_model']}.joblib")
    preprocessor = load_model(f"{models_dir}/preprocessor.joblib")

    # Load data
    df = load_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    # Store applicant_id if present
    applicant_ids = df.get("applicant_id", pd.Series([f"APP_{i:06d}" for i in range(len(df))]))

    # Drop target if present
    target_col = config.data.target_column
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Drop applicant_id for features
    id_col = config.data.id_column
    if id_col in df.columns:
        df_features = df.drop(columns=[id_col])
    else:
        df_features = df.copy()

    # Add engineered features
    df_features = add_engineered_features(df_features)

    # Preprocess
    X = preprocessor.transform(df_features)

    # Predict
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Get thresholds from config
    low_threshold = config.threshold.manual_review_low
    high_threshold = config.threshold.manual_review_high

    # Create results
    results = pd.DataFrame({
        "applicant_id": applicant_ids,
        "probability_default": y_pred_proba,
    })

    # Add decisions
    results["decision"] = results["probability_default"].apply(
        lambda p: "approve" if p < low_threshold else ("reject" if p >= high_threshold else "manual_review")
    )

    # Add risk buckets
    results["risk_bucket"] = pd.cut(
        results["probability_default"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["very_low", "low", "medium", "high", "very_high"],
        include_lowest=True,
    )

    # Add reason codes
    results["reason_codes"] = [
        generate_reason_codes(row, df_features.columns.tolist())
        for _, row in df_features.iterrows()
    ]
    results["reason_codes"] = results["reason_codes"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )

    # Save
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

    return results
