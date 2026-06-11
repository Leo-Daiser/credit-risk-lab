"""Preprocessing pipeline using sklearn."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import CONFIG


def build_preprocessing_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = True,
) -> Pipeline:
    """Build sklearn preprocessing pipeline.

    Args:
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names.
        scale_numeric: Whether to apply standard scaling to numeric features.

    Returns:
        Fitted Pipeline object.
    """
    numeric_transformer_steps = [
        ("imputer", SimpleImputer(strategy=CONFIG.preprocessing.numeric_strategy)),
    ]
    if scale_numeric:
        numeric_transformer_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_transformer_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=CONFIG.preprocessing.categorical_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])


def get_feature_names_after_preprocessing(
    pipeline: Pipeline,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    """Get feature names after preprocessing transformation.

    Args:
        pipeline: Fitted preprocessing pipeline.
        numeric_cols: Original numeric column names.
        categorical_cols: Original categorical column names.

    Returns:
        List of feature names after transformation.
    """
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = []

    # Numeric features
    if hasattr(preprocessor, "transformers_"):
        for name, _, cols in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
                if hasattr(encoder, "get_feature_names_out"):
                    cat_features = encoder.get_feature_names_out(cols)
                    feature_names.extend(cat_features.tolist())
                else:
                    feature_names.extend(cols)

    return feature_names
