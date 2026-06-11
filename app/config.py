"""Configuration management using Pydantic and YAML."""
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    random_state: int = 42
    artifact_dir: str = "artifacts"
    data_dir: str = "data"


class SyntheticConfig(BaseModel):
    rows: int = 5000
    default_rate_target: float = 0.35
    noise_level: float = 0.05
    missing_rate: float = 0.02
    outlier_rate: float = 0.01


class DataConfig(BaseModel):
    default_rows: int = 5000
    test_size: float = 0.2
    validation_size: float = 0.0
    target_column: str = "target"
    id_column: str = "applicant_id"


class FeaturesConfig(BaseModel):
    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    target: str = "target"
    id_column: str = "applicant_id"


class PreprocessingConfig(BaseModel):
    numeric_strategy: str = "median"
    categorical_strategy: str = "most_frequent"
    scaling: bool = True


class TrainingConfig(BaseModel):
    primary_metric: str = "pr_auc"
    cv_folds: int = 5
    n_jobs: int = 1
    enable_hyperparameter_search: bool = True
    max_search_iter: int = 20
    random_state: int = 42
    models: list[str] = Field(default_factory=lambda: ["logistic_regression", "random_forest", "gradient_boosting", "hist_gradient_boosting"])
    scoring_metric: str = "pr_auc"
    test_size: float = 0.2


class ModelConfig(BaseModel):
    logistic_regression: dict = Field(default_factory=lambda: {"class_weight": "balanced", "max_iter": 5000})
    random_forest: dict = Field(default_factory=lambda: {"class_weight": "balanced", "n_estimators": 200})
    gradient_boosting: dict = Field(default_factory=lambda: {"n_estimators": 200})
    hist_gradient_boosting: dict = Field(default_factory=lambda: {"max_iter": 200})


class ThresholdConfig(BaseModel):
    fn_cost: float = 5.0
    fp_cost: float = 1.0
    min_recall: float = 0.75
    min_precision: float = 0.60
    manual_review_low: float = 0.3
    manual_review_high: float = 0.7


class ValidationConfig(BaseModel):
    max_missing_share: float = 0.5
    max_duplicate_share: float = 0.1
    leakage_keywords: list[str] = Field(default_factory=lambda: ["target", "default", "approved", "decision", "status", "paid", "delinquent", "outcome"])


class ReportsConfig(BaseModel):
    generate_plots: bool = True
    generate_html: bool = True
    generate_markdown: bool = True
    format: str = "html"
    title: str = "Credit Risk Research Lab Report"


class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_samples: str = "data/samples"
    artifacts_models: str = "artifacts/models"
    artifacts_reports: str = "artifacts/reports"
    artifacts_plots: str = "artifacts/plots"
    artifacts_metrics: str = "artifacts/metrics"
    artifacts_predictions: str = "artifacts/predictions"
    artifacts_audit: str = "artifacts/audit"
    artifacts_logs: str = "artifacts/logs"


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_config(config_path: str | Path = "configs/default.yaml") -> AppConfig:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return AppConfig(**data)
    return AppConfig()


CONFIG = load_config()


def get_config(config_path: str | Path | None = None) -> AppConfig:
    """Get config from path or return default."""
    if config_path is None:
        return CONFIG
    return load_config(config_path)
