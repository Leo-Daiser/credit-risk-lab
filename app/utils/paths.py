"""Path utilities for the Credit Risk Research Lab."""
from pathlib import Path

from app.config import get_config


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def get_data_raw_dir(config=None) -> Path:
    """Get raw data directory."""
    if config is None:
        config = get_config()
    return get_project_root() / config.paths.data_raw


def get_data_processed_dir(config=None) -> Path:
    """Get processed data directory."""
    if config is None:
        config = get_config()
    return get_project_root() / config.paths.data_processed


def get_artifacts_dir(config=None) -> Path:
    """Get artifacts root directory."""
    if config is None:
        config = get_config()
    return get_project_root() / "artifacts"


def get_models_dir(config=None) -> Path:
    """Get models directory."""
    if config is None:
        config = get_config()
    return get_project_root() / config.paths.artifacts_models


def get_reports_dir(config=None) -> Path:
    """Get reports directory."""
    if config is None:
        config = get_config()
    return get_project_root() / config.paths.artifacts_reports


def get_plots_dir(config=None) -> Path:
    """Get plots directory."""
    if config is None:
        config = get_config()
    return get_project_root() / config.paths.artifacts_plots


def get_metrics_dir(config=None) -> Path:
    """Get metrics directory."""
    if config is None:
        config = get_config()
    return get_project_root() / config.paths.artifacts_metrics


def get_predictions_dir(config=None) -> Path:
    """Get predictions directory."""
    if config is None:
        config = get_config()
    return get_project_root() / config.paths.artifacts_predictions


def get_audit_dir(config=None) -> Path:
    """Get audit directory."""
    if config is None:
        config = get_config()
    return get_project_root() / config.paths.artifacts_audit


def get_logs_dir(config=None) -> Path:
    """Get logs directory."""
    if config is None:
        config = get_config()
    return get_project_root() / config.paths.artifacts_logs


def ensure_all_dirs(config=None) -> None:
    """Ensure all required directories exist."""
    dirs = [
        get_data_raw_dir(config),
        get_data_processed_dir(config),
        get_models_dir(config),
        get_reports_dir(config),
        get_plots_dir(config),
        get_metrics_dir(config),
        get_predictions_dir(config),
        get_audit_dir(config),
        get_logs_dir(config),
        get_project_root() / "data" / "samples",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
