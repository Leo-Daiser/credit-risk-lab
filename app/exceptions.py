"""Custom exceptions for the Credit Risk Research Lab."""


class CreditRiskError(Exception):
    """Base exception for credit risk lab."""
    pass


class DataValidationError(CreditRiskError):
    """Raised when data validation fails."""
    pass


class MissingColumnError(DataValidationError):
    """Raised when required column is missing."""
    pass


class TargetColumnError(DataValidationError):
    """Raised when target column is invalid."""
    pass


class LeakageDetectedError(CreditRiskError):
    """Raised when data leakage is detected."""
    pass


class ModelNotFoundError(CreditRiskError):
    """Raised when trained model is not found."""
    pass


class ConfigError(CreditRiskError):
    """Raised when configuration is invalid."""
    pass


class PredictionError(CreditRiskError):
    """Raised when prediction fails."""
    pass
