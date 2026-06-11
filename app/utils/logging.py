"""Logging utilities."""
import logging
import sys


def setup_logger(name: str = "credit_risk_lab", level: int = logging.INFO) -> logging.Logger:
    """Setup and return a logger with console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger()
