"""Tests for CLI functionality."""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestCLI:
    """Tests for CLI commands."""

    def test_generate_data_cli(self):
        """Test CLI generate-data command."""
        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-data", "--rows", "100"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (PROJECT_ROOT / "data" / "raw" / "synthetic_credit_data.csv").exists()

    def test_validate_data_cli(self):
        """Test CLI validate-data command."""
        # First generate data
        subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-data", "--rows", "100"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "validate-data", "data/raw/synthetic_credit_data.csv"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_train_cli(self):
        """Test CLI train command."""
        # First generate data
        subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-data", "--rows", "200"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "train", "data/raw/synthetic_credit_data.csv"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (PROJECT_ROOT / "artifacts" / "models" / "metadata.json").exists()

    def test_generate_report_cli(self):
        """Test CLI generate-report command."""
        # Run full pipeline first
        subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-data", "--rows", "200"],
            cwd=str(PROJECT_ROOT), capture_output=True,
        )
        subprocess.run(
            [sys.executable, "-m", "app.cli", "train", "data/raw/synthetic_credit_data.csv"],
            cwd=str(PROJECT_ROOT), capture_output=True,
        )
        subprocess.run(
            [sys.executable, "-m", "app.cli", "evaluate"],
            cwd=str(PROJECT_ROOT), capture_output=True,
        )
        subprocess.run(
            [sys.executable, "-m", "app.cli", "select-threshold"],
            cwd=str(PROJECT_ROOT), capture_output=True,
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-report"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (PROJECT_ROOT / "artifacts" / "reports" / "final_report.html").exists()
