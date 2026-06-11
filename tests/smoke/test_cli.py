"""Smoke tests for CLI commands."""
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestCLISmoke:
    """Smoke tests for CLI commands."""

    def test_cli_generate_data(self):
        """Test CLI generate-data works."""
        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-data", "--rows", "100"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_cli_validate_data(self):
        """Test CLI validate-data works."""
        # First generate data
        subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-data", "--rows", "100"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=30,
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "validate-data", "data/raw/synthetic_credit_data.csv"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_cli_run_eda(self):
        """Test CLI run-eda works."""
        subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-data", "--rows", "100"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=30,
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "run-eda", "data/raw/synthetic_credit_data.csv"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_cli_train_on_fast_debug(self):
        """Test CLI train works on fast_debug config."""
        subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-data", "--rows", "100", "--config", "configs/fast_debug.yaml"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=30,
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "train", "data/raw/synthetic_credit_data.csv", "--config", "configs/fast_debug.yaml"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0

    def test_cli_evaluate(self):
        """Test CLI evaluate works."""
        subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-data", "--rows", "100", "--config", "configs/fast_debug.yaml"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            [sys.executable, "-m", "app.cli", "train", "data/raw/synthetic_credit_data.csv", "--config", "configs/fast_debug.yaml"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=60,
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "evaluate", "--config", "configs/fast_debug.yaml"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_cli_select_threshold(self):
        """Test CLI select-threshold works."""
        subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-data", "--rows", "100", "--config", "configs/fast_debug.yaml"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            [sys.executable, "-m", "app.cli", "train", "data/raw/synthetic_credit_data.csv", "--config", "configs/fast_debug.yaml"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=60,
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "select-threshold", "--config", "configs/fast_debug.yaml"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_cli_predict(self):
        """Test CLI predict works."""
        subprocess.run(
            [sys.executable, "-m", "app.cli", "generate-data", "--rows", "100", "--config", "configs/fast_debug.yaml"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            [sys.executable, "-m", "app.cli", "train", "data/raw/synthetic_credit_data.csv", "--config", "configs/fast_debug.yaml"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=60,
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.cli", "predict", "data/raw/synthetic_credit_data.csv", "--config", "configs/fast_debug.yaml"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
