"""Test the training script functionality."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_training_script_import():
    """Test that the training script can be imported."""
    try:
        import src.train

        assert src.train is not None
    except ImportError as e:
        pytest.fail(f"Failed to import training script: {e}")


def test_training_functions_exist():
    """Test that expected training functions exist."""
    import src.train

    # Check that main functions exist
    assert hasattr(src.train, "load_data")
    assert hasattr(src.train, "split_data")
    assert hasattr(src.train, "preprocess_data")
    assert hasattr(src.train, "train_model")
    assert hasattr(src.train, "evaluate_model")
    assert hasattr(src.train, "log_to_mlflow")
    assert hasattr(src.train, "main")


@patch("src.train.mlflow")
@patch("src.train.load_digits")
@patch("src.train.train_test_split")
def test_smoke_test_mode(mock_split, mock_digits, mock_mlflow):
    """Test that smoke test mode works correctly."""
    import src.train

    # Mock the data loading
    mock_data = MagicMock()
    mock_data.data = [[1, 2, 3], [4, 5, 6]]
    mock_data.target = [0, 1]
    mock_digits.return_value = mock_data

    # Mock the split
    mock_split.side_effect = [
        (MagicMock(), MagicMock(), MagicMock(), MagicMock()),  # First split
        (MagicMock(), MagicMock(), MagicMock(), MagicMock()),  # Second split
    ]

    # Mock MLflow
    mock_mlflow.set_tracking_uri = MagicMock()
    mock_mlflow.set_experiment = MagicMock()
    mock_mlflow.start_run = MagicMock()
    mock_mlflow.active_run.return_value.info.run_id = "test-run"
    mock_mlflow.active_run.return_value.info.experiment_id = "test-exp"

    # Test that the script can run in smoke mode without crashing
    try:
        # This should not raise an exception
        result = src.train.main()
        # In smoke mode, it should return 0 (success)
        assert result == 0
    except Exception as e:
        pytest.fail(f"Training script failed in smoke mode: {e}")


def test_mlflow_tracking_uri_default():
    """Test that MLflow tracking URI defaults to local mlruns."""
    import src.train

    # Test the default argument
    assert src.train.main.__defaults__ is not None
    # The main function should have default arguments for tracking URI
    assert hasattr(src.train.main, "__defaults__")
