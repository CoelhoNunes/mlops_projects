"""End-to-end tests for the MLOps pipeline."""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


class TestEndToEndPipeline:
    """Test the complete MLOps pipeline end-to-end."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic dataset for testing."""
        np.random.seed(42)

        # Create synthetic classification dataset
        n_samples = 100
        n_features = 5

        data = {}
        for i in range(n_features):
            if i % 2 == 0:
                # Numeric features
                data[f"numeric_{i}"] = np.random.normal(0, 1, n_samples)
            else:
                # Categorical features
                data[f"cat_{i}"] = np.random.choice(["A", "B", "C"], n_samples)

        # Create target (simple rule: if sum of numeric > 0, then 1, else 0)
        numeric_sum = sum([data[f"numeric_{i}"] for i in range(0, n_features, 2)])
        data["target"] = (numeric_sum > 0).astype(int)

        # Add some noise
        noise = np.random.random(n_samples) < 0.1
        data["target"] = data["target"] ^ noise

        return pd.DataFrame(data)

    def test_training_script_import(self):
        """Test that training scripts can be imported."""
        import sys
        from pathlib import Path

        # Add src to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        try:
            import src.train
            import src.train_customer

            assert src.train is not None
            assert src.train_customer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import training scripts: {e}")

    def test_training_script_functions_exist(self):
        """Test that training scripts have expected functions."""
        import sys
        from pathlib import Path

        # Add src to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        import src.train
        import src.train_customer

        # Check that main functions exist in train.py
        assert hasattr(src.train, "load_data")
        assert hasattr(src.train, "split_data")
        assert hasattr(src.train, "preprocess_data")
        assert hasattr(src.train, "train_model")
        assert hasattr(src.train, "evaluate_model")
        assert hasattr(src.train, "log_to_mlflow")
        assert hasattr(src.train, "main")

        # Check that main functions exist in train_customer.py
        assert hasattr(src.train_customer, "load_customer_data")
        assert hasattr(src.train_customer, "engineer_features")
        assert hasattr(src.train_customer, "split_data")
        assert hasattr(src.train_customer, "train_models")
        assert hasattr(src.train_customer, "evaluate_model")
        assert hasattr(src.train_customer, "log_to_mlflow")
        assert hasattr(src.train_customer, "main")

    @patch("src.train.mlflow")
    @patch("src.train.load_digits")
    @patch("src.train.train_test_split")
    def test_smoke_test_mode(self, mock_split, mock_digits, mock_mlflow):
        """Test that smoke test mode works correctly."""
        import sys
        from pathlib import Path

        # Add src to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        import src.train

        # Mock the data loading
        mock_data = type("MockData", (), {})()
        mock_data.data = [[1, 2, 3], [4, 5, 6]]
        mock_data.target = [0, 1]
        mock_digits.return_value = mock_data

        # Mock the split
        mock_split.side_effect = [
            (
                type("MockArray", (), {})(),
                type("MockArray", (), {})(),
                type("MockArray", (), {})(),
                type("MockArray", (), {})(),
            ),  # First split
            (
                type("MockArray", (), {})(),
                type("MockArray", (), {})(),
                type("MockArray", (), {})(),
                type("MockArray", (), {})(),
            ),  # Second split
        ]

        # Mock MLflow
        mock_mlflow.set_tracking_uri = patch.MagicMock()
        mock_mlflow.set_experiment = patch.MagicMock()
        mock_mlflow.start_run = patch.MagicMock()
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

    def test_data_creation_and_manipulation(self, synthetic_data):
        """Test basic data creation and manipulation."""
        # Test that synthetic data was created correctly
        assert len(synthetic_data) == 100
        assert "target" in synthetic_data.columns

        # Test basic pandas operations
        assert synthetic_data["target"].dtype in [np.int64, np.int32]
        assert synthetic_data["target"].min() >= 0
        assert synthetic_data["target"].max() <= 1

    def test_file_operations(self, temp_dir, synthetic_data):
        """Test file operations with synthetic data."""
        # Save synthetic data
        data_path = os.path.join(temp_dir, "test_data.csv")
        synthetic_data.to_csv(data_path, index=False)

        # Verify file was created
        assert os.path.exists(data_path)

        # Load data back
        loaded_data = pd.read_csv(data_path)
        assert len(loaded_data) == len(synthetic_data)
        assert list(loaded_data.columns) == list(synthetic_data.columns)

    def test_numpy_operations(self):
        """Test basic numpy operations."""
        # Test array creation
        arr = np.array([1, 2, 3, 4, 5])
        assert len(arr) == 5
        assert arr.sum() == 15

        # Test random operations
        np.random.seed(42)
        random_arr = np.random.random(10)
        assert len(random_arr) == 10
        assert all(0 <= x <= 1 for x in random_arr)

    def test_pandas_operations(self):
        """Test basic pandas operations."""
        # Test DataFrame creation
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

        assert len(df) == 3
        assert list(df.columns) == ["A", "B", "C"]
        assert df["A"].sum() == 6

        # Test basic operations
        df["D"] = df["A"] + df["B"]
        assert df["D"].sum() == 21
