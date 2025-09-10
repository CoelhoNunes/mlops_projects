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
    @patch("src.train.StandardScaler")
    @patch("src.train.RandomForestClassifier")
    @patch("src.train.plt")
    def test_smoke_test_mode(self, mock_plt, mock_rf, mock_scaler, mock_split, mock_digits, mock_mlflow):
        """Test that smoke test mode works correctly."""
        import sys
        from pathlib import Path

        # Add src to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        import src.train
        import numpy as np
        import pandas as pd

        # Mock the data loading with proper numpy arrays
        mock_data = type("MockData", (), {})()
        mock_data.data = np.array([[1, 2, 3], [4, 5, 6]])
        mock_data.target = np.array([0, 1])
        mock_digits.return_value = mock_data

        # Mock the split with proper arrays
        mock_X_train = pd.DataFrame([[1, 2, 3]], columns=['pixel_0', 'pixel_1', 'pixel_2'])
        mock_X_val = pd.DataFrame([[4, 5, 6]], columns=['pixel_0', 'pixel_1', 'pixel_2'])
        mock_X_test = pd.DataFrame([[7, 8, 9]], columns=['pixel_0', 'pixel_1', 'pixel_2'])
        mock_y_train = pd.Series([0], name='digit')
        mock_y_val = pd.Series([1], name='digit')
        mock_y_test = pd.Series([0], name='digit')
        
        mock_split.side_effect = [
            (mock_X_train, mock_X_test, mock_y_train, mock_y_test),  # First split
            (mock_X_train, mock_X_val, mock_y_train, mock_y_val),    # Second split
        ]

        # Mock scaler
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.array([[1, 2, 3]])
        mock_scaler_instance.transform.return_value = np.array([[4, 5, 6]])
        mock_scaler.return_value = mock_scaler_instance

        # Mock model
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.n_estimators = 100
        mock_model.max_depth = 10
        mock_model.feature_importances_ = np.array([0.3, 0.4, 0.3])
        mock_rf.return_value = mock_model

        # Mock MLflow
        from unittest.mock import MagicMock

        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        mock_mlflow.start_run = MagicMock()
        mock_mlflow.active_run.return_value.info.run_id = "test-run"
        mock_mlflow.active_run.return_value.info.experiment_id = "test-exp"
        mock_mlflow.log_params = MagicMock()
        mock_mlflow.log_metrics = MagicMock()
        mock_mlflow.sklearn.log_model = MagicMock()
        mock_mlflow.log_artifact = MagicMock()

        # Mock matplotlib
        mock_plt.figure.return_value = MagicMock()
        mock_plt.imshow.return_value = MagicMock()
        mock_plt.title.return_value = MagicMock()
        mock_plt.colorbar.return_value = MagicMock()
        mock_plt.text.return_value = MagicMock()
        mock_plt.ylabel.return_value = MagicMock()
        mock_plt.xlabel.return_value = MagicMock()
        mock_plt.tight_layout.return_value = MagicMock()
        mock_plt.savefig.return_value = MagicMock()
        mock_plt.close.return_value = MagicMock()

        # Test that the script can run in smoke mode without crashing
        try:
            # Mock sys.argv to provide proper arguments for the main function
            import sys

            original_argv = sys.argv
            sys.argv = ["train.py", "--smoke"]

            # This should not raise an exception
            result = src.train.main()
            # In smoke mode, it should return 0 (success)
            assert result == 0

            # Restore original argv
            sys.argv = original_argv
        except Exception as e:
            # Restore original argv in case of exception
            sys.argv = original_argv
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
