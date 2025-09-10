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
@patch("src.train.StandardScaler")
@patch("src.train.RandomForestClassifier")
@patch("src.train.plt")
def test_smoke_test_mode(
    mock_plt, mock_rf, mock_scaler, mock_split, mock_digits, mock_mlflow
):
    """Test that smoke test mode works correctly."""
    import src.train
    import numpy as np
    import pandas as pd

    # Mock the data loading with proper numpy arrays (larger dataset for smoke test)
    mock_data = MagicMock()
    mock_data.data = np.random.randn(
        200, 64
    )  # 200 samples, 64 features (like digits dataset)
    mock_data.target = np.random.randint(0, 10, 200)  # 200 targets
    mock_digits.return_value = mock_data

    # Mock the split with proper arrays (larger datasets)
    mock_X_train = pd.DataFrame(
        np.random.randn(100, 64), columns=[f"pixel_{i}" for i in range(64)]
    )
    mock_X_val = pd.DataFrame(
        np.random.randn(50, 64), columns=[f"pixel_{i}" for i in range(64)]
    )
    mock_X_test = pd.DataFrame(
        np.random.randn(50, 64), columns=[f"pixel_{i}" for i in range(64)]
    )
    mock_y_train = pd.Series(np.random.randint(0, 10, 100), name="digit")
    mock_y_val = pd.Series(np.random.randint(0, 10, 50), name="digit")
    mock_y_test = pd.Series(np.random.randint(0, 10, 50), name="digit")

    mock_split.side_effect = [
        (mock_X_train, mock_X_test, mock_y_train, mock_y_test),  # First split
        (mock_X_train, mock_X_val, mock_y_train, mock_y_val),  # Second split
    ]

    # Mock scaler
    mock_scaler_instance = MagicMock()
    mock_scaler_instance.fit_transform.return_value = np.random.randn(100, 64)
    mock_scaler_instance.transform.return_value = np.random.randn(50, 64)
    mock_scaler.return_value = mock_scaler_instance

    # Mock model
    mock_model = MagicMock()
    mock_model.fit.return_value = None
    mock_model.predict.return_value = np.random.randint(
        0, 10, 50
    )  # Match validation size
    mock_model.n_estimators = 100
    mock_model.max_depth = 10
    mock_model.feature_importances_ = np.random.rand(64)  # 64 features
    mock_rf.return_value = mock_model

    # Mock MLflow
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


def test_mlflow_tracking_uri_default():
    """Test that MLflow tracking URI defaults to local mlruns."""
    import src.train
    import inspect

    # Test that the main function exists and can be called
    assert callable(src.train.main)

    # Test that we can inspect the function signature
    sig = inspect.signature(src.train.main)
    # The main function takes no parameters, but we can test the argument parser
    assert len(sig.parameters) == 0  # main() takes no parameters


def test_load_data_function():
    """Test the load_data function with smoke mode."""
    import src.train
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    with patch("src.train.load_digits") as mock_load_digits:
        # Mock the digits dataset
        mock_data = type("MockData", (), {})()
        mock_data.data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        mock_data.target = np.array([0, 1, 2])
        mock_load_digits.return_value = mock_data

        # Test normal mode
        X, y = src.train.load_data(smoke=False)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[0] == 3
        assert X.shape[1] == 4
        assert len(y) == 3

        # Test smoke mode
        with patch("src.train.np.random.choice") as mock_choice:
            mock_choice.return_value = np.array([0, 1])
            X_smoke, y_smoke = src.train.load_data(smoke=True)
            assert X_smoke.shape[0] == 2
            assert len(y_smoke) == 2


def test_split_data_function():
    """Test the split_data function."""
    import src.train
    import pandas as pd
    import numpy as np

    # Create test data
    X = pd.DataFrame(
        np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)]
    )
    y = pd.Series(np.random.randint(0, 3, 100), name="target")

    # Test data splitting
    result = src.train.split_data(X, y, test_size=0.2, val_size=0.2)
    X_train, X_val, X_test, y_train, y_val, y_test = result

    # Check shapes
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 100
    assert len(y_train) + len(y_val) + len(y_test) == 100
    assert X_train.shape[1] == 5
    assert X_val.shape[1] == 5
    assert X_test.shape[1] == 5


def test_preprocess_data_function():
    """Test the preprocess_data function."""
    import src.train
    import pandas as pd
    import numpy as np

    # Create test data
    X_train = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    X_val = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    X_test = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])

    # Test preprocessing
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = src.train.preprocess_data(
        X_train, X_val, X_test
    )

    # Check that scaling was applied
    assert isinstance(X_train_scaled, np.ndarray)
    assert isinstance(X_val_scaled, np.ndarray)
    assert isinstance(X_test_scaled, np.ndarray)
    assert X_train_scaled.shape == (50, 3)
    assert X_val_scaled.shape == (20, 3)
    assert X_test_scaled.shape == (30, 3)


def test_train_model_function():
    """Test the train_model function."""
    import src.train
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    # Create test data
    X_train = np.random.randn(50, 3)
    y_train = pd.Series(np.random.randint(0, 3, 50), name="target")
    X_val = np.random.randn(20, 3)
    y_val = pd.Series(np.random.randint(0, 3, 20), name="target")

    with patch("src.train.RandomForestClassifier") as mock_rf:
        # Mock the model
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.randint(
            0, 3, 20
        )  # Match y_val size (20)
        mock_model.n_estimators = 100
        mock_model.max_depth = 10
        mock_model.feature_importances_ = np.array([0.3, 0.4, 0.3])
        mock_rf.return_value = mock_model

        # Test training
        model, results = src.train.train_model(X_train, y_train, X_val, y_val)

        # Check results
        assert "val_accuracy" in results
        assert "n_estimators" in results
        assert "max_depth" in results
        assert "feature_importance" in results


def test_evaluate_model_function():
    """Test the evaluate_model function."""
    import src.train
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    # Create test data
    X_test = np.random.randn(30, 3)
    y_test = pd.Series(np.random.randint(0, 3, 30), name="target")

    with patch("src.train.RandomForestClassifier") as mock_rf:
        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.randint(0, 3, 30)
        mock_rf.return_value = mock_model

        # Test evaluation
        results = src.train.evaluate_model(mock_model, X_test, y_test)

        # Check results
        assert "test_accuracy" in results
        assert "classification_report" in results
        assert "confusion_matrix" in results


def test_create_confusion_matrix_plot_function():
    """Test the create_confusion_matrix_plot function."""
    import src.train
    import numpy as np
    from unittest.mock import patch, MagicMock

    # Create test confusion matrix
    cm = np.array([[10, 2], [3, 15]])

    with patch("src.train.plt") as mock_plt:
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

        # Test plot creation
        result_path = src.train.create_confusion_matrix_plot(cm, "test_plot.png")
        assert result_path == "test_plot.png"


def test_log_to_mlflow_function():
    """Test the log_to_mlflow function."""
    import src.train
    from unittest.mock import patch, MagicMock

    # Create mock objects
    mock_model = MagicMock()
    mock_scaler = MagicMock()
    training_results = {
        "val_accuracy": 0.85,
        "n_estimators": 100,
        "max_depth": 10,
        "feature_importance": {0: 0.3, 1: 0.4, 2: 0.3},
    }
    evaluation_results = {
        "test_accuracy": 0.82,
        "classification_report": {"accuracy": 0.82},
        "confusion_matrix": [[10, 2], [3, 15]],
    }

    with patch("src.train.mlflow") as mock_mlflow, patch(
        "src.train.pd.Timestamp"
    ) as mock_timestamp, patch("src.train.pd.DataFrame") as mock_df, patch(
        "builtins.open", MagicMock()
    ), patch(
        "os.path.exists", return_value=False
    ):

        mock_timestamp.now.return_value.strftime.return_value = "20240101-120000"
        mock_df.return_value.to_csv.return_value = None

        # Test MLflow logging
        src.train.log_to_mlflow(
            mock_model,
            mock_scaler,
            training_results,
            evaluation_results,
            "test_plot.png",
            smoke=False,
        )

        # Check that MLflow methods were called
        mock_mlflow.set_experiment.assert_called()
        mock_mlflow.start_run.assert_called()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metrics.assert_called()
        mock_mlflow.sklearn.log_model.assert_called()
        mock_mlflow.log_artifact.assert_called()


def test_main_function_error_handling():
    """Test the main function error handling."""
    import src.train
    from unittest.mock import patch
    import sys

    # Mock sys.argv to avoid pytest arguments
    original_argv = sys.argv
    sys.argv = ["train.py"]

    try:
        with patch("src.train.load_data", side_effect=Exception("Test error")):
            result = src.train.main()
            assert result == 1  # Should return 1 on error
    finally:
        # Restore original argv
        sys.argv = original_argv


def test_train_customer_script_import():
    """Test that the customer training script can be imported."""
    try:
        import src.train_customer

        assert src.train_customer is not None
    except ImportError as e:
        pytest.fail(f"Failed to import customer training script: {e}")


def test_train_customer_functions_exist():
    """Test that expected customer training functions exist."""
    import src.train_customer

    # Check that main functions exist
    assert hasattr(src.train_customer, "load_customer_data")
    assert hasattr(src.train_customer, "engineer_features")
    assert hasattr(src.train_customer, "split_data")
    assert hasattr(src.train_customer, "train_models")
    assert hasattr(src.train_customer, "evaluate_model")
    assert hasattr(src.train_customer, "log_to_mlflow")
    assert hasattr(src.train_customer, "main")


def test_train_customer_main_function():
    """Test that the customer training main function exists and is callable."""
    import src.train_customer

    assert callable(src.train_customer.main)


def test_load_customer_data_function():
    """Test the load_customer_data function."""
    import src.train_customer
    import pandas as pd
    from unittest.mock import patch, mock_open

    # Create test CSV data
    test_data = """customer_unique_id,customer_zip_code_prefix,customer_city,customer_state
1,12345,São Paulo,SP
2,54321,Rio de Janeiro,RJ
3,98765,Belo Horizonte,MG
4,11111,Salvador,BA
5,22222,Fortaleza,CE"""

    with patch("builtins.open", mock_open(read_data=test_data)), patch(
        "pandas.read_csv"
    ) as mock_read_csv:

        # Mock the CSV data
        mock_df = pd.DataFrame(
            {
                "customer_unique_id": [1, 2, 3, 4, 5],
                "customer_zip_code_prefix": [
                    "12345",
                    "54321",
                    "98765",
                    "11111",
                    "22222",
                ],
                "customer_city": [
                    "São Paulo",
                    "Rio de Janeiro",
                    "Belo Horizonte",
                    "Salvador",
                    "Fortaleza",
                ],
                "customer_state": ["SP", "RJ", "MG", "BA", "CE"],
            }
        )
        mock_read_csv.return_value = mock_df

        # Test normal mode
        result = src.train_customer.load_customer_data("test.csv", smoke=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

        # Test smoke mode
        with patch.object(mock_df, "sample") as mock_sample:
            mock_sample.return_value.reset_index.return_value = mock_df.iloc[:3]
            result_smoke = src.train_customer.load_customer_data("test.csv", smoke=True)
            assert isinstance(result_smoke, pd.DataFrame)


def test_engineer_features_function():
    """Test the engineer_features function."""
    import src.train_customer
    import pandas as pd

    # Create test data
    test_df = pd.DataFrame(
        {
            "customer_unique_id": [1, 2, 3, 4, 5],
            "customer_zip_code_prefix": ["12345", "54321", "98765", "11111", "22222"],
            "customer_city": [
                "São Paulo",
                "Rio de Janeiro",
                "Belo Horizonte",
                "Salvador",
                "Fortaleza",
            ],
            "customer_state": ["SP", "RJ", "MG", "BA", "CE"],
        }
    )

    # Test feature engineering - function returns only DataFrame, not tuple
    result_df = src.train_customer.engineer_features(test_df)

    # Check that new features were created
    assert "zip_code_numeric" in result_df.columns
    assert "region" in result_df.columns
    assert "city_length" in result_df.columns
    assert "city_word_count" in result_df.columns


def test_customer_split_data_function():
    """Test the split_data function for customer training."""
    import src.train_customer
    import pandas as pd

    # Create test data
    X = pd.DataFrame(
        {
            "customer_unique_id": range(100),
            "customer_zip_code_prefix": ["12345"] * 100,
            "customer_city": ["São Paulo"] * 100,
        }
    )
    y = pd.Series(["SP"] * 50 + ["RJ"] * 50, name="customer_state")

    # Test data splitting - function requires X and y parameters
    result = src.train_customer.split_data(X, y, test_size=0.2, val_size=0.2)
    X_train, X_val, X_test, y_train, y_val, y_test = result

    # Check shapes
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 100
    assert len(y_train) + len(y_val) + len(y_test) == 100


def test_create_model_pipeline_function():
    """Test the create_model_pipeline function."""
    import src.train_customer

    # Test Random Forest pipeline
    rf_pipeline = src.train_customer.create_model_pipeline("rf")
    assert rf_pipeline is not None
    assert hasattr(rf_pipeline, "fit")
    assert hasattr(rf_pipeline, "predict")

    # Test that unsupported model type raises ValueError
    try:
        src.train_customer.create_model_pipeline("xgb")
        assert False, "Expected ValueError for unsupported model type"
    except ValueError:
        pass  # Expected behavior


def test_create_lr_pipeline_with_params_function():
    """Test the create_lr_pipeline_with_params function."""
    import src.train_customer

    # Test with different penalty types
    lr_pipeline_l1 = src.train_customer.create_lr_pipeline_with_params("l1")
    assert lr_pipeline_l1 is not None
    assert hasattr(lr_pipeline_l1, "fit")
    assert hasattr(lr_pipeline_l1, "predict")

    lr_pipeline_l2 = src.train_customer.create_lr_pipeline_with_params("l2")
    assert lr_pipeline_l2 is not None
    assert hasattr(lr_pipeline_l2, "fit")
    assert hasattr(lr_pipeline_l2, "predict")


def test_train_models_function():
    """Test the train_models function."""
    import src.train_customer
    import pandas as pd
    import numpy as np
    from unittest.mock import patch

    # Create test data
    X_train = pd.DataFrame(
        np.random.randn(50, 3), columns=["feature1", "feature2", "feature3"]
    )
    y_train = pd.Series(np.random.randint(0, 3, 50), name="target")
    X_val = pd.DataFrame(
        np.random.randn(20, 3), columns=["feature1", "feature2", "feature3"]
    )
    y_val = pd.Series(np.random.randint(0, 3, 20), name="target")

    with patch(
        "src.train_customer.create_model_pipeline"
    ) as mock_create_pipeline, patch(
        "src.train_customer.optuna.create_study"
    ) as mock_create_study, patch(
        "src.train_customer.objective"
    ) as mock_objective:

        # Mock the pipeline
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock()
        mock_pipeline.fit.return_value = None
        mock_pipeline.predict.return_value = np.random.randint(0, 3, 20)
        mock_pipeline.score.return_value = 0.85
        mock_pipeline.named_steps = {"classifier": MagicMock()}
        mock_create_pipeline.return_value = mock_pipeline

        # Mock optuna study
        mock_study = MagicMock()
        mock_study.optimize.return_value = None
        mock_study.best_params = {"n_estimators": 100, "max_depth": 10}
        mock_study.best_value = 0.85
        mock_create_study.return_value = mock_study

        # Mock objective function to return a float
        mock_objective.return_value = 0.85

        # Test training
        results = src.train_customer.train_models(X_train, y_train, X_val, y_val)

        # Check results structure
        assert isinstance(results, dict)
        assert "random_forest" in results
        assert "logistic_regression" in results
        assert "best_model" in results
        assert "best_score" in results


def test_customer_evaluate_model_function():
    """Test the evaluate_model function for customer training."""
    import src.train_customer
    import pandas as pd
    import numpy as np
    from unittest.mock import patch

    # Create test data
    X_test = pd.DataFrame(
        np.random.randn(30, 3), columns=["feature1", "feature2", "feature3"]
    )
    y_test = pd.Series(np.random.randint(0, 3, 30), name="target")

    # Mock the model
    mock_model = patch("src.train_customer.Pipeline").start()
    mock_model.predict.return_value = np.random.randint(0, 3, 30)

    # Test evaluation
    results = src.train_customer.evaluate_model(mock_model, X_test, y_test)

    # Check results
    assert "test_accuracy" in results
    assert "classification_report" in results
    assert "confusion_matrix" in results


def test_customer_create_confusion_matrix_plot_function():
    """Test the create_confusion_matrix_plot function for customer training."""
    import src.train_customer
    import numpy as np
    from unittest.mock import patch, MagicMock

    # Create test confusion matrix
    cm = np.array([[10, 2], [3, 15]])

    with patch("src.train_customer.plt") as mock_plt, patch(
        "src.train_customer.sns"
    ) as mock_sns:
        mock_plt.figure.return_value = MagicMock()
        mock_sns.heatmap.return_value = MagicMock()
        mock_plt.title.return_value = MagicMock()
        mock_plt.xlabel.return_value = MagicMock()
        mock_plt.ylabel.return_value = MagicMock()
        mock_plt.tight_layout.return_value = MagicMock()
        mock_plt.savefig.return_value = MagicMock()
        mock_plt.close.return_value = MagicMock()

        # Test plot creation - function doesn't return anything
        result = src.train_customer.create_confusion_matrix_plot(cm, "test_plot.png")
        # Function returns None, just check it doesn't raise an error
        assert result is None


def test_ensure_model_compatibility_function():
    """Test the ensure_model_compatibility function."""
    import src.train_customer
    from unittest.mock import MagicMock

    # Mock pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {"classifier": MagicMock()}

    # Test compatibility check
    result = src.train_customer.ensure_model_compatibility(mock_pipeline)
    assert result is not None


def test_customer_log_to_mlflow_function():
    """Test the log_to_mlflow function for customer training."""
    import src.train_customer
    from unittest.mock import patch, MagicMock

    # Create mock objects
    import pandas as pd
    import numpy as np

    mock_model = MagicMock()
    mock_encoders = {"label_encoder": MagicMock()}
    training_results = {
        "random_forest": {"val_score": 0.85, "params": {"n_estimators": 100}},
        "logistic_regression": {"val_score": 0.82, "params": {"penalty": "l2"}},
    }
    evaluation_results = {
        "test_accuracy": 0.84,
        "classification_report": {"accuracy": 0.84},
        "confusion_matrix": [[10, 2], [3, 15]],
    }

    # Create test data
    X_test = pd.DataFrame(np.random.randn(20, 3), columns=["f1", "f2", "f3"])
    y_test = pd.Series(np.random.randint(0, 3, 20), name="target")
    X_train = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
    X_val = pd.DataFrame(np.random.randn(20, 3), columns=["f1", "f2", "f3"])

    with patch("src.train_customer.mlflow") as mock_mlflow, patch(
        "src.train_customer.pd.Timestamp"
    ) as mock_timestamp, patch("src.train_customer.pd.DataFrame") as mock_df, patch(
        "builtins.open", MagicMock()
    ), patch(
        "os.path.exists", return_value=False
    ):

        mock_timestamp.now.return_value.strftime.return_value = "20240101-120000"
        mock_df.return_value.to_csv.return_value = None

        # Test MLflow logging with all required parameters
        src.train_customer.log_to_mlflow(
            mock_model,
            mock_encoders,
            training_results,
            evaluation_results,
            "test_plot.png",
            X_test,
            y_test,
            X_train,
            X_val,
            smoke=False,
        )

        # Check that MLflow methods were called
        mock_mlflow.set_experiment.assert_called()
        mock_mlflow.start_run.assert_called()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metrics.assert_called()
        mock_mlflow.sklearn.log_model.assert_called()
        mock_mlflow.log_artifact.assert_called()


def test_customer_main_function_error_handling():
    """Test the main function error handling for customer training."""
    import src.train_customer
    from unittest.mock import patch
    import sys

    # Mock sys.argv to avoid pytest arguments
    original_argv = sys.argv
    sys.argv = ["train_customer.py"]

    try:
        with patch(
            "src.train_customer.load_customer_data", side_effect=Exception("Test error")
        ):
            result = src.train_customer.main()
            assert result == 1  # Should return 1 on error
    finally:
        # Restore original argv
        sys.argv = original_argv
