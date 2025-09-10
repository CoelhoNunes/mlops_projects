"""End-to-end tests for the MLOps pipeline."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mlops.data.features import FeatureEngineer
from mlops.data.loader import DataLoader
from mlops.data.validate import DataValidator
from mlops.modeling.evaluate import ModelEvaluator
from mlops.modeling.promote import ModelPromoter
from mlops.modeling.train import ModelTrainer
from mlops.utils.config import Config


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

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        config = Config()
        config.paths.artifacts = os.path.join(temp_dir, "artifacts")
        config.paths.models = os.path.join(temp_dir, "models")
        config.paths.logs = os.path.join(temp_dir, "logs")
        config.data.data_path = os.path.join(temp_dir, "data")
        config.data.target_column = "target"
        config.data.task_type = "classification"
        config.models.roster = ["logreg", "rf"]  # Skip heavy models for testing
        config.models.cv_folds = 2
        config.models.optuna_trials = 3
        config.mlflow.tracking_uri = os.path.join(temp_dir, "mlruns")
        config.ensure_directories()
        return config

    def test_complete_pipeline(self, temp_dir, synthetic_data, config):
        """Test the complete pipeline from data to model serving."""
        # Save synthetic data
        data_path = os.path.join(temp_dir, "data", "test_data.csv")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        synthetic_data.to_csv(data_path, index=False)

        # Mock MLflow to avoid actual tracking server
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run"),
            patch("mlflow.log_params"),
            patch("mlflow.log_metrics"),
            patch("mlflow.log_artifacts"),
            patch("mlflow.log_model"),
            patch("mlflow.end_run"),
        ):

            # 1. Data Loading
            loader = DataLoader(config)
            train_data, val_data, test_data = loader.load_and_split_data(data_path)

            assert len(train_data) > 0
            assert len(val_data) > 0
            assert len(test_data) > 0
            assert "target" in train_data.columns
            assert "target" in val_data.columns
            assert "target" in test_data.columns

            # 2. Data Validation
            validator = DataValidator(config)
            validation_results = validator.validate_data(train_data, "target")

            assert validation_results is not None
            assert "basic_info" in validation_results
            assert "schema" in validation_results
            assert "data_quality" in validation_results

            # 3. Feature Engineering
            feature_engineer = FeatureEngineer(config)
            X_train, y_train = feature_engineer.fit_transform(train_data, "target")
            X_val, y_val = feature_engineer.transform(val_data)
            X_test, y_test = feature_engineer.transform(test_data)

            assert X_train.shape[0] == len(train_data)
            assert X_val.shape[0] == len(val_data)
            assert X_test.shape[0] == len(test_data)
            assert len(y_train) == len(train_data)
            assert len(y_val) == len(val_data)
            assert len(y_test) == len(test_data)

            # 4. Model Training
            trainer = ModelTrainer(config)
            training_results = trainer.train_models(
                train_data, val_data, "target", feature_engineer
            )

            assert training_results is not None
            assert "best_model" in training_results
            assert "best_score" in training_results
            assert "training_summary" in training_results

            # 5. Model Evaluation
            evaluator = ModelEvaluator(config)
            evaluation_results = evaluator.evaluate_model(
                test_data, "target", training_results["best_model"], feature_engineer
            )

            assert evaluation_results is not None
            assert "metrics" in evaluation_results
            assert "predictions" in evaluation_results
            assert "model_card" in evaluation_results

            # 6. Model Registration (Mock)
            with patch("mlflow.register_model") as mock_register:
                promoter = ModelPromoter(config)
                registration_result = promoter.register_model(
                    training_results["best_model"],
                    "test-model",
                    "sklearn",
                    evaluation_results,
                )

                assert registration_result is not None
                mock_register.assert_called_once()

            # 7. Verify Artifacts
            artifacts_dir = os.path.join(temp_dir, "artifacts")
            models_dir = os.path.join(temp_dir, "models")

            assert os.path.exists(artifacts_dir)
            assert os.path.exists(models_dir)

            # Check that model files were saved
            model_files = list(Path(models_dir).glob("*.pkl"))
            assert len(model_files) > 0

            # Check that feature engineer was saved
            fe_files = list(Path(models_dir).glob("*feature_engineer*"))
            assert len(fe_files) > 0

    def test_pipeline_with_missing_data(self, temp_dir, config):
        """Test pipeline behavior with missing or invalid data."""
        # Test with non-existent data path
        loader = DataLoader(config)

        with pytest.raises(FileNotFoundError):
            loader.load_and_split_data("non_existent_file.csv")

    def test_pipeline_with_invalid_config(self, temp_dir, synthetic_data):
        """Test pipeline behavior with invalid configuration."""
        # Create config with invalid split ratios
        config = Config()
        config.paths.artifacts = os.path.join(temp_dir, "artifacts")
        config.paths.models = os.path.join(temp_dir, "models")
        config.paths.logs = os.path.join(temp_dir, "logs")
        config.data.data_path = os.path.join(temp_dir, "data")
        config.data.target_column = "target"
        config.data.train_size = 0.5
        config.data.val_size = 0.3
        config.data.test_size = 0.3  # Invalid: total > 1.0

        with pytest.raises(ValueError):
            config.ensure_directories()

    def test_pipeline_reproducibility(self, temp_dir, synthetic_data, config):
        """Test that pipeline produces reproducible results."""
        # Run pipeline twice with same seed
        results1 = self._run_pipeline_with_seed(temp_dir, synthetic_data, config, 42)
        results2 = self._run_pipeline_with_seed(temp_dir, synthetic_data, config, 42)

        # Results should be identical
        assert results1["best_score"] == results2["best_score"]
        assert results1["best_model_name"] == results2["best_model_name"]

    def _run_pipeline_with_seed(self, temp_dir, synthetic_data, config, seed):
        """Helper method to run pipeline with specific seed."""
        # Set seed
        config.system.random_seed = seed

        # Save synthetic data
        data_path = os.path.join(temp_dir, "data", f"test_data_{seed}.csv")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        synthetic_data.to_csv(data_path, index=False)

        # Mock MLflow
        with (
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run"),
            patch("mlflow.log_params"),
            patch("mlflow.log_metrics"),
            patch("mlflow.log_artifacts"),
            patch("mlflow.log_model"),
            patch("mlflow.end_run"),
        ):

            # Run pipeline
            loader = DataLoader(config)
            train_data, val_data, test_data = loader.load_and_split_data(data_path)

            feature_engineer = FeatureEngineer(config)
            feature_engineer.fit_transform(train_data, "target")

            trainer = ModelTrainer(config)
            training_results = trainer.train_models(
                train_data, val_data, "target", feature_engineer
            )

            return training_results


class TestPipelineIntegration:
    """Test integration between pipeline components."""

    def test_data_loader_feature_engineer_integration(self):
        """Test integration between DataLoader and FeatureEngineer."""
        # This test would verify that the data loaded by DataLoader
        # is compatible with FeatureEngineer
        pass

    def test_feature_engineer_model_trainer_integration(self):
        """Test integration between FeatureEngineer and ModelTrainer."""
        # This test would verify that the feature engineering pipeline
        # produces data compatible with the training pipeline
        pass

    def test_model_trainer_evaluator_integration(self):
        """Test integration between ModelTrainer and ModelEvaluator."""
        # This test would verify that the trained model can be
        # properly evaluated by the evaluator
        pass


class TestPipelineErrorHandling:
    """Test error handling in the pipeline."""

    def test_graceful_failure_on_model_training_error(self):
        """Test that pipeline handles model training failures gracefully."""
        pass

    def test_graceful_failure_on_evaluation_error(self):
        """Test that pipeline handles evaluation failures gracefully."""
        pass

    def test_graceful_failure_on_registration_error(self):
        """Test that pipeline handles model registration failures gracefully."""
        pass
