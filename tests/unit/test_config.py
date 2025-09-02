"""Unit tests for configuration module."""

import os
import tempfile
from unittest.mock import patch

import pytest

from mlops.utils.config import Config, DataConfig, MLflowConfig, ModelConfig


class TestDataConfig:
    """Test DataConfig class."""

    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        assert config.data_path == "./data/*.csv"
        assert config.target_column == "target"
        assert config.task_type == "auto"
        assert config.train_size == 0.7
        assert config.val_size == 0.15
        assert config.test_size == 0.15
        assert config.cv_folds == 5
        assert config.random_seed == 42

    def test_data_config_custom_values(self):
        """Test DataConfig with custom values."""
        config = DataConfig(
            data_path="./custom/path/*.csv",
            target_column="label",
            task_type="classification",
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            cv_folds=3,
            random_seed=123
        )
        assert config.data_path == "./custom/path/*.csv"
        assert config.target_column == "label"
        assert config.task_type == "classification"
        assert config.train_size == 0.8
        assert config.val_size == 0.1
        assert config.test_size == 0.1
        assert config.cv_folds == 3
        assert config.random_seed == 123


class TestModelConfig:
    """Test ModelConfig class."""

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.roster == ["logreg", "rf", "xgboost", "lightgbm", "torch_mlp"]
        assert config.cv_folds == 5
        assert config.optuna_trials == 50
        assert config.random_seed == 42

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            roster=["logreg", "rf"],
            cv_folds=3,
            optuna_trials=25,
            random_seed=123
        )
        assert config.roster == ["logreg", "rf"]
        assert config.cv_folds == 3
        assert config.optuna_trials == 25
        assert config.random_seed == 123


class TestMLflowConfig:
    """Test MLflowConfig class."""

    def test_mlflow_config_defaults(self):
        """Test MLflowConfig default values."""
        config = MLflowConfig()
        assert config.tracking_uri == "./mlruns"
        assert config.experiment_name == "mlops-experiment"
        assert config.registry_name == "mlops-registry"
        assert config.log_artifacts is True
        assert config.log_models is True

    def test_mlflow_config_custom_values(self):
        """Test MLflowConfig with custom values."""
        config = MLflowConfig(
            tracking_uri="http://localhost:5000",
            experiment_name="custom-exp",
            registry_name="custom-reg",
            log_artifacts=False,
            log_models=False
        )
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "custom-exp"
        assert config.registry_name == "custom-reg"
        assert config.log_artifacts is False
        assert config.log_models is False


class TestConfig:
    """Test main Config class."""

    def test_config_defaults(self):
        """Test Config default values."""
        config = Config()
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.models, ModelConfig)
        assert isinstance(config.mlflow, MLflowConfig)
        assert config.system.random_seed == 42

    def test_config_from_yaml(self):
        """Test loading config from YAML."""
        yaml_content = """
        data:
          data_path: "./custom/data/*.csv"
          target_column: "label"
        models:
          roster: ["logreg", "rf"]
        mlflow:
          tracking_uri: "http://localhost:5000"
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = Config.from_yaml(yaml_path)
            assert config.data.data_path == "./custom/data/*.csv"
            assert config.data.target_column == "label"
            assert config.models.roster == ["logreg", "rf"]
            assert config.mlflow.tracking_uri == "http://localhost:5000"
        finally:
            os.unlink(yaml_path)

    def test_config_from_env(self):
        """Test loading config from environment variables."""
        env_vars = {
            "MLOPS_DATA_PATH": "./env/data/*.csv",
            "MLOPS_TARGET_COLUMN": "env_label",
            "MLOPS_MODELS": "logreg,rf",
            "MLFLOW_TRACKING_URI": "http://env:5000"
        }

        with patch.dict(os.environ, env_vars):
            config = Config.from_env()
            assert config.data.data_path == "./env/data/*.csv"
            assert config.data.target_column == "env_label"
            assert config.models.roster == ["logreg", "rf"]
            assert config.mlflow.tracking_uri == "http://env:5000"

    def test_config_save_yaml(self):
        """Test saving config to YAML."""
        config = Config()
        config.data.data_path = "./test/path/*.csv"
        config.models.roster = ["test_model"]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name

        try:
            config.save_yaml(yaml_path)

            # Load back and verify
            loaded_config = Config.from_yaml(yaml_path)
            assert loaded_config.data.data_path == "./test/path/*.csv"
            assert loaded_config.models.roster == ["test_model"]
        finally:
            os.unlink(yaml_path)

    def test_config_get_model_config(self):
        """Test getting specific model configuration."""
        config = Config()
        logreg_config = config.get_model_config("logreg")
        assert logreg_config is not None
        assert "C" in logreg_config
        assert "max_iter" in logreg_config

        # Test non-existent model
        with pytest.raises(ValueError):
            config.get_model_config("non_existent_model")

    def test_config_ensure_directories(self):
        """Test directory creation."""
        config = Config()

        with tempfile.TemporaryDirectory() as temp_dir:
            config.paths.artifacts = os.path.join(temp_dir, "artifacts")
            config.paths.models = os.path.join(temp_dir, "models")
            config.paths.logs = os.path.join(temp_dir, "logs")

            config.ensure_directories()

            assert os.path.exists(config.paths.artifacts)
            assert os.path.exists(config.paths.models)
            assert os.path.exists(config.paths.logs)


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_train_val_test_split(self):
        """Test that train/val/test split validation works."""
        with pytest.raises(ValueError):
            DataConfig(
                train_size=0.5,
                val_size=0.3,
                test_size=0.3  # Total > 1.0
            )

    def test_invalid_cv_folds(self):
        """Test that CV folds validation works."""
        with pytest.raises(ValueError):
            ModelConfig(cv_folds=0)

    def test_invalid_optuna_trials(self):
        """Test that Optuna trials validation works."""
        with pytest.raises(ValueError):
            ModelConfig(optuna_trials=0)

    def test_invalid_random_seed(self):
        """Test that random seed validation works."""
        with pytest.raises(ValueError):
            DataConfig(random_seed=-1)
