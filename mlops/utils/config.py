"""Configuration management for the MLOps project."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    """Data configuration settings."""

    paths: list[str] = Field(default=["./data/*.csv"], description="Paths to CSV files")
    target_column: str | None = Field(default=None, description="Target column name")
    task_type: str = Field(default="auto", description="Task type: auto, classification, or regression")
    merge_key: str | None = Field(default=None, description="Merge key for multiple CSVs")

    class SplitConfig(BaseModel):
        train_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
        val_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
        test_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
        random_state: int = Field(default=42)
        time_aware: bool = Field(default=False)
        timestamp_column: str | None = Field(default=None)

    class CVConfig(BaseModel):
        n_folds: int = Field(default=5, ge=2)
        random_state: int = Field(default=42)
        stratified: bool = Field(default=True)

    split: SplitConfig = Field(default_factory=SplitConfig)
    cv: CVConfig = Field(default_factory=CVConfig)

    @validator('paths')
    def validate_paths(cls, v: list[str]) -> list[str]:
        """Validate that paths are not empty."""
        if not v:
            raise ValueError("At least one data path must be specified")
        return v

    @validator('split')
    def validate_split_ratios(cls, v: SplitConfig) -> SplitConfig:
        """Validate that split ratios sum to approximately 1.0."""
        total = v.train_ratio + v.val_ratio + v.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return v


class ModelConfig(BaseModel):
    """Model configuration settings."""

    roster: list[str] = Field(default=["logreg", "rf", "torch_mlp"])

    class LogRegConfig(BaseModel):
        solver: str = Field(default="lbfgs")
        max_iter: int = Field(default=1000, ge=1)
        random_state: int = Field(default=42)
        optuna_trials: int = Field(default=20, ge=1)
        optuna_params: dict[str, Any] = Field(default_factory=dict)

    class RFConfig(BaseModel):
        n_estimators: int = Field(default=100, ge=1)
        max_depth: int | None = Field(default=None, ge=1)
        min_samples_split: int = Field(default=2, ge=2)
        min_samples_leaf: int = Field(default=1, ge=1)
        random_state: int = Field(default=42)
        optuna_trials: int = Field(default=30, ge=1)
        optuna_params: dict[str, Any] = Field(default_factory=dict)

    class TorchMLPConfig(BaseModel):
        hidden_dims: list[int] = Field(default=[128, 64, 32])
        dropout: float = Field(default=0.2, ge=0.0, le=1.0)
        activation: str = Field(default="relu")
        batch_norm: bool = Field(default=True)
        epochs: int = Field(default=100, ge=1)
        batch_size: int = Field(default=32, ge=1)
        learning_rate: float = Field(default=0.001, gt=0.0)
        weight_decay: float = Field(default=0.0001, ge=0.0)
        patience: int = Field(default=10, ge=1)
        min_delta: float = Field(default=0.001, ge=0.0)
        amp: bool = Field(default=True)
        gradient_clip: float = Field(default=1.0, gt=0.0)
        scheduler: str = Field(default="reduce_lr_on_plateau")
        optuna_trials: int = Field(default=20, ge=1)
        optuna_params: dict[str, Any] = Field(default_factory=dict)

    logreg: LogRegConfig = Field(default_factory=LogRegConfig)
    rf: RFConfig = Field(default_factory=RFConfig)
    torch_mlp: TorchMLPConfig = Field(default_factory=TorchMLPConfig)


class EvaluationConfig(BaseModel):
    """Evaluation configuration settings."""

    class MetricsConfig(BaseModel):
        classification: list[str] = Field(default=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
        regression: list[str] = Field(default=["rmse", "mae", "r2", "mape"])

    class ThresholdConfig(BaseModel):
        enabled: bool = Field(default=True)
        metric: str = Field(default="f1")
        beta: float = Field(default=1.0, gt=0.0)

    class ExplainabilityConfig(BaseModel):
        enabled: bool = Field(default=True)
        shap_samples: int = Field(default=100, ge=1)
        permutation_importance: bool = Field(default=True)
        feature_importance: bool = Field(default=True)

    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    threshold_optimization: ThresholdConfig = Field(default_factory=ThresholdConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)


class RegistryConfig(BaseModel):
    """Model registry configuration settings."""

    experiment_name: str = Field(default="mlops-project")
    registry_name: str = Field(default="mlops-models")

    class PromotionConfig(BaseModel):
        staging_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
        production_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
        metric: str = Field(default="f1")

    class VersioningConfig(BaseModel):
        archive_previous: bool = Field(default=True)
        max_versions: int = Field(default=10, ge=1)

    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)


class MLflowConfig(BaseModel):
    """MLflow configuration settings."""

    tracking_uri: str = Field(default="./mlruns")
    artifact_location: str = Field(default="./mlruns")
    experiment_name: str = Field(default="mlops-project")
    log_artifacts: bool = Field(default=True)
    log_model_signature: bool = Field(default=True)
    log_input_example: bool = Field(default=True)
    custom_metrics: bool = Field(default=True)
    custom_artifacts: bool = Field(default=True)


class LoggingConfig(BaseModel):
    """Logging configuration settings."""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: str = Field(default="./logs/mlops.log")
    structured: bool = Field(default=True)
    json_format: bool = Field(default=False)
    mlflow_logging: bool = Field(default=True)


class FeaturesConfig(BaseModel):
    """Feature engineering configuration settings."""

    class NumericConfig(BaseModel):
        imputation: str = Field(default="median")
        scaling: str = Field(default="standard")
        outlier_handling: str = Field(default="iqr")

    class CategoricalConfig(BaseModel):
        imputation: str = Field(default="constant")
        encoding: str = Field(default="onehot")
        handle_unknown: str = Field(default="ignore")

    class SelectionConfig(BaseModel):
        enabled: bool = Field(default=False)
        method: str = Field(default="mutual_info")
        k_features: int = Field(default=20, ge=1)

    numeric: NumericConfig = Field(default_factory=NumericConfig)
    categorical: CategoricalConfig = Field(default_factory=CategoricalConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)


class SystemConfig(BaseModel):
    """System configuration settings."""

    random_seed: int = Field(default=42)
    n_jobs: int = Field(default=-1)
    memory_efficient: bool = Field(default=True)

    class GPUConfig(BaseModel):
        enabled: bool = Field(default=True)
        device: str = Field(default="auto")
        memory_fraction: float = Field(default=0.8, gt=0.0, le=1.0)

    gpu: GPUConfig = Field(default_factory=GPUConfig)
    deterministic: bool = Field(default=True)
    cudnn_deterministic: bool = Field(default=True)
    cudnn_benchmark: bool = Field(default=False)


class PathsConfig(BaseModel):
    """Paths configuration settings."""

    data_artifacts: str = Field(default="./artifacts/data")
    feature_artifacts: str = Field(default="./artifacts/features")
    model_artifacts: str = Field(default="./artifacts/models")
    reports: str = Field(default="./artifacts/reports")
    plots: str = Field(default="./artifacts/plots")
    temp: str = Field(default="./temp")
    logs: str = Field(default="./logs")


class Config(BaseModel):
    """Main configuration class for the MLOps project."""

    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    @classmethod
    def from_env(cls, config_path: str | Path = None) -> "Config":
        """Load configuration from YAML file and override with environment variables."""
        if config_path is None:
            config_path = Path("mlops/config/settings.yaml")

        config = cls.from_yaml(config_path)

        # Override with environment variables
        env_mappings = {
            "MLOPS_TARGET": ("data.target_column", str),
            "MLOPS_TASK": ("data.task_type", str),
            "MLOPS_SEED": ("system.random_seed", int),
            "MLFLOW_TRACKING_URI": ("mlflow.tracking_uri", str),
            "MLOPS_USE_AMP": ("models.torch_mlp.amp", lambda x: x.lower() == "true"),
        }

        for env_var, (config_path, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Navigate to nested config and set value
                    keys = config_path.split(".")
                    current = config
                    for key in keys[:-1]:
                        current = getattr(current, key)
                    setattr(current, keys[-1], converter(env_value))
                except (ValueError, AttributeError) as e:
                    print(f"Warning: Could not set {config_path} from {env_var}={env_value}: {e}")

        return config

    def save_yaml(self, config_path: str | Path) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2, sort_keys=False)

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        """Get configuration for a specific model."""
        if not hasattr(self.models, model_name):
            raise ValueError(f"Unknown model: {model_name}")

        model_config = getattr(self.models, model_name)
        return model_config.dict()

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for path_attr in dir(self.paths):
            if not path_attr.startswith('_'):
                path_value = getattr(self.paths, path_attr)
                Path(path_value).mkdir(parents=True, exist_ok=True)
