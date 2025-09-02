"""
Model training module for MLOps project.

Provides comprehensive model training with multiple candidates, cross-validation,
Optuna hyperparameter optimization, and MLflow logging.
"""

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ..data.features import FeatureEngineer
from ..modeling.dl.torch_estimator import TorchMLPClassifier, TorchMLPRegressor
from ..utils.io import ensure_dir, save_pickle
from ..utils.logging import get_logger, log_training_end, log_training_start
from ..utils.seed import set_seed

logger = get_logger(__name__)


class ModelTrainer:
    """
    Comprehensive model trainer with multiple candidates and hyperparameter optimization.
    
    Supports both classical ML and deep learning models, cross-validation,
    Optuna tuning, and comprehensive MLflow logging.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration dictionary containing training parameters
        """
        self.config = config
        self.model_config = config.get("models", {})
        self.evaluation_config = config.get("evaluation", {})
        self.mlflow_config = config.get("mlflow", {})
        self.paths_config = config.get("paths", {})

        # Extract configuration
        self.model_roster = self.model_config.get("roster", ["logreg", "rf", "torch_mlp"])
        self.cv_folds = self.evaluation_config.get("cv_folds", 5)
        self.scoring_metric = self.evaluation_config.get("scoring_metric", "auto")
        self.optuna_trials = self.model_config.get("optuna_trials", 50)

        # Set random seed
        set_seed(config.get("system", {}).get("seed", 42))

        # Initialize components
        self.feature_engineer = None
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.training_results = {}

        # Ensure output directories
        self.output_path = self.paths_config.get("models", "artifacts/models")
        ensure_dir(self.output_path)

        logger.info(f"Initialized ModelTrainer with roster: {self.model_roster}")

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> dict[str, Any]:
        """
        Train multiple model candidates with cross-validation and hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary containing training results and best model
        """
        logger.info("Starting model training pipeline")

        # Log training start
        log_training_start(len(X_train), len(X_train.columns), y_train.dtype)

        # Determine task type
        task_type = self._determine_task_type(y_train)
        logger.info(f"Detected task type: {task_type}")

        # Initialize feature engineering
        self.feature_engineer = FeatureEngineer(self.config)
        X_train_processed, feature_names = self.feature_engineer.fit_transform(X_train, y_train.name)

        if X_val is not None:
            X_val_processed = self.feature_engineer.transform(X_val, y_val.name)
        else:
            X_val_processed = None

        # Train each model candidate
        for model_name in self.model_roster:
            try:
                logger.info(f"Training model: {model_name}")

                # Create and train model
                model = self._create_model(model_name, task_type, X_train_processed.shape[1])
                model = self._train_model(
                    model, model_name, X_train_processed, y_train,
                    X_val_processed, y_val, task_type
                )

                # Store model
                self.models[model_name] = model

                # Update best model if better
                if self.training_results[model_name]["cv_score"] > self.best_score:
                    self.best_score = self.training_results[model_name]["cv_score"]
                    self.best_model = model
                    logger.info(f"New best model: {model_name} with score: {self.best_score:.4f}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                self.training_results[model_name] = {
                    "status": "failed",
                    "error": str(e),
                    "cv_score": -np.inf
                }

        # Save best model
        if self.best_model is not None:
            self._save_best_model()

        # Log training end
        log_training_end(self.best_score, task_type)

        # Generate training report
        self._generate_training_report()

        return {
            "best_model": self.best_model,
            "best_score": self.best_score,
            "training_results": self.training_results,
            "feature_engineer": self.feature_engineer
        }

    def _determine_task_type(self, y: pd.Series) -> str:
        """Determine the machine learning task type."""
        # Check config override first
        config_task = self.config.get("data", {}).get("task_type", "auto")
        if config_task != "auto":
            return config_task

        # Auto-detect from target dtype
        if y.dtype in ['object', 'category']:
            unique_values = y.nunique()
            if unique_values == 2:
                return "binary_classification"
            elif unique_values > 2:
                return "multiclass_classification"
            else:
                return "classification"
        else:
            return "regression"

    def _create_model(self, model_name: str, task_type: str, n_features: int) -> BaseEstimator:
        """Create a model instance based on name and task type."""
        if model_name == "logreg":
            if "classification" in task_type:
                return LogisticRegression(random_state=42, max_iter=1000)
            else:
                return LinearRegression()

        elif model_name == "rf":
            if "classification" in task_type:
                return RandomForestClassifier(random_state=42, n_jobs=-1)
            else:
                return RandomForestRegressor(random_state=42, n_jobs=-1)

        elif model_name == "xgboost":
            try:
                import xgboost as xgb
                if "classification" in task_type:
                    return xgb.XGBClassifier(random_state=42, n_jobs=-1)
                else:
                    return xgb.XGBRegressor(random_state=42, n_jobs=-1)
            except ImportError:
                logger.warning("XGBoost not available, skipping")
                raise ImportError("XGBoost not installed")

        elif model_name == "lightgbm":
            try:
                import lightgbm as lgb
                if "classification" in task_type:
                    return lgb.LGBMClassifier(random_state=42, n_jobs=-1)
                else:
                    return lgb.LGBMRegressor(random_state=42, n_jobs=-1)
            except ImportError:
                logger.warning("LightGBM not available, skipping")
                raise ImportError("LightGBM not installed")

        elif model_name == "torch_mlp":
            if "classification" in task_type:
                return TorchMLPClassifier(
                    input_dim=n_features,
                    hidden_dims=self.model_config.get("torch_mlp", {}).get("hidden_dims", [128, 64]),
                    num_classes=y.nunique() if "classification" in task_type else 1,
                    **self.model_config.get("torch_mlp", {})
                )
            else:
                return TorchMLPRegressor(
                    input_dim=n_features,
                    hidden_dims=self.model_config.get("torch_mlp", {}).get("hidden_dims", [128, 64]),
                    **self.model_config.get("torch_mlp", {})
                )

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _train_model(
        self,
        model: BaseEstimator,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        task_type: str = "classification"
    ) -> BaseEstimator:
        """Train a single model with cross-validation and hyperparameter tuning."""
        logger.info(f"Training {model_name} with cross-validation")

        # Create full pipeline with feature engineering
        pipeline = Pipeline([
            ('features', self.feature_engineer.transformer),
            ('model', model)
        ])

        # Perform cross-validation
        cv_scores = self._cross_validate(pipeline, X_train, y_train, task_type)
        cv_score = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # Hyperparameter tuning with Optuna if configured
        if self.model_config.get("use_optuna", True):
            logger.info(f"Performing Optuna hyperparameter optimization for {model_name}")
            best_params = self._optimize_hyperparameters(
                model_name, X_train, y_train, task_type
            )

            # Update model with best parameters
            if best_params:
                model.set_params(**best_params)
                pipeline = Pipeline([
                    ('features', self.feature_engineer.transformer),
                    ('model', model)
                ])

                # Re-run cross-validation with best parameters
                cv_scores = self._cross_validate(pipeline, X_train, y_train, task_type)
                cv_score = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                logger.info(f"Best CV score after optimization: {cv_score:.4f} ± {cv_std:.4f}")

        # Final training on full training set
        pipeline.fit(X_train, y_train)

        # Validation set evaluation if available
        val_score = None
        if X_val is not None and y_val is not None:
            val_score = self._evaluate_model(pipeline, X_val, y_val, task_type)
            logger.info(f"Validation score: {val_score:.4f}")

        # Store training results
        self.training_results[model_name] = {
            "status": "success",
            "cv_scores": cv_scores,
            "cv_score": cv_score,
            "cv_std": cv_std,
            "val_score": val_score,
            "task_type": task_type,
            "model_params": model.get_params(),
            "pipeline": pipeline
        }

        # Log to MLflow
        self._log_model_training(model_name, pipeline, cv_scores, val_score)

        return pipeline

    def _cross_validate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ) -> list[float]:
        """Perform cross-validation with appropriate scoring."""
        # Determine scoring metric
        if self.scoring_metric == "auto":
            if "classification" in task_type:
                if y.nunique() == 2:
                    scoring = "roc_auc"
                else:
                    scoring = "f1_weighted"
            else:
                scoring = "r2"
        else:
            scoring = self.scoring_metric

        # Choose CV strategy
        if "classification" in task_type and y.nunique() > 2:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        return scores.tolist()

    def _optimize_hyperparameters(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ) -> dict[str, Any] | None:
        """Optimize hyperparameters using Optuna."""
        try:
            # Get model-specific search space
            search_space = self._get_search_space(model_name, task_type)

            if not search_space:
                logger.info(f"No search space defined for {model_name}")
                return None

            # Create objective function
            def objective(trial):
                # Sample hyperparameters
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config["type"] == "categorical":
                        params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])
                    elif param_config["type"] == "int":
                        params[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"])
                    elif param_config["type"] == "float":
                        params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"])
                    elif param_config["type"] == "log":
                        params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"], log=True)

                # Create model with sampled parameters
                model = self._create_model(model_name, task_type, X.shape[1])
                model.set_params(**params)

                # Create pipeline
                pipeline = Pipeline([
                    ('features', self.feature_engineer.transformer),
                    ('model', model)
                ])

                # Cross-validate
                cv_scores = self._cross_validate(pipeline, X, y, task_type)
                return np.mean(cv_scores)

            # Run optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.optuna_trials)

            logger.info(f"Best trial: {study.best_trial.value:.4f}")
            logger.info(f"Best parameters: {study.best_trial.params}")

            return study.best_trial.params

        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed for {model_name}: {e}")
            return None

    def _get_search_space(self, model_name: str, task_type: str) -> dict[str, Any]:
        """Get hyperparameter search space for a specific model."""
        model_config = self.model_config.get(model_name, {})
        search_space = model_config.get("optuna_search_space", {})

        if not search_space:
            # Default search spaces
            if model_name == "logreg" and "classification" in task_type:
                search_space = {
                    "C": {"type": "log", "low": 1e-4, "high": 1e2},
                    "penalty": {"type": "categorical", "choices": ["l1", "l2", "elasticnet"]},
                    "solver": {"type": "categorical", "choices": ["liblinear", "saga"]}
                }
            elif model_name == "rf":
                search_space = {
                    "n_estimators": {"type": "int", "low": 50, "high": 300},
                    "max_depth": {"type": "int", "low": 3, "high": 20},
                    "min_samples_split": {"type": "int", "low": 2, "high": 20},
                    "min_samples_leaf": {"type": "int", "low": 1, "high": 10}
                }
            elif model_name == "torch_mlp":
                search_space = {
                    "learning_rate": {"type": "log", "low": 1e-4, "high": 1e-1},
                    "weight_decay": {"type": "log", "low": 1e-6, "high": 1e-2},
                    "dropout": {"type": "float", "low": 0.0, "high": 0.5}
                }

        return search_space

    def _evaluate_model(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ) -> float:
        """Evaluate a trained model on validation data."""
        y_pred = model.predict(X)

        if "classification" in task_type:
            if y.nunique() == 2:
                score = roc_auc_score(y, y_pred)
            else:
                score = f1_score(y, y_pred, average='weighted')
        else:
            score = r2_score(y, y_pred)

        return score

    def _log_model_training(
        self,
        model_name: str,
        model: BaseEstimator,
        cv_scores: list[float],
        val_score: float | None
    ) -> None:
        """Log model training information to MLflow."""
        try:
            # Log model parameters
            mlflow.log_params({
                f"{model_name}_cv_folds": self.cv_folds,
                f"{model_name}_cv_score": np.mean(cv_scores),
                f"{model_name}_cv_std": np.std(cv_scores)
            })

            if val_score is not None:
                mlflow.log_metrics({
                    f"{model_name}_val_score": val_score
                })

            # Log model artifact
            model_path = f"{model_name}_model.pkl"
            save_pickle(model, model_path)
            mlflow.log_artifact(model_path)

            # Log feature engineering info
            if self.feature_engineer:
                feature_info = self.feature_engineer.get_transformer_summary()
                mlflow.log_params({
                    f"{model_name}_feature_dim": feature_info.get("total_features", 0),
                    f"{model_name}_numeric_strategy": feature_info.get("numeric_strategy", "unknown"),
                    f"{model_name}_categorical_strategy": feature_info.get("categorical_strategy", "unknown")
                })

            logger.info(f"Successfully logged {model_name} training to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log {model_name} training to MLflow: {e}")

    def _save_best_model(self) -> None:
        """Save the best performing model."""
        if self.best_model is None:
            return

        try:
            # Save best model
            best_model_path = Path(self.output_path) / "best_model.pkl"
            save_pickle(self.best_model, best_model_path)

            # Save feature engineer
            feature_engineer_path = Path(self.output_path) / "feature_engineer.pkl"
            save_pickle(self.feature_engineer, feature_engineer_path)

            # Save training results
            results_path = Path(self.output_path) / "training_results.yaml"
            with open(results_path, 'w') as f:
                yaml.dump(self.training_results, f, default_flow_style=False)

            logger.info(f"Saved best model and artifacts to {self.output_path}")

        except Exception as e:
            logger.error(f"Failed to save best model: {e}")

    def _generate_training_report(self) -> None:
        """Generate comprehensive training report."""
        try:
            # Create comparison plot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Cross-Validation Scores",
                    "Model Comparison",
                    "Training Results Summary",
                    "Feature Engineering Info"
                )
            )

            # CV Scores
            model_names = list(self.training_results.keys())
            cv_scores = [self.training_results[name].get("cv_score", 0) for name in model_names]
            cv_stds = [self.training_results[name].get("cv_std", 0) for name in model_names]

            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=cv_scores,
                    error_y=dict(type='data', array=cv_stds, visible=True),
                    name="CV Score",
                    marker_color='lightblue'
                ),
                row=1, col=1
            )

            # Model comparison
            if any("val_score" in self.training_results[name] for name in model_names):
                val_scores = [self.training_results[name].get("val_score", 0) for name in model_names]
                fig.add_trace(
                    go.Bar(
                        x=model_names,
                        y=val_scores,
                        name="Validation Score",
                        marker_color='lightgreen'
                    ),
                    row=1, col=2
                )

            # Training summary
            summary_data = []
            for name in model_names:
                result = self.training_results[name]
                summary_data.append([
                    name,
                    result.get("status", "unknown"),
                    f"{result.get('cv_score', 0):.4f}",
                    f"{result.get('cv_std', 0):.4f}",
                    result.get("task_type", "unknown")
                ])

            fig.add_trace(
                go.Table(
                    header=dict(values=["Model", "Status", "CV Score", "CV Std", "Task Type"]),
                    cells=dict(values=list(zip(*summary_data, strict=False)))
                ),
                row=2, col=1
            )

            # Feature engineering info
            if self.feature_engineer:
                feature_info = self.feature_engineer.get_transformer_summary()
                feature_data = [
                    ["Total Features", feature_info.get("total_features", 0)],
                    ["Numeric Strategy", feature_info.get("numeric_strategy", "unknown")],
                    ["Categorical Strategy", feature_info.get("categorical_strategy", "unknown")],
                    ["Feature Selection", feature_info.get("feature_selection_enabled", False)]
                ]

                fig.add_trace(
                    go.Table(
                        header=dict(values=["Feature Engineering", "Value"]),
                        cells=dict(values=list(zip(*feature_data, strict=False)))
                    ),
                    row=2, col=2
                )

            # Update layout
            fig.update_layout(
                height=800,
                title_text="Model Training Report",
                showlegend=True
            )

            # Save report
            report_path = Path(self.output_path) / "training_report.html"
            fig.write_html(str(report_path))

            # Log to MLflow
            mlflow.log_artifact(str(report_path))

            logger.info("Generated training report")

        except Exception as e:
            logger.warning(f"Failed to generate training report: {e}")

    def get_training_summary(self) -> dict[str, Any]:
        """Get a summary of the training results."""
        if not self.training_results:
            return {"status": "no_training_performed"}

        successful_models = [
            name for name, result in self.training_results.items()
            if result.get("status") == "success"
        ]

        failed_models = [
            name for name, result in self.training_results.items()
            if result.get("status") == "failed"
        ]

        summary = {
            "total_models": len(self.training_results),
            "successful_models": len(successful_models),
            "failed_models": len(failed_models),
            "best_model": None,
            "best_score": self.best_score,
            "task_type": None,
            "cv_folds": self.cv_folds
        }

        if successful_models:
            # Find best model
            best_model_name = max(
                successful_models,
                key=lambda x: self.training_results[x].get("cv_score", -np.inf)
            )
            summary["best_model"] = best_model_name
            summary["task_type"] = self.training_results[best_model_name].get("task_type")

        return summary

    def load_best_model(self, path: str = None) -> BaseEstimator | None:
        """Load the best model from disk."""
        if path is None:
            path = self.output_path

        try:
            best_model_path = Path(path) / "best_model.pkl"
            if best_model_path.exists():
                self.best_model = load_pickle(best_model_path)
                logger.info(f"Loaded best model from {best_model_path}")
                return self.best_model
            else:
                logger.warning(f"Best model not found at {best_model_path}")
                return None
        except Exception as e:
            logger.error(f"Failed to load best model: {e}")
            return None
