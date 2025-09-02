"""
Model promotion module for MLOps project.

Provides comprehensive model promotion workflow including MLflow Model Registry
integration, promotion gates, stage transitions, and model archiving.
"""

import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from ..utils.io import ensure_dir, save_json
from ..utils.logging import get_logger, log_model_promotion, log_model_registration

logger = get_logger(__name__)


class ModelPromoter:
    """
    Comprehensive model promoter for MLflow Model Registry.
    
    Handles model registration, promotion gates, stage transitions,
    and model lifecycle management.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the model promoter.
        
        Args:
            config: Configuration dictionary containing promotion parameters
        """
        self.config = config
        self.registry_config = config.get("registry", {})
        self.mlflow_config = config.get("mlflow", {})
        self.paths_config = config.get("paths", {})

        # Extract configuration
        self.experiment_name = self.mlflow_config.get("experiment_name", "mlops-experiment")
        self.registry_name = self.registry_config.get("registry_name", "mlops-registry")
        self.promotion_rules = self.registry_config.get("promotion_rules", {})
        self.stages = self.registry_config.get("stages", ["None", "Staging", "Production"])
        self.archive_previous = self.registry_config.get("archive_previous", True)

        # Ensure output directory
        self.output_path = self.paths_config.get("registry", "artifacts/registry")
        ensure_dir(self.output_path)

        # Initialize MLflow
        self._setup_mlflow()

        # Initialize results
        self.promotion_results = {}

        logger.info("Initialized ModelPromoter")

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking and registry."""
        try:
            # Set tracking URI
            tracking_uri = self.mlflow_config.get("tracking_uri", "sqlite:///mlruns.db")
            mlflow.set_tracking_uri(tracking_uri)

            # Set experiment
            mlflow.set_experiment(self.experiment_name)

            # Set registry URI if different from tracking URI
            registry_uri = self.mlflow_config.get("registry_uri", tracking_uri)
            if registry_uri != tracking_uri:
                mlflow.set_registry_uri(registry_uri)

            logger.info(f"MLflow setup complete: experiment={self.experiment_name}, registry={self.registry_name}")

        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise

    def register_model(
        self,
        model: Any,
        model_name: str,
        model_type: str = "sklearn",
        feature_engineer: Any = None,
        evaluation_results: dict[str, Any] = None,
        metadata: dict[str, Any] = None
    ) -> str:
        """
        Register a model to MLflow Model Registry.
        
        Args:
            model: Trained model to register
            model_name: Name for the model
            model_type: Type of model ("sklearn", "pytorch", "custom")
            feature_engineer: Feature engineering pipeline (optional)
            evaluation_results: Model evaluation results (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Model version URI
        """
        logger.info(f"Registering model: {model_name}")

        try:
            # Start MLflow run
            with mlflow.start_run():
                # Log model parameters
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())

                # Log evaluation results if available
                if evaluation_results:
                    self._log_evaluation_metadata(evaluation_results)

                # Log additional metadata
                if metadata:
                    mlflow.log_params(metadata)

                # Register model based on type
                if model_type == "sklearn":
                    model_uri = self._register_sklearn_model(model, model_name)
                elif model_type == "pytorch":
                    model_uri = self._register_pytorch_model(model, model_name)
                else:
                    model_uri = self._register_custom_model(model, model_name)

                # Log feature engineer if available
                if feature_engineer:
                    self._log_feature_engineer(feature_engineer, model_name)

                # Log model registration
                log_model_registration(model_name, model_uri, model_type)

                # Store registration result
                self.promotion_results[model_name] = {
                    "status": "registered",
                    "model_uri": model_uri,
                    "model_type": model_type,
                    "registration_timestamp": pd.Timestamp.now().isoformat()
                }

                logger.info(f"Successfully registered model: {model_name} -> {model_uri}")
                return model_uri

        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            self.promotion_results[model_name] = {
                "status": "failed",
                "error": str(e),
                "registration_timestamp": pd.Timestamp.now().isoformat()
            }
            raise

    def _register_sklearn_model(self, model: Any, model_name: str) -> str:
        """Register a scikit-learn model."""
        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=self._create_input_example(model),
            signature=self._create_model_signature(model)
        )

        # Get model URI
        model_uri = f"models:/{model_name}/latest"
        return model_uri

    def _register_pytorch_model(self, model: Any, model_name: str) -> str:
        """Register a PyTorch model."""
        # Log model
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=self._create_input_example(model),
            signature=self._create_model_signature(model)
        )

        # Get model URI
        model_uri = f"models:/{model_name}/latest"
        return model_uri

    def _register_custom_model(self, model: Any, model_name: str) -> str:
        """Register a custom model using pickle."""
        import pickle

        # Save model to temporary file
        model_path = f"temp_{model_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Log artifact
        mlflow.log_artifact(model_path, artifact_path="model")

        # Clean up
        os.remove(model_path)

        # Get model URI
        model_uri = f"models:/{model_name}/latest"
        return model_uri

    def _create_input_example(self, model: Any) -> np.ndarray:
        """Create input example for model signature."""
        try:
            # Try to get feature names from model
            if hasattr(model, 'feature_names_in_'):
                n_features = len(model.feature_names_in_)
            elif hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
            else:
                n_features = 10  # Default

            # Create sample input
            input_example = np.random.randn(1, n_features)
            return input_example

        except Exception:
            # Fallback to default
            return np.random.randn(1, 10)

    def _create_model_signature(self, model: Any) -> Any:
        """Create model signature for MLflow."""
        try:
            from mlflow.models.signature import ModelSignature
            from mlflow.types.schema import Schema, TensorSpec

            # Try to get feature names from model
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = [f"feature_{i}" for i in range(10)]

            # Create input schema
            input_schema = Schema([
                TensorSpec(np.dtype(np.float64), (-1, len(feature_names)), name)
                for name in feature_names
            ])

            # Create output schema (assuming single output)
            output_schema = Schema([
                TensorSpec(np.dtype(np.float64), (-1, 1), "prediction")
            ])

            return ModelSignature(inputs=input_schema, outputs=output_schema)

        except Exception:
            # Return None if signature creation fails
            return None

    def _log_evaluation_metadata(self, evaluation_results: dict[str, Any]) -> None:
        """Log evaluation results as MLflow metadata."""
        try:
            # Log metrics
            if "metrics" in evaluation_results:
                mlflow.log_metrics(evaluation_results["metrics"])

            # Log evaluation parameters
            mlflow.log_params({
                "evaluation_task_type": evaluation_results.get("task_type", "unknown"),
                "evaluation_timestamp": evaluation_results.get("evaluation_timestamp", pd.Timestamp.now().isoformat())
            })

            # Log evaluation results as artifact
            eval_file = "evaluation_results.json"
            save_json(evaluation_results, eval_file)
            mlflow.log_artifact(eval_file)

        except Exception as e:
            logger.warning(f"Failed to log evaluation metadata: {e}")

    def _log_feature_engineer(self, feature_engineer: Any, model_name: str) -> None:
        """Log feature engineering pipeline as artifact."""
        try:
            # Save feature engineer
            feature_file = f"{model_name}_feature_engineer.pkl"
            import pickle
            with open(feature_file, 'wb') as f:
                pickle.dump(feature_engineer, f)

            # Log as artifact
            mlflow.log_artifact(feature_file)

            # Clean up
            os.remove(feature_file)

        except Exception as e:
            logger.warning(f"Failed to log feature engineer: {e}")

    def promote_model(
        self,
        model_name: str,
        target_stage: str = "Production",
        evaluation_results: dict[str, Any] = None
    ) -> bool:
        """
        Promote a model to a target stage after passing promotion gates.
        
        Args:
            model_name: Name of the model to promote
            target_stage: Target stage for promotion
            evaluation_results: Evaluation results for promotion gates
            
        Returns:
            True if promotion successful, False otherwise
        """
        logger.info(f"Attempting to promote {model_name} to {target_stage}")

        try:
            # Check if model exists
            if not self._model_exists(model_name):
                logger.error(f"Model {model_name} not found in registry")
                return False

            # Apply promotion gates
            if not self._apply_promotion_gates(model_name, target_stage, evaluation_results):
                logger.warning(f"Model {model_name} failed promotion gates for {target_stage}")
                return False

            # Archive previous model in target stage if configured
            if self.archive_previous and target_stage == "Production":
                self._archive_previous_production(model_name)

            # Transition model to target stage
            self._transition_model_stage(model_name, target_stage)

            # Log promotion
            log_model_promotion(model_name, target_stage)

            # Update promotion results
            if model_name in self.promotion_results:
                self.promotion_results[model_name]["promoted_to"] = target_stage
                self.promotion_results[model_name]["promotion_timestamp"] = pd.Timestamp.now().isoformat()

            logger.info(f"Successfully promoted {model_name} to {target_stage}")
            return True

        except Exception as e:
            logger.error(f"Failed to promote {model_name} to {target_stage}: {e}")
            return False

    def _model_exists(self, model_name: str) -> bool:
        """Check if model exists in registry."""
        try:
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions(model_name)
            return len(latest_versions) > 0
        except Exception:
            return False

    def _apply_promotion_gates(
        self,
        model_name: str,
        target_stage: str,
        evaluation_results: dict[str, Any] = None
    ) -> bool:
        """Apply promotion gates to determine if model can be promoted."""
        if not self.promotion_rules:
            logger.info("No promotion rules defined, allowing promotion")
            return True

        stage_rules = self.promotion_rules.get(target_stage, {})
        if not stage_rules:
            logger.info(f"No promotion rules for {target_stage}, allowing promotion")
            return True

        logger.info(f"Applying promotion gates for {target_stage}")

        # Check metric thresholds
        if evaluation_results and "metrics" in evaluation_results:
            metrics = evaluation_results["metrics"]

            for metric_name, threshold in stage_rules.get("metric_thresholds", {}).items():
                if metric_name in metrics:
                    metric_value = metrics[metric_name]
                    if metric_value < threshold:
                        logger.warning(f"Metric {metric_name} ({metric_value:.4f}) below threshold ({threshold})")
                        return False
                    logger.info(f"Metric {metric_name} ({metric_value:.4f}) above threshold ({threshold})")
                else:
                    logger.warning(f"Required metric {metric_name} not found in evaluation results")
                    return False

        # Check additional rules
        additional_rules = stage_rules.get("additional_rules", [])
        for rule in additional_rules:
            if not self._evaluate_rule(rule, evaluation_results):
                logger.warning(f"Failed additional rule: {rule}")
                return False

        logger.info("All promotion gates passed")
        return True

    def _evaluate_rule(self, rule: dict[str, Any], evaluation_results: dict[str, Any]) -> bool:
        """Evaluate a custom promotion rule."""
        rule_type = rule.get("type", "unknown")

        if rule_type == "min_samples":
            min_samples = rule.get("value", 1000)
            if evaluation_results and "predictions" in evaluation_results:
                n_samples = len(evaluation_results["predictions"]["y_true"])
                return n_samples >= min_samples

        elif rule_type == "max_null_percentage":
            max_null_pct = rule.get("value", 0.1)
            # This would need to be implemented based on data validation results
            return True  # Placeholder

        elif rule_type == "custom_function":
            # Custom function evaluation
            try:
                func_name = rule.get("function")
                if func_name and hasattr(self, func_name):
                    func = getattr(self, func_name)
                    return func(evaluation_results)
            except Exception as e:
                logger.warning(f"Custom rule evaluation failed: {e}")
                return False

        return True  # Default to allowing promotion

    def _archive_previous_production(self, model_name: str) -> None:
        """Archive the previous Production model."""
        try:
            client = mlflow.tracking.MlflowClient()

            # Find Production models
            production_models = client.search_model_versions(
                f"name='{model_name}' AND stage='Production'"
            )

            for model_version in production_models:
                logger.info(f"Archiving previous Production model: {model_version.name} v{model_version.version}")

                # Transition to Archived stage
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Archived"
                )

                # Add archive metadata
                client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=f"Archived on {pd.Timestamp.now().isoformat()}"
                )

        except Exception as e:
            logger.warning(f"Failed to archive previous Production models: {e}")

    def _transition_model_stage(self, model_name: str, target_stage: str) -> None:
        """Transition model to target stage."""
        try:
            client = mlflow.tracking.MlflowClient()

            # Get latest version
            latest_versions = client.get_latest_versions(model_name)
            if not latest_versions:
                raise ValueError(f"No versions found for model {model_name}")

            latest_version = latest_versions[0]

            # Transition stage
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage=target_stage
            )

            # Update description
            client.update_model_version(
                name=model_name,
                version=latest_version.version,
                description=f"Promoted to {target_stage} on {pd.Timestamp.now().isoformat()}"
            )

            logger.info(f"Transitioned {model_name} v{latest_version.version} to {target_stage}")

        except Exception as e:
            logger.error(f"Failed to transition {model_name} to {target_stage}: {e}")
            raise

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get information about a registered model."""
        try:
            client = mlflow.tracking.MlflowClient()

            # Get model details
            model_details = client.get_registered_model(model_name)

            # Get all versions
            all_versions = client.search_model_versions(f"name='{model_name}'")

            # Get latest versions by stage
            latest_versions = client.get_latest_versions(model_name)

            model_info = {
                "name": model_name,
                "description": model_details.description,
                "tags": model_details.tags,
                "creation_timestamp": model_details.creation_timestamp,
                "last_updated_timestamp": model_details.last_updated_timestamp,
                "total_versions": len(all_versions),
                "stages": {},
                "versions": []
            }

            # Organize by stage
            for stage in self.stages:
                stage_versions = [v for v in all_versions if v.current_stage == stage]
                if stage_versions:
                    latest_stage_version = max(stage_versions, key=lambda x: x.version)
                    model_info["stages"][stage] = {
                        "version": latest_stage_version.version,
                        "run_id": latest_stage_version.run_id,
                        "status": latest_stage_version.status,
                        "transition_timestamp": latest_stage_version.last_updated_timestamp
                    }

            # Add version details
            for version in all_versions:
                model_info["versions"].append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "status": version.status,
                    "created_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp
                })

            return model_info

        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {"error": str(e)}

    def list_models(self, stage: str = None) -> list[dict[str, Any]]:
        """List all models in the registry, optionally filtered by stage."""
        try:
            client = mlflow.tracking.MlflowClient()

            # Get all registered models
            all_models = client.list_registered_models()

            models_info = []
            for model in all_models:
                model_info = self.get_model_info(model.name)
                if "error" not in model_info:
                    # Filter by stage if specified
                    if stage is None or stage in model_info["stages"]:
                        models_info.append(model_info)

            return models_info

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def delete_model(self, model_name: str, version: int = None) -> bool:
        """Delete a model or specific version."""
        try:
            client = mlflow.tracking.MlflowClient()

            if version is None:
                # Delete entire model
                client.delete_registered_model(model_name)
                logger.info(f"Deleted model: {model_name}")
            else:
                # Delete specific version
                client.delete_model_version(model_name, version)
                logger.info(f"Deleted model version: {model_name} v{version}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False

    def get_promotion_summary(self) -> dict[str, Any]:
        """Get a summary of promotion activities."""
        if not self.promotion_results:
            return {"status": "no_promotions_performed"}

        summary = {
            "total_models": len(self.promotion_results),
            "successful_registrations": len([r for r in self.promotion_results.values() if r.get("status") == "registered"]),
            "failed_registrations": len([r for r in self.promotion_results.values() if r.get("status") == "failed"]),
            "promoted_models": len([r for r in self.promotion_results.values() if "promoted_to" in r]),
            "models": self.promotion_results
        }

        return summary

    def save_promotion_results(self, path: str = None) -> None:
        """Save promotion results to disk."""
        if path is None:
            path = self.output_path

        try:
            results_path = Path(path) / "promotion_results.yaml"
            with open(results_path, 'w') as f:
                yaml.dump(self.promotion_results, f, default_flow_style=False)

            logger.info(f"Saved promotion results to {path}")

        except Exception as e:
            logger.error(f"Failed to save promotion results: {e}")

    def load_promotion_results(self, path: str = None) -> None:
        """Load promotion results from disk."""
        if path is None:
            path = self.output_path

        try:
            results_path = Path(path) / "promotion_results.yaml"
            if results_path.exists():
                with open(results_path) as f:
                    self.promotion_results = yaml.safe_load(f)

                logger.info(f"Loaded promotion results from {path}")

        except Exception as e:
            logger.error(f"Failed to load promotion results: {e}")

    def create_promotion_report(self) -> str:
        """Create a comprehensive promotion report."""
        try:
            summary = self.get_promotion_summary()

            report = f"""# Model Promotion Report

## Summary
- **Total Models**: {summary['total_models']}
- **Successful Registrations**: {summary['successful_registrations']}
- **Failed Registrations**: {summary['failed_registrations']}
- **Promoted Models**: {summary['promoted_models']}

## Model Details
"""

            for model_name, result in summary.get("models", {}).items():
                report += f"""
### {model_name}
- **Status**: {result.get('status', 'unknown')}
- **Model Type**: {result.get('model_type', 'unknown')}
- **Registration Time**: {result.get('registration_timestamp', 'unknown')}
"""

                if "promoted_to" in result:
                    report += f"- **Promoted To**: {result['promoted_to']}\n"
                    report += f"- **Promotion Time**: {result.get('promotion_timestamp', 'unknown')}\n"

                if "error" in result:
                    report += f"- **Error**: {result['error']}\n"

                if "model_uri" in result:
                    report += f"- **Model URI**: {result['model_uri']}\n"

            # Save report
            report_path = Path(self.output_path) / "promotion_report.md"
            with open(report_path, 'w') as f:
                f.write(report)

            # Log to MLflow
            mlflow.log_artifact(str(report_path))

            logger.info("Created promotion report")
            return str(report_path)

        except Exception as e:
            logger.warning(f"Failed to create promotion report: {e}")
            return ""
