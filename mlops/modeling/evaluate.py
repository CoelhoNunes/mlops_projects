"""
Model evaluation module for MLOps project.

Provides comprehensive model evaluation including holdout testing,
classification and regression metrics, explainability analysis,
and model card generation.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import yaml
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_curve,
)

from ..utils.io import ensure_dir, save_json
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for holdout testing and analysis.
    
    Performs detailed evaluation including metrics calculation,
    visualization, explainability analysis, and model card generation.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the model evaluator.
        
        Args:
            config: Configuration dictionary containing evaluation parameters
        """
        self.config = config
        self.evaluation_config = config.get("evaluation", {})
        self.mlflow_config = config.get("mlflow", {})
        self.paths_config = config.get("paths", {})

        # Extract configuration
        self.metrics = self.evaluation_config.get("metrics", [])
        self.threshold_optimization = self.evaluation_config.get("threshold_optimization", {})
        self.explainability = self.evaluation_config.get("explainability", {})
        self.output_path = self.paths_config.get("evaluation", "artifacts/evaluation")

        # Ensure output directory
        ensure_dir(self.output_path)

        # Initialize results
        self.evaluation_results = {}
        self.explainability_results = {}

        logger.info("Initialized ModelEvaluator")

    def evaluate_model(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str] = None,
        model_name: str = "model"
    ) -> dict[str, Any]:
        """
        Perform comprehensive model evaluation on holdout test set.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets
            feature_names: Names of features (optional)
            model_name: Name identifier for the model
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation of {model_name}")

        # Reset results
        self.evaluation_results = {}
        self.explainability_results = {}

        # Determine task type
        task_type = self._determine_task_type(y_test)
        logger.info(f"Detected task type: {task_type}")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None

        if hasattr(model, 'predict_proba') and "classification" in task_type:
            y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba, task_type)

        # Threshold optimization for classification
        if "classification" in task_type and y_pred_proba is not None:
            threshold_results = self._optimize_thresholds(y_test, y_pred_proba, task_type)
            metrics.update(threshold_results)

        # Store evaluation results
        self.evaluation_results[model_name] = {
            "task_type": task_type,
            "metrics": metrics,
            "predictions": {
                "y_true": y_test.values.tolist(),
                "y_pred": y_pred.tolist(),
                "y_pred_proba": y_pred_proba.tolist() if y_pred_proba is not None else None
            },
            "feature_names": feature_names
        }

        # Generate visualizations
        self._generate_evaluation_plots(model_name, y_test, y_pred, y_pred_proba, task_type)

        # Perform explainability analysis
        if self.explainability.get("enabled", True):
            self._perform_explainability_analysis(
                model, X_test, y_test, feature_names, model_name
            )

        # Generate model card
        self._generate_model_card(model_name, task_type)

        # Log results to MLflow
        self._log_evaluation_results(model_name)

        logger.info(f"Completed evaluation of {model_name}")

        return self.evaluation_results[model_name]

    def _determine_task_type(self, y: pd.Series) -> str:
        """Determine the machine learning task type."""
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

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray | None,
        task_type: str
    ) -> dict[str, float]:
        """Calculate comprehensive metrics for the model."""
        metrics = {}

        if "classification" in task_type:
            # Basic classification metrics
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision_macro"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics["recall_macro"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics["f1_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)

            if y_true.nunique() == 2:
                # Binary classification specific metrics
                metrics["precision_binary"] = precision_score(y_true, y_pred, zero_division=0)
                metrics["recall_binary"] = recall_score(y_true, y_pred, zero_division=0)
                metrics["f1_binary"] = f1_score(y_true, y_pred, zero_division=0)

                if y_pred_proba is not None:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    metrics["pr_auc"] = average_precision_score(y_true, y_pred_proba[:, 1])
            else:
                # Multiclass classification specific metrics
                metrics["precision_weighted"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics["recall_weighted"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics["f1_weighted"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

                if y_pred_proba is not None:
                    metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                    metrics["roc_auc_ovo"] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')

        else:
            # Regression metrics
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)
            metrics["mape"] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return metrics

    def _optimize_thresholds(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        task_type: str
    ) -> dict[str, Any]:
        """Optimize classification thresholds for better performance."""
        if not self.threshold_optimization.get("enabled", False):
            return {}

        threshold_results = {}

        if "binary_classification" in task_type:
            # Optimize threshold for binary classification
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_f1 = 0
            best_threshold = 0.5

            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba[:, 1] >= threshold).astype(int)
                f1 = f1_score(y_true, y_pred_thresh, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            threshold_results["optimal_threshold"] = best_threshold
            threshold_results["optimal_threshold_f1"] = best_f1

            # Calculate metrics with optimal threshold
            y_pred_optimal = (y_pred_proba[:, 1] >= best_threshold).astype(int)
            threshold_results["optimal_precision"] = precision_score(y_true, y_pred_optimal, zero_division=0)
            threshold_results["optimal_recall"] = recall_score(y_true, y_pred_optimal, zero_division=0)

        return threshold_results

    def _generate_evaluation_plots(
        self,
        model_name: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray | None,
        task_type: str
    ) -> None:
        """Generate comprehensive evaluation visualizations."""
        try:
            if "classification" in task_type:
                self._generate_classification_plots(model_name, y_true, y_pred, y_pred_proba)
            else:
                self._generate_regression_plots(model_name, y_true, y_pred)

            logger.info(f"Generated evaluation plots for {model_name}")

        except Exception as e:
            logger.warning(f"Failed to generate evaluation plots: {e}")

    def _generate_classification_plots(
        self,
        model_name: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray | None
    ) -> None:
        """Generate classification-specific evaluation plots."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Confusion Matrix",
                "ROC Curve" if y_pred_proba is not None else "Classification Report",
                "Precision-Recall Curve" if y_pred_proba is not None else "Feature Importance",
                "Prediction Distribution"
            )
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=list(range(len(np.unique(y_true)))),
                y=list(range(len(np.unique(y_true)))),
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16}
            ),
            row=1, col=1
        )

        # ROC Curve
        if y_pred_proba is not None and y_pred_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name='ROC Curve',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )

            # Add diagonal line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='black', width=1, dash='dash')
                ),
                row=1, col=2
            )

        # Precision-Recall Curve
        if y_pred_proba is not None and y_pred_proba.shape[1] == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name='PR Curve',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )

        # Prediction Distribution
        fig.add_trace(
            go.Histogram(
                x=y_pred,
                name='Predictions',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Classification Evaluation - {model_name}",
            showlegend=True
        )

        # Save plot
        plot_path = Path(self.output_path) / f"{model_name}_classification_plots.html"
        fig.write_html(str(plot_path))

        # Log to MLflow
        mlflow.log_artifact(str(plot_path))

    def _generate_regression_plots(
        self,
        model_name: str,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> None:
        """Generate regression-specific evaluation plots."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Predicted vs Actual",
                "Residuals Plot",
                "Residuals Distribution",
                "Prediction Error"
            )
        )

        # Predicted vs Actual
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', size=8, opacity=0.6)
            ),
            row=1, col=1
        )

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )

        # Residuals Plot
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='green', size=8, opacity=0.6)
            ),
            row=1, col=2
        )

        # Add zero line
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()], y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=1, col=2
        )

        # Residuals Distribution
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residuals',
                marker_color='lightgreen',
                opacity=0.7
            ),
            row=2, col=1
        )

        # Prediction Error
        error_percentage = np.abs(residuals / y_true) * 100
        fig.add_trace(
            go.Histogram(
                x=error_percentage,
                name='Error %',
                marker_color='lightcoral',
                opacity=0.7
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Regression Evaluation - {model_name}",
            showlegend=True
        )

        # Save plot
        plot_path = Path(self.output_path) / f"{model_name}_regression_plots.html"
        fig.write_html(str(plot_path))

        # Log to MLflow
        mlflow.log_artifact(str(plot_path))

    def _perform_explainability_analysis(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str] = None,
        model_name: str = "model"
    ) -> None:
        """Perform explainability analysis using SHAP."""
        try:
            if not self.explainability.get("enabled", True):
                return

            logger.info(f"Performing explainability analysis for {model_name}")

            # Use provided feature names or extract from model
            if feature_names is None:
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                else:
                    feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

            # Sample data for SHAP analysis if configured
            max_samples = self.explainability.get("max_samples", 1000)
            if len(X_test) > max_samples:
                sample_indices = np.random.choice(len(X_test), max_samples, replace=False)
                X_sample = X_test.iloc[sample_indices]
                y_sample = y_test.iloc[sample_indices]
            else:
                X_sample = X_test
                y_sample = y_test

            # SHAP analysis
            if hasattr(model, 'predict_proba'):
                # For models with predict_proba
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict_proba, X_sample)
                shap_values = explainer.shap_values(X_sample)

                if isinstance(shap_values, list):
                    # Multi-class or multi-output
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                # For models without predict_proba
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict, X_sample)
                shap_values = explainer.shap_values(X_sample)

            # Generate SHAP plots
            self._generate_shap_plots(
                model_name, X_sample, shap_values, feature_names, explainer
            )

            # Store explainability results
            self.explainability_results[model_name] = {
                "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                "feature_names": feature_names,
                "explainer_type": type(explainer).__name__
            }

            logger.info(f"Completed explainability analysis for {model_name}")

        except Exception as e:
            logger.warning(f"Explainability analysis failed for {model_name}: {e}")

    def _generate_shap_plots(
        self,
        model_name: str,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        feature_names: list[str],
        explainer: Any
    ) -> None:
        """Generate SHAP visualization plots."""
        try:
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            summary_path = Path(self.output_path) / f"{model_name}_shap_summary.png"
            plt.savefig(summary_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Log to MLflow
            mlflow.log_artifact(str(summary_path))

            # Feature importance plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
            importance_path = Path(self.output_path) / f"{model_name}_shap_importance.png"
            plt.savefig(importance_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Log to MLflow
            mlflow.log_artifact(str(importance_path))

            # Waterfall plot for a sample prediction
            if len(X) > 0:
                plt.figure(figsize=(12, 8))
                shap.waterfall_plot(explainer.expected_value, shap_values[0], X.iloc[0], show=False)
                waterfall_path = Path(self.output_path) / f"{model_name}_shap_waterfall.png"
                plt.savefig(waterfall_path, bbox_inches='tight', dpi=300)
                plt.close()

                # Log to MLflow
                mlflow.log_artifact(str(waterfall_path))

        except Exception as e:
            logger.warning(f"Failed to generate SHAP plots: {e}")

    def _generate_model_card(self, model_name: str, task_type: str) -> None:
        """Generate a comprehensive model card."""
        try:
            results = self.evaluation_results[model_name]
            metrics = results["metrics"]

            # Create model card content
            model_card = f"""# Model Card for {model_name}

## Model Overview
- **Task Type**: {task_type}
- **Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model Version**: 1.0.0

## Performance Metrics

### {task_type.title()} Metrics
"""

            # Add metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    model_card += f"- **{metric_name}**: {metric_value:.4f}\n"
                else:
                    model_card += f"- **{metric_name}**: {metric_value}\n"

            # Add threshold optimization results if available
            if "optimal_threshold" in metrics:
                model_card += f"""
### Threshold Optimization
- **Optimal Threshold**: {metrics['optimal_threshold']:.3f}
- **F1 Score at Optimal Threshold**: {metrics['optimal_threshold_f1']:.4f}
- **Precision at Optimal Threshold**: {metrics['optimal_precision']:.4f}
- **Recall at Optimal Threshold**: {metrics['optimal_recall']:.4f}
"""

            # Add explainability information
            if model_name in self.explainability_results:
                model_card += """
## Explainability
- **SHAP Analysis**: Enabled
- **Feature Importance**: Available
- **Sample Explanations**: Generated
"""

            # Add recommendations
            model_card += """
## Recommendations
- Monitor model performance on new data
- Retrain model if performance degrades significantly
- Consider feature engineering improvements
- Validate predictions on edge cases

## Limitations
- Model performance may vary on data with different distributions
- Feature importance may change over time
- Consider retraining for concept drift

## Contact
For questions about this model, contact the MLOps team.
"""

            # Save model card
            card_path = Path(self.output_path) / f"{model_name}_model_card.md"
            with open(card_path, 'w') as f:
                f.write(model_card)

            # Log to MLflow
            mlflow.log_artifact(str(card_path))

            logger.info(f"Generated model card for {model_name}")

        except Exception as e:
            logger.warning(f"Failed to generate model card: {e}")

    def _log_evaluation_results(self, model_name: str) -> None:
        """Log evaluation results to MLflow."""
        try:
            results = self.evaluation_results[model_name]
            metrics = results["metrics"]

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log evaluation parameters
            mlflow.log_params({
                "evaluation_task_type": results["task_type"],
                "evaluation_timestamp": pd.Timestamp.now().isoformat(),
                "evaluation_config": str(self.evaluation_config)
            })

            # Log evaluation results as artifact
            results_file = f"{model_name}_evaluation_results.json"
            save_json(results, results_file)
            mlflow.log_artifact(results_file)

            # Log explainability results if available
            if model_name in self.explainability_results:
                explainability_file = f"{model_name}_explainability_results.json"
                save_json(self.explainability_results[model_name], explainability_file)
                mlflow.log_artifact(explainability_file)

            logger.info(f"Successfully logged evaluation results for {model_name} to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log evaluation results to MLflow: {e}")

    def get_evaluation_summary(self) -> dict[str, Any]:
        """Get a summary of all evaluation results."""
        if not self.evaluation_results:
            return {"status": "no_evaluation_performed"}

        summary = {
            "total_models": len(self.evaluation_results),
            "models": {}
        }

        for model_name, results in self.evaluation_results.items():
            metrics = results["metrics"]
            task_type = results["task_type"]

            # Get key metrics based on task type
            if "classification" in task_type:
                key_metrics = {
                    "accuracy": metrics.get("accuracy", 0),
                    "f1_score": metrics.get("f1_macro", metrics.get("f1_binary", 0)),
                    "roc_auc": metrics.get("roc_auc", metrics.get("roc_auc_ovr", 0))
                }
            else:
                key_metrics = {
                    "r2": metrics.get("r2", 0),
                    "rmse": metrics.get("rmse", 0),
                    "mae": metrics.get("mae", 0)
                }

            summary["models"][model_name] = {
                "task_type": task_type,
                "key_metrics": key_metrics,
                "has_explainability": model_name in self.explainability_results
            }

        return summary

    def save_evaluation_results(self, path: str = None) -> None:
        """Save evaluation results to disk."""
        if path is None:
            path = self.output_path

        try:
            # Save evaluation results
            results_path = Path(path) / "evaluation_results.yaml"
            with open(results_path, 'w') as f:
                yaml.dump(self.evaluation_results, f, default_flow_style=False)

            # Save explainability results
            if self.explainability_results:
                explainability_path = Path(path) / "explainability_results.yaml"
                with open(explainability_path, 'w') as f:
                    yaml.dump(self.explainability_results, f, default_flow_style=False)

            logger.info(f"Saved evaluation results to {path}")

        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

    def load_evaluation_results(self, path: str = None) -> None:
        """Load evaluation results from disk."""
        if path is None:
            path = self.output_path

        try:
            # Load evaluation results
            results_path = Path(path) / "evaluation_results.yaml"
            if results_path.exists():
                with open(results_path) as f:
                    self.evaluation_results = yaml.safe_load(f)

            # Load explainability results
            explainability_path = Path(path) / "explainability_results.yaml"
            if explainability_path.exists():
                with open(explainability_path) as f:
                    self.explainability_results = yaml.safe_load(f)

            logger.info(f"Loaded evaluation results from {path}")

        except Exception as e:
            logger.error(f"Failed to load evaluation results: {e}")


# Helper function for average precision score
def average_precision_score(y_true, y_score):
    """Calculate average precision score."""
    try:
        from sklearn.metrics import average_precision_score as _aps
        return _aps(y_true, y_score)
    except ImportError:
        # Fallback calculation
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return np.trapz(precision, recall)
