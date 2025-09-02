"""Metrics calculation utilities for the MLOps project."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer


def calculate_classification_metrics(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    y_prob: np.ndarray | pd.Series | list | None = None,
    average: str = "weighted",
    zero_division: str | int = 0,
) -> dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC AUC and PR AUC)
        average: Averaging method for multi-class metrics
        zero_division: Value to return when there is a zero division
        
    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}

    # Basic classification metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Handle multi-class case
    if len(np.unique(y_true)) > 2:
        # Multi-class metrics
        metrics["precision_macro"] = precision_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        )
        metrics["precision_weighted"] = precision_score(
            y_true, y_pred, average="weighted", zero_division=zero_division
        )
        metrics["recall_macro"] = recall_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        )
        metrics["recall_weighted"] = recall_score(
            y_true, y_pred, average="weighted", zero_division=zero_division
        )
        metrics["f1_macro"] = f1_score(
            y_true, y_pred, average="macro", zero_division=zero_division
        )
        metrics["f1_weighted"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=zero_division
        )

        # Use weighted as default
        metrics["precision"] = metrics["precision_weighted"]
        metrics["recall"] = metrics["recall_weighted"]
        metrics["f1"] = metrics["f1_weighted"]

    else:
        # Binary classification
        metrics["precision"] = precision_score(
            y_true, y_pred, average=average, zero_division=zero_division
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average=average, zero_division=zero_division
        )
        metrics["f1"] = f1_score(
            y_true, y_pred, average=average, zero_division=zero_division
        )

    # Probability-based metrics (if probabilities provided)
    if y_prob is not None:
        y_prob = np.array(y_prob)

        # Handle multi-class case for ROC AUC
        if len(np.unique(y_true)) > 2:
            # Multi-class ROC AUC (one-vs-rest)
            try:
                lb = LabelBinarizer()
                y_true_bin = lb.fit_transform(y_true)
                if y_prob.shape[1] == y_true_bin.shape[1]:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true_bin, y_prob, average=average, multi_class="ovr"
                    )
                else:
                    metrics["roc_auc"] = np.nan
            except ValueError:
                metrics["roc_auc"] = np.nan

            # Multi-class PR AUC (one-vs-rest)
            try:
                if y_prob.shape[1] == y_true_bin.shape[1]:
                    metrics["pr_auc"] = average_precision_score(
                        y_true_bin, y_prob, average=average
                    )
                else:
                    metrics["pr_auc"] = np.nan
            except ValueError:
                metrics["pr_auc"] = np.nan
        else:
            # Binary classification
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                # Use positive class probabilities
                y_prob_binary = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob[:, 0]
            else:
                y_prob_binary = y_prob.flatten()

            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob_binary)
            except ValueError:
                metrics["roc_auc"] = np.nan

            try:
                metrics["pr_auc"] = average_precision_score(y_true, y_prob_binary)
            except ValueError:
                metrics["pr_auc"] = np.nan

    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
) -> dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}

    # Basic regression metrics
    metrics["mse"] = mean_squared_error(y_true, y_pred)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["r2"] = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error
    try:
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        metrics["mape"] = mape
    except (ZeroDivisionError, RuntimeWarning):
        metrics["mape"] = np.nan

    # Additional metrics
    metrics["mae_std"] = np.std(np.abs(y_true - y_pred))
    metrics["residual_std"] = np.std(y_true - y_pred)

    return metrics


def calculate_metrics(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    y_prob: np.ndarray | pd.Series | list | None = None,
    task_type: str = "auto",
    **kwargs
) -> dict[str, float]:
    """
    Calculate metrics based on task type.
    
    Args:
        y_true: True values/labels
        y_pred: Predicted values/labels
        y_prob: Predicted probabilities (for classification)
        task_type: Task type ('classification', 'regression', or 'auto')
        **kwargs: Additional arguments for metric calculation
        
    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    # Auto-detect task type if not specified
    if task_type == "auto":
        # Check if target is numeric (regression) or categorical (classification)
        y_true_array = np.array(y_true)
        if np.issubdtype(y_true_array.dtype, np.number):
            task_type = "regression"
        else:
            task_type = "classification"

    if task_type == "classification":
        return calculate_classification_metrics(y_true, y_pred, y_prob, **kwargs)
    elif task_type == "regression":
        return calculate_regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def calculate_threshold_metrics(
    y_true: np.ndarray | pd.Series | list,
    y_prob: np.ndarray | pd.Series | list,
    thresholds: list[float] | None = None,
    metric: str = "f1",
    beta: float = 1.0,
) -> tuple[list[float], list[float], float, float]:
    """
    Calculate metrics for different probability thresholds.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        thresholds: List of thresholds to evaluate
        metric: Metric to optimize ('f1', 'fbeta', 'precision', 'recall')
        beta: Beta parameter for F-beta score
        
    Returns:
        Tuple: (thresholds, metric_values, best_threshold, best_metric_value)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)

    metric_values = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        if metric == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "fbeta":
            value = f1_score(y_true, y_pred, beta=beta, zero_division=0)
        elif metric == "precision":
            value = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            value = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        metric_values.append(value)

    # Find best threshold
    best_idx = np.argmax(metric_values)
    best_threshold = thresholds[best_idx]
    best_metric_value = metric_values[best_idx]

    return thresholds, metric_values, best_threshold, best_metric_value


def calculate_feature_importance(
    model: Any,
    feature_names: list[str],
    method: str = "default",
    X: np.ndarray | pd.DataFrame | None = None,
    y: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """
    Calculate feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
        method: Method to use ('default', 'permutation', 'shap')
        X: Feature matrix for permutation importance
        y: Target values for permutation importance
        
    Returns:
        Dict[str, float]: Dictionary mapping feature names to importance scores
    """
    if method == "default":
        # Try to get feature importance from model attributes
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = np.mean(importances, axis=0)
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute")

        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importances, strict=False))

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return feature_importance

    elif method == "permutation":
        if X is None or y is None:
            raise ValueError("X and y are required for permutation importance")

        from sklearn.inspection import permutation_importance

        # Calculate permutation importance
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importances = result.importances_mean

        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importances, strict=False))

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return feature_importance

    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_confusion_matrix_metrics(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    labels: list | None = None,
) -> dict[str, Any]:
    """
    Calculate confusion matrix and related metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names for confusion matrix
        
    Returns:
        Dict[str, Any]: Dictionary containing confusion matrix and metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate per-class metrics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "confusion_matrix": cm,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    # Calculate additional metrics if binary classification
    if cm.size == 4:
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (
            metrics["precision"] + metrics["recall"]
        ) if (metrics["precision"] + metrics["recall"]) > 0 else 0

    return metrics


def calculate_custom_metrics(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    y_prob: np.ndarray | pd.Series | list | None = None,
    custom_functions: dict[str, callable] | None = None,
) -> dict[str, float]:
    """
    Calculate custom metrics using user-defined functions.
    
    Args:
        y_true: True values/labels
        y_pred: Predicted values/labels
        y_prob: Predicted probabilities
        custom_functions: Dictionary mapping metric names to functions
        
    Returns:
        Dict[str, float]: Dictionary of custom metric names and values
    """
    if custom_functions is None:
        return {}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob) if y_prob is not None else None

    custom_metrics = {}

    for metric_name, metric_func in custom_functions.items():
        try:
            if y_prob is not None:
                value = metric_func(y_true, y_pred, y_prob)
            else:
                value = metric_func(y_true, y_pred)
            custom_metrics[metric_name] = value
        except Exception as e:
            print(f"Warning: Could not calculate custom metric {metric_name}: {e}")
            custom_metrics[metric_name] = np.nan

    return custom_metrics


def calculate_class_imbalance(
    y: np.ndarray | pd.Series | list,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """
    Calculate class imbalance metrics for classification tasks.
    
    Args:
        y: Target labels
        threshold: Threshold for considering imbalance (ratio of minority to majority class)
        
    Returns:
        Dict[str, Any]: Dictionary containing imbalance metrics
    """
    y = np.array(y)
    unique_classes, counts = np.unique(y, return_counts=True)
    
    if len(unique_classes) < 2:
        return {
            "is_imbalanced": False,
            "imbalance_ratio": 1.0,
            "minority_class": None,
            "majority_class": None,
            "minority_count": 0,
            "majority_count": 0,
            "severity": "none"
        }
    
    # Sort by counts (ascending)
    sorted_indices = np.argsort(counts)
    minority_class = unique_classes[sorted_indices[0]]
    majority_class = unique_classes[sorted_indices[-1]]
    minority_count = counts[sorted_indices[0]]
    majority_count = counts[sorted_indices[-1]]
    
    # Calculate imbalance ratio
    imbalance_ratio = minority_count / majority_count
    
    # Determine severity
    if imbalance_ratio >= 0.5:
        severity = "none"
    elif imbalance_ratio >= 0.2:
        severity = "mild"
    elif imbalance_ratio >= 0.1:
        severity = "moderate"
    else:
        severity = "severe"
    
    is_imbalanced = imbalance_ratio < threshold
    
    return {
        "is_imbalanced": is_imbalanced,
        "imbalance_ratio": imbalance_ratio,
        "minority_class": minority_class,
        "majority_class": majority_class,
        "minority_count": minority_count,
        "majority_count": majority_count,
        "severity": severity,
        "total_samples": len(y),
        "class_distribution": dict(zip(unique_classes, counts))
    }
