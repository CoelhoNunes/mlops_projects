#!/usr/bin/env python3
"""
Simplified MLflow-only training script for MLOps project.

This script demonstrates a complete ML training pipeline:
1. Load a small dataset (sklearn digits)
2. Split data into train/validation/test
3. Train a simple model
4. Evaluate performance
5. Log everything to MLflow
6. Save model and artifacts
"""

import argparse
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(smoke: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for training."""
    print("📊 Loading dataset...")
    
    # Load sklearn digits dataset
    digits = load_digits()
    X = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(digits.data.shape[1])])
    y = pd.Series(digits.target, name="digit")
    
    if smoke:
        # Use only a small subset for smoke testing
        print("🔥 Smoke test mode: using small dataset subset")
        indices = np.random.choice(len(X), size=100, replace=False)
        X = X.iloc[indices].reset_index(drop=True)
        y = y.iloc[indices].reset_index(drop=True)
    
    print(f"✅ Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
    """Split data into train/validation/test sets."""
    print("✂️  Splitting data...")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print("✅ Data split complete:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples") 
    print(f"   Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
    """Preprocess data using StandardScaler."""
    print("🔧 Preprocessing data...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Data preprocessing complete")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_model(X_train: np.ndarray, y_train: pd.Series, X_val: np.ndarray, y_val: pd.Series) -> Tuple[RandomForestClassifier, dict]:
    """Train a Random Forest model and evaluate on validation set."""
    print("🚀 Training model...")
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"✅ Model training complete. Validation accuracy: {val_accuracy:.4f}")
    
    training_results = {
        "val_accuracy": val_accuracy,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "feature_importance": dict(zip(range(X_train.shape[1]), model.feature_importances_))
    }
    
    return model, training_results


def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: pd.Series) -> dict:
    """Evaluate model on test set and generate metrics."""
    print("📊 Evaluating model...")
    
    # Make predictions
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    classification_rep = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    evaluation_results = {
        "test_accuracy": test_accuracy,
        "classification_report": classification_rep,
        "confusion_matrix": cm.tolist()
    }
    
    print(f"✅ Model evaluation complete. Test accuracy: {test_accuracy:.4f}")
    return evaluation_results


def create_confusion_matrix_plot(cm: np.ndarray, save_path: str) -> str:
    """Create and save confusion matrix visualization."""
    print("📈 Creating confusion matrix plot...")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix plot saved to: {save_path}")
    return save_path


def log_to_mlflow(model, scaler, training_results: dict, evaluation_results: dict, 
                  confusion_matrix_path: str, smoke: bool = False):
    """Log all training artifacts to MLflow."""
    print("📝 Logging to MLflow...")
    
    # Set experiment name
    experiment_name = "mlops-training"
    if smoke:
        experiment_name += "-smoke"
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"training-run-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": training_results["n_estimators"],
            "max_depth": training_results["max_depth"],
            "random_state": 42,
            "smoke_test": smoke
        })
        
        # Log metrics
        mlflow.log_metrics({
            "val_accuracy": training_results["val_accuracy"],
            "test_accuracy": evaluation_results["test_accuracy"]
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log scaler
        mlflow.sklearn.log_model(scaler, "scaler")
        
        # Log confusion matrix plot
        mlflow.log_artifact(confusion_matrix_path, "confusion_matrix.png")
        
        # Log feature importance
        feature_importance_df = pd.DataFrame([
            {"feature": f"pixel_{i}", "importance": importance}
            for i, importance in training_results["feature_importance"].items()
        ]).sort_values("importance", ascending=False)
        
        feature_importance_path = "feature_importance.csv"
        feature_importance_df.to_csv(feature_importance_path, index=False)
        mlflow.log_artifact(feature_importance_path, "feature_importance.csv")
        
        # Log run info
        run_info = {
            "run_id": mlflow.active_run().info.run_id,
            "experiment_id": mlflow.active_run().info.experiment_id,
            "status": "FINISHED"
        }
        
        print("✅ MLflow logging complete:")
        print(f"   Run ID: {run_info['run_id']}")
        print(f"   Experiment: {experiment_name}")
        
        return run_info


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="MLOps Training Pipeline")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with small dataset")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="./mlruns", 
                       help="MLflow tracking URI (default: ./mlruns)")
    args = parser.parse_args()
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        print(f"🔗 MLflow tracking URI: {args.mlflow_tracking_uri}")
        
        # Load data
        X, y = load_data(smoke=args.smoke)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Preprocess data
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = preprocess_data(
            X_train, X_val, X_test
        )
        
        # Train model
        model, training_results = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Evaluate model
        evaluation_results = evaluate_model(model, X_test_scaled, y_test)
        
        # Create confusion matrix plot
        confusion_matrix_path = "confusion_matrix.png"
        create_confusion_matrix_plot(
            np.array(evaluation_results["confusion_matrix"]), 
            confusion_matrix_path
        )
        
        # Log everything to MLflow
        log_to_mlflow(
            model, scaler, training_results, evaluation_results, 
            confusion_matrix_path, args.smoke
        )
        
        # Clean up temporary files
        for temp_file in [confusion_matrix_path, "feature_importance.csv"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print("🎉 Training pipeline completed successfully!")
        print(f"📊 Final test accuracy: {evaluation_results['test_accuracy']:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
