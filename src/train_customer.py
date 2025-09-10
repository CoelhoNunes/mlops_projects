#!/usr/bin/env python3
"""
Customer Classification Training Script for MLOps project.

This script trains a model to predict customer_state (Brazilian states) based on:
- customer_zip_code_prefix
- customer_city
- customer_unique_id (encoded)

Features:
1. Load customer dataset (99,441 samples)
2. Feature engineering for categorical variables
3. Train multiple models (Random Forest, XGBoost, Logistic Regression)
4. Hyperparameter tuning with Optuna
5. Complete MLflow logging
6. Model comparison and selection
"""

import argparse
import os
import sys
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import optuna
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def load_customer_data(data_path: str, smoke: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare customer dataset."""
    print("📊 Loading customer dataset...")
    
    # Load data
    df = pd.read_csv(data_path)
    
    if smoke:
        # Use only a small subset for smoke testing
        print("🔥 Smoke test mode: using small dataset subset")
        # Ensure we have enough samples per class for stratified splitting
        df = df.sample(n=2000, random_state=42).reset_index(drop=True)
    
    print(f"✅ Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🎯 Target classes: {df['customer_state'].nunique()}")
    print(f"📊 Target distribution:\n{df['customer_state'].value_counts().head(10)}")
    
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Engineer features for customer classification."""
    print("🔧 Engineering features...")
    
    # Create a copy for feature engineering
    df_engineered = df.copy()
    
    # 1. Extract numeric features from zip code
    df_engineered['zip_code_numeric'] = pd.to_numeric(df_engineered['customer_zip_code_prefix'], errors='coerce')
    
    # 2. Create region features based on zip code ranges
    df_engineered['region'] = df_engineered['zip_code_numeric'].apply(lambda x: 
        'south' if x >= 80000 else
        'southeast' if x >= 20000 else
        'northeast' if x >= 40000 else
        'north' if x >= 60000 else
        'midwest' if x >= 70000 else 'unknown'
    )
    
    # 3. City length and word count
    df_engineered['city_length'] = df_engineered['customer_city'].str.len()
    df_engineered['city_word_count'] = df_engineered['customer_city'].str.count(' ') + 1
    
    # 4. Customer ID features (hash-based)
    df_engineered['customer_id_hash'] = df_engineered['customer_unique_id'].apply(hash) % 1000
    
    # 5. Encode categorical variables
    le_region = LabelEncoder()
    le_city = LabelEncoder()
    
    df_engineered['region_encoded'] = le_region.fit_transform(df_engineered['region'])
    df_engineered['city_encoded'] = le_city.fit_transform(df_engineered['customer_city'])
    
    # 6. Create interaction features
    df_engineered['zip_region_interaction'] = df_engineered['zip_code_numeric'] * df_engineered['region_encoded']
    
    # Select features for training
    feature_columns = [
        'zip_code_numeric', 'region_encoded', 'city_encoded', 
        'city_length', 'city_word_count', 'customer_id_hash',
        'zip_region_interaction'
    ]
    
    X = df_engineered[feature_columns].fillna(0)
    y = df_engineered['customer_state']
    
    # Filter out classes with too few samples for stratified splitting
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 3].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    
    print(f"✅ Filtered dataset: {X.shape[0]} samples, {len(valid_classes)} classes")
    
    # Store encoders for later use
    encoders = {
        'region_encoder': le_region,
        'city_encoder': le_city
    }
    
    print(f"✅ Feature engineering complete. Features: {list(X.columns)}")
    print(f"📊 Feature matrix shape: {X.shape}")
    
    return X, y, encoders


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
    """Split data into train/validation/test sets with stratification."""
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


def create_model_pipeline(model_type: str = 'rf') -> Pipeline:
    """Create a scikit-learn pipeline with preprocessing and model."""
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_type == 'lr':
        model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    return pipeline


def create_lr_pipeline_with_params(penalty: str) -> Pipeline:
    """Create a LogisticRegression pipeline with compatible solver for the given penalty."""
    if penalty == 'l1':
        # L1 penalty requires liblinear or saga solver
        model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1, solver='liblinear')
    else:
        # L2 penalty works with lbfgs (default) or other solvers
        model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    return pipeline


def objective(trial, X_train, y_train, X_val, y_val, model_type='rf'):
    """Optuna objective function for hyperparameter optimization."""
    if model_type == 'rf':
        params = {
            'classifier__n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'classifier__max_depth': trial.suggest_int('max_depth', 5, 20),
            'classifier__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'classifier__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }
        pipeline = create_model_pipeline('rf')
        pipeline.set_params(**params)
    elif model_type == 'lr':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        params = {
            'classifier__C': trial.suggest_float('C', 1e-4, 1e2, log=True),
        }
        
        # Create pipeline with compatible solver for the penalty
        pipeline = create_lr_pipeline_with_params(penalty)
        pipeline.set_params(**params)
    else:
        pipeline = create_model_pipeline(model_type)
        pipeline.set_params(**params)
    
    # Use cross-validation for more robust evaluation
    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()


def train_models(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> Dict[str, Any]:
    """Train multiple models and select the best one."""
    print("🚀 Training models...")
    
    models = {}
    best_score = 0
    best_model = None
    best_model_name = None
    
    # Train Random Forest
    print("🌲 Training Random Forest...")
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, 'rf'), n_trials=20)
    
    best_rf_params = study_rf.best_params
    rf_pipeline = create_model_pipeline('rf')
    # Set parameters on the classifier step of the pipeline
    for param, value in best_rf_params.items():
        param_name = param.replace('classifier__', '')
        rf_pipeline.named_steps['classifier'].set_params(**{param_name: value})
    rf_pipeline.fit(X_train, y_train)
    
    rf_score = rf_pipeline.score(X_val, y_val)
    models['random_forest'] = {
        'pipeline': rf_pipeline,
        'params': best_rf_params,
        'val_score': rf_score,
        'study': study_rf
    }
    
    if rf_score > best_score:
        best_score = rf_score
        best_model = rf_pipeline
        best_model_name = 'random_forest'
    
    # Train Logistic Regression
    print("📊 Training Logistic Regression...")
    study_lr = optuna.create_study(direction='maximize')
    study_lr.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, 'lr'), n_trials=20)
    
    best_lr_params = study_lr.best_params
    # Extract penalty from best params to create compatible pipeline
    penalty = best_lr_params.get('classifier__penalty', 'l2')
    lr_pipeline = create_lr_pipeline_with_params(penalty)
    # Set parameters on the classifier step of the pipeline
    for param, value in best_lr_params.items():
        param_name = param.replace('classifier__', '')
        lr_pipeline.named_steps['classifier'].set_params(**{param_name: value})
    lr_pipeline.fit(X_train, y_train)
    
    lr_score = lr_pipeline.score(X_val, y_val)
    models['logistic_regression'] = {
        'pipeline': lr_pipeline,
        'params': best_lr_params,
        'val_score': lr_score,
        'study': study_lr
    }
    
    if lr_score > best_score:
        best_score = lr_score
        best_model = lr_pipeline
        best_model_name = 'logistic_regression'
    
    print("✅ Model training complete!")
    print(f"🏆 Best model: {best_model_name} (Validation accuracy: {best_score:.4f})")
    print(f"🌲 Random Forest: {rf_score:.4f}")
    print(f"📊 Logistic Regression: {lr_score:.4f}")
    
    return {
        'models': models,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_score': best_score
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluate model on test set and generate comprehensive metrics."""
    print("📊 Evaluating model...")
    
    # Make predictions
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    classification_rep = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    print(f"✅ Model evaluation complete. Test accuracy: {test_accuracy:.4f}")
    
    evaluation_results = {
        "test_accuracy": test_accuracy,
        "classification_report": classification_rep,
        "confusion_matrix": cm,
        "predictions": y_test_pred,
        "true_labels": y_test
    }
    
    return evaluation_results


def create_confusion_matrix_plot(cm: np.ndarray, save_path: str):
    """Create and save confusion matrix visualization."""
    print("📈 Creating confusion matrix plot...")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=True, yticklabels=True)
    plt.title('Customer State Classification - Confusion Matrix')
    plt.xlabel('Predicted State')
    plt.ylabel('True State')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix plot saved to: {save_path}")


def ensure_model_compatibility(model: Pipeline) -> Pipeline:
    """Ensure the model has all required methods for MLflow logging."""
    # Check if the model has predict_proba method
    if not hasattr(model, 'predict_proba'):
        # Add predict_proba method if missing
        def predict_proba(X):
            # For models without predict_proba, return one-hot encoded predictions
            predictions = model.predict(X)
            unique_classes = np.unique(predictions)
            proba = np.zeros((len(predictions), len(unique_classes)))
            for i, pred in enumerate(predictions):
                pred_idx = np.where(unique_classes == pred)[0][0]
                proba[i, pred_idx] = 1.0
            return proba
        
        model.predict_proba = predict_proba
    
    return model


def log_to_mlflow(model: Pipeline, encoders: Dict, training_results: Dict, 
                  evaluation_results: Dict, confusion_matrix_path: str, 
                  X_test: pd.DataFrame, y_test: pd.Series, X_train: pd.DataFrame, 
                  X_val: pd.DataFrame, smoke: bool = False) -> Dict[str, Any]:
    """Log all results to MLflow."""
    print("📝 Logging to MLflow...")
    
    # Ensure model compatibility for MLflow logging
    model = ensure_model_compatibility(model)
    
    # Set experiment name
    experiment_name = "customer-classification-smoke" if smoke else "customer-classification"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model_type": training_results['best_model_name'],
            "smoke_test": smoke,
            "n_features": model.named_steps['classifier'].n_features_in_ if hasattr(model.named_steps['classifier'], 'n_features_in_') else 'unknown',
            "random_state": 42
        })
        
        # Log model-specific parameters
        if training_results['best_model_name'] == 'random_forest':
            mlflow.log_params(training_results['models']['random_forest']['params'])
        elif training_results['best_model_name'] == 'logistic_regression':
            mlflow.log_params(training_results['models']['logistic_regression']['params'])
        
        # Log metrics
        mlflow.log_metrics({
            "val_accuracy": training_results['best_score'],
            "test_accuracy": evaluation_results["test_accuracy"]
        })
        
        # Log detailed classification metrics
        for metric, value in evaluation_results["classification_report"].items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        mlflow.log_metric(f"{metric}_{sub_metric}", sub_value)
        
        # Log additional model metadata
        mlflow.log_param("n_classes", len(np.unique(y_test)))
        mlflow.log_param("feature_names", list(X_test.columns) if hasattr(X_test, 'columns') else "unknown")
        mlflow.log_param("model_pipeline_steps", list(model.named_steps.keys()))
        
        # Create input example for model signature
        input_example = X_test.iloc[:1] if hasattr(X_test, 'iloc') else X_test[:1]
        
        # Create model signature for better MLflow integration
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, TensorSpec
        
        # Define input schema
        input_schema = Schema([
            TensorSpec(np.dtype(np.float64), (-1, X_test.shape[1]), name="features")
        ])
        
        # Define output schema
        output_schema = Schema([
            TensorSpec(np.dtype(np.int64), (-1,), name="predictions")
        ])
        
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Log model with proper signature and input example
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=input_example,
            signature=signature,
            registered_model_name=f"customer-classification-{training_results['best_model_name']}"
        )
        
        # Log encoders with proper names
        for name, encoder in encoders.items():
            mlflow.sklearn.log_model(
                encoder, 
                name,
                registered_model_name=f"customer-classification-{name}"
            )
        
        # Log feature engineering information
        feature_info = {
            "feature_names": list(X_test.columns) if hasattr(X_test, 'columns') else "unknown",
            "n_features": X_test.shape[1],
            "feature_types": {
                "zip_code_numeric": "numeric",
                "region_encoded": "categorical_encoded", 
                "city_encoded": "categorical_encoded",
                "city_length": "numeric",
                "city_word_count": "numeric",
                "customer_id_hash": "numeric",
                "zip_region_interaction": "interaction"
            }
        }
        mlflow.log_dict(feature_info, "feature_engineering.json")
        
        # Create and log model card for documentation
        model_card = f"""
# Customer State Classification Model

## Model Overview
This model predicts the Brazilian state of customers based on their location and demographic features.

## Model Type
{training_results['best_model_name'].replace('_', ' ').title()}

## Performance
- Validation Accuracy: {training_results['best_score']:.4f}
- Test Accuracy: {evaluation_results['test_accuracy']:.4f}

## Features
- zip_code_numeric: Numeric zip code prefix
- region_encoded: Encoded geographic region
- city_encoded: Encoded city name
- city_length: Length of city name
- city_word_count: Number of words in city name
- customer_id_hash: Hashed customer ID
- zip_region_interaction: Interaction between zip code and region

## Training Data
- Total samples: {X_test.shape[0] + X_train.shape[0] + X_val.shape[0]}
- Number of classes: {len(np.unique(y_test))}
- Feature count: {X_test.shape[1]}

## Model Pipeline
{list(model.named_steps.keys())}

## Usage
This model expects preprocessed features in the same format as the training data.
        """
        mlflow.log_text(model_card, "model_card.md")
        
        # Log confusion matrix plot
        mlflow.log_artifact(confusion_matrix_path, "confusion_matrix.png")
        
        # Log conda environment for reproducibility
        import subprocess
        try:
            conda_env = subprocess.check_output(['conda', 'env', 'export'], 
                                              stderr=subprocess.DEVNULL, 
                                              universal_newlines=True)
            mlflow.log_text(conda_env, "conda_environment.yml")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to pip requirements if conda not available
            try:
                pip_freeze = subprocess.check_output(['pip', 'freeze'], 
                                                   stderr=subprocess.DEVNULL, 
                                                   universal_newlines=True)
                mlflow.log_text(pip_freeze, "requirements.txt")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠️  Could not capture environment dependencies")
        
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
    parser = argparse.ArgumentParser(description="Customer Classification Training Pipeline")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with small dataset")
    parser.add_argument("--data-path", type=str, default="./customer_dataset.csv", 
                       help="Path to customer dataset CSV")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="./mlruns", 
                       help="MLflow tracking URI (default: ./mlruns)")
    args = parser.parse_args()
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        print(f"🔗 MLflow tracking URI: {args.mlflow_tracking_uri}")
        
        # Load data
        df = load_customer_data(args.data_path, smoke=args.smoke)
        
        # Engineer features
        X, y, encoders = engineer_features(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Train models
        training_results = train_models(X_train, X_val, y_train, y_val)
        
        # Evaluate best model
        evaluation_results = evaluate_model(training_results['best_model'], X_test, y_test)
        
        # Create confusion matrix plot
        confusion_matrix_path = "customer_confusion_matrix.png"
        create_confusion_matrix_plot(
            np.array(evaluation_results["confusion_matrix"]), 
            confusion_matrix_path
        )
        
        # Log everything to MLflow
        log_to_mlflow(
            training_results['best_model'], encoders, training_results, evaluation_results, 
            confusion_matrix_path, X_test, y_test, X_train, X_val, args.smoke
        )
        
        # Clean up temporary files
        for temp_file in [confusion_matrix_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print("🎉 Customer classification training pipeline completed successfully!")
        print(f"📊 Final test accuracy: {evaluation_results['test_accuracy']:.4f}")
        print(f"🏆 Best model: {training_results['best_model_name']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
