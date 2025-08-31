import logging
import contextlib

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step

from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)



@step  # let the active ZenML stack provide an experiment tracker if configured
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str = "lightgbm",
    fine_tuning: bool = False,
) -> RegressorMixin:
    """
    Train a machine learning model for customer satisfaction prediction.
    
    Args:
        x_train: Training features
        x_test: Test features  
        y_train: Training targets
        y_test: Test targets
        model_name: Model to train (lightgbm, randomforest, xgboost, linear_regression)
        fine_tuning: Whether to perform hyperparameter optimization
        
    Returns:
        Trained model ready for prediction
    """
    try:
        # Select model + autolog flavor
        if model_name == "lightgbm":
            autolog = mlflow.lightgbm.autolog
            model = LightGBMModel()
        elif model_name == "randomforest":
            autolog = mlflow.sklearn.autolog
            model = RandomForestModel()
        elif model_name == "xgboost":
            autolog = mlflow.xgboost.autolog
            model = XGBoostModel()
        elif model_name == "linear_regression":
            autolog = mlflow.sklearn.autolog
            model = LinearRegressionModel()
        else:
            raise ValueError(f"Model name not supported: {model_name}")

        # Ensure there's an active MLflow run (works with or without a ZenML tracker)
        run_ctx = mlflow.start_run() if mlflow.active_run() is None else contextlib.nullcontext()
        with run_ctx:
            autolog()  # enable flavor-specific autologging

            tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

            if fine_tuning:
                best_params = tuner.optimize()
                trained_model = model.train(x_train, y_train, **best_params)
            else:
                trained_model = model.train(x_train, y_train)

        return trained_model

    except Exception as e:
        logging.exception("Training failed: %s", e)
        raise
