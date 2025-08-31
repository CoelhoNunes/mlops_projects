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
from .config import ModelNameConfig


@step  # let the active ZenML stack provide an experiment tracker if configured
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Train a model chosen via `config.model_name`. If `config.fine_tuning` is True,
    run hyperparameter tuning first. MLflow autologging is enabled inside a safe run.
    """
    try:
        # Select model + autolog flavor
        if config.model_name == "lightgbm":
            autolog = mlflow.lightgbm.autolog
            model = LightGBMModel()
        elif config.model_name == "randomforest":
            autolog = mlflow.sklearn.autolog
            model = RandomForestModel()
        elif config.model_name == "xgboost":
            autolog = mlflow.xgboost.autolog
            model = XGBoostModel()
        elif config.model_name == "linear_regression":
            autolog = mlflow.sklearn.autolog
            model = LinearRegressionModel()
        else:
            raise ValueError(f"Model name not supported: {config.model_name}")

        # Ensure there's an active MLflow run (works with or without a ZenML tracker)
        run_ctx = mlflow.start_run() if mlflow.active_run() is None else contextlib.nullcontext()
        with run_ctx:
            autolog()  # enable flavor-specific autologging

            tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

            if config.fine_tuning:
                best_params = tuner.optimize()
                trained_model = model.train(x_train, y_train, **best_params)
            else:
                trained_model = model.train(x_train, y_train)

        return trained_model

    except Exception as e:
        logging.exception("Training failed: %s", e)
        raise
