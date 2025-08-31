# src/steps/evaluation.py
import logging
import contextlib

import mlflow
import numpy as np
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin
from zenml import step

from model.evaluation import MSE, RMSE, R2Score


@step  # let the active ZenML stack provide an experiment tracker if configured
def evaluation(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    try:
        prediction = model.predict(x_test)

        # Compute metrics
        mse = MSE().calculate_score(y_test, prediction)
        r2_score = R2Score().calculate_score(y_test, prediction)
        rmse = RMSE().calculate_score(y_test, prediction)

        # Log to MLflow: if no active run (no tracker), open a local one so it won't crash
        run_ctx = mlflow.start_run() if mlflow.active_run() is None else contextlib.nullcontext()
        with run_ctx:
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2_score)
            mlflow.log_metric("rmse", rmse)

        return r2_score, rmse
    except Exception as e:
        logging.error(e)
        raise