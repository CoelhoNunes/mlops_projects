from pydantic import BaseModel
from typing import Literal

class ModelNameConfig(BaseModel):
    """Model configurations for the training step."""
    model_name: Literal["lightgbm", "randomforest", "xgboost", "linear_regression"] = "lightgbm"
    fine_tuning: bool = False
