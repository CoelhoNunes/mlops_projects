from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline

# Import steps directly and call them inside the pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline():
    """
    End-to-end wiring with local defaults:
      ingest_data() -> df
      clean_data(df) -> x_train, x_test, y_train, y_test
      train_model(...) -> model
      evaluation(model, x_test, y_test) -> r2, rmse
    """
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2, rmse = evaluation(model, x_test, y_test)
    return r2, rmse
