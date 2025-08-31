from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    pipe = train_pipeline()   # build the pipeline (no step instances passed in)
    run = pipe.run()          # run once

    print(
        "To inspect runs in MLflow, execute:\n"
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "Then open the UI and look for your latest pipeline run."
    )
