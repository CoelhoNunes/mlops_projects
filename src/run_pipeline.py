from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    try:
        print("Starting pipeline execution...")
        
        # Run the pipeline directly - this will execute all steps
        pipeline_run = train_pipeline()
        
        print("Pipeline completed successfully!")
        print(f"Pipeline run ID: {pipeline_run.id if hasattr(pipeline_run, 'id') else 'Unknown'}")
        
        print("\nTo inspect runs in MLflow, execute:")
        print(f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'")
        print("Then open the UI and look for your latest pipeline run.")
        
        # Try to get some basic info about the run
        if hasattr(pipeline_run, 'status'):
            print(f"Pipeline status: {pipeline_run.status}")
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        if "MLflow experiment tracker" in str(e):
            print("\nTo fix the MLflow tracker issue, run these commands:")
            print("  zenml experiment-tracker register mlflow_tracker --type=mlflow")
            print("  zenml stack register default_stack -e mlflow_tracker")
            print("  zenml stack set default_stack")
        else:
            print("Please ensure all dependencies are installed and ZenML is properly configured.")
