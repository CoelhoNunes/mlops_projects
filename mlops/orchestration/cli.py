"""Command-line interface for the MLOps project."""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..data import DataLoader, DataValidator, FeatureEngineer
from ..modeling import ModelEvaluator, ModelPromoter, ModelTrainer
from ..serving import ModelServer
from ..utils import Config, get_logger, set_seed, setup_logging
from ..utils.config import Config

# Create Typer app
app = typer.Typer(
    name="mlops",
    help="Production-ready MLOps project with MLflow tracking and PyTorch support",
    add_completion=False,
)

# Rich console for better output
console = Console()


@app.command()
def data_validate(
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Validate data quality and generate reports."""
    console.print("🔍 [bold blue]Data Validation[/bold blue]")

    # Load configuration
    config = Config.from_env(config_path)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(
        config_path="mlops/config/logging.yaml",
        log_level=log_level,
    )

    logger = get_logger(__name__)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data...", total=None)

        try:
            # Load data
            data_loader = DataLoader(config)
            df, target_column, task_type = data_loader.load_data()

            progress.update(task, description="Validating data quality...")

            # Validate data
            validator = DataValidator(config)
            validation_results = validator.validate_dataset(df, target_column, task_type)

            progress.update(task, description="Generating reports...")

            # Generate and save reports
            validator.generate_reports(validation_results)

            progress.update(task, description="Complete!", total=1)
            progress.advance(task)

            console.print("✅ [bold green]Data validation completed successfully![/bold green]")

            # Display summary
            table = Table(title="Data Validation Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            # Extract basic info from validation results
            basic_info = validation_results.get("validation_results", {}).get("classification_basic_info", {})
            target_info = validation_results.get("validation_results", {}).get("classification_target", {})
            
            table.add_row("Total Rows", str(basic_info.get("shape", [0, 0])[0]))
            table.add_row("Total Columns", str(basic_info.get("shape", [0, 0])[1]))
            table.add_row("Memory Usage (MB)", f"{basic_info.get('memory_usage_mb', 0):.2f}")
            table.add_row("Duplicate Rows", str(basic_info.get("duplicate_count", 0)))
            table.add_row("Task Type", target_info.get("task_type", "Unknown"))
            table.add_row("Target Column", "customer_state")

            console.print(table)

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            console.print(f"❌ [bold red]Data validation failed: {e}[/bold red]")
            raise typer.Exit(1)


@app.command()
def train(
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    model_name: str | None = typer.Option(
        None, "--model", "-m", help="Specific model to train (default: all)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Train models with hyperparameter optimization."""
    console.print("🚀 [bold blue]Model Training[/bold blue]")

    # Load configuration
    config = Config.from_env(config_path)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(
        config_path="mlops/config/logging.yaml",
        log_level=log_level,
    )

    logger = get_logger(__name__)

    # Set random seed
    set_seed(config.system.random_seed)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading and preprocessing data...", total=None)

        try:
            # Load and preprocess data
            data_loader = DataLoader(config)
            df, target_column, task_type = data_loader.load_data()

            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(
                df, target_column, task_type
            )

            # Save splits
            data_loader.save_splits(
                X_train, X_val, X_test, y_train, y_val, y_test, target_column
            )

            # Log to MLflow
            data_loader.log_to_mlflow("training_run")

            progress.update(task, description="Engineering features...")

            # Feature engineering
            feature_engineer = FeatureEngineer(config)
            X_train_processed, X_val_processed, X_test_processed, transformer = (
                feature_engineer.fit_transform(X_train, X_val, X_test)
            )

            progress.update(task, description="Training models...")

            # Train models
            trainer = ModelTrainer(config)

            if model_name:
                # Train specific model
                models_to_train = [model_name]
            else:
                # Train all models in roster
                models_to_train = config.models.roster

            training_results = {}

            for model_name in models_to_train:
                console.print(f"🎯 Training {model_name}...")

                model_result = trainer.train_model(
                    model_name,
                    X_train_processed,
                    y_train,
                    X_val_processed,
                    y_val,
                    task_type,
                )

                training_results[model_name] = model_result

                console.print(f"✅ {model_name} training completed!")

            progress.update(task, description="Complete!", total=1)
            progress.advance(task)

            console.print("✅ [bold green]Model training completed successfully![/bold green]")

            # Display results summary
            table = Table(title="Training Results Summary")
            table.add_column("Model", style="cyan")
            table.add_column("Best Score", style="magenta")
            table.add_column("Training Time", style="green")

            for model_name, result in training_results.items():
                best_score = result.get("best_score", "N/A")
                training_time = result.get("training_time", "N/A")

                if isinstance(training_time, float):
                    training_time = f"{training_time:.2f}s"

                table.add_row(model_name, str(best_score), str(training_time))

            console.print(table)

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            console.print(f"❌ [bold red]Model training failed: {e}[/bold red]")
            raise typer.Exit(1)


@app.command()
def evaluate(
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    model_name: str | None = typer.Option(
        None, "--model", "-m", help="Specific model to evaluate (default: best model)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Evaluate trained models on test set."""
    console.print("📊 [bold blue]Model Evaluation[/bold blue]")

    # Load configuration
    config = Config.from_env(config_path)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(
        config_path="mlops/config/logging.yaml",
        log_level=log_level,
    )

    logger = get_logger(__name__)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data and models...", total=None)

        try:
            # Load test data
            data_loader = DataLoader(config)
            df, target_column, task_type = data_loader.load_data()

            # Load saved splits
            test_data_path = Path(config.paths.data_artifacts)
            X_test = data_loader.load_csv(test_data_path / "X_test.csv")
            y_test = data_loader.load_csv(test_data_path / "y_test.csv")[target_column]

            # Load feature transformer
            feature_engineer = FeatureEngineer(config)
            transformer = feature_engineer.load_transformer()
            X_test_processed = feature_engineer.transform(X_test)

            progress.update(task, description="Evaluating models...")

            # Evaluate models
            evaluator = ModelEvaluator(config)

            if model_name:
                # Evaluate specific model
                evaluation_result = evaluator.evaluate_model(
                    model_name, X_test_processed, y_test, task_type
                )
                evaluation_results = {model_name: evaluation_result}
            else:
                # Evaluate all trained models
                evaluation_results = evaluator.evaluate_all_models(
                    X_test_processed, y_test, task_type
                )

            progress.update(task, description="Generating evaluation reports...")

            # Generate reports
            evaluator.generate_evaluation_reports(evaluation_results)

            progress.update(task, description="Complete!", total=1)
            progress.advance(task)

            console.print("✅ [bold green]Model evaluation completed successfully![/bold green]")

            # Display evaluation summary
            table = Table(title="Evaluation Results Summary")
            table.add_column("Model", style="cyan")
            table.add_column("Test Score", style="magenta")
            table.add_column("Best Metric", style="green")

            for model_name, result in evaluation_results.items():
                test_score = result.get("test_score", "N/A")
                best_metric = result.get("best_metric", "N/A")

                table.add_row(model_name, str(test_score), str(best_metric))

            console.print(table)

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            console.print(f"❌ [bold red]Model evaluation failed: {e}[/bold red]")
            raise typer.Exit(1)


@app.command()
def register(
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    model_name: str | None = typer.Option(
        None, "--model", "-m", help="Specific model to register (default: best model)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Register best model to MLflow Model Registry."""
    console.print("📝 [bold blue]Model Registration[/bold blue]")

    # Load configuration
    config = Config.from_env(config_path)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(
        config_path="mlops/config/logging.yaml",
        log_level=log_level,
    )

    logger = get_logger(__name__)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Registering model...", total=None)

        try:
            # Register model
            promoter = ModelPromoter(config)

            if model_name:
                # Register specific model
                registration_result = promoter.register_model(model_name)
            else:
                # Register best model
                registration_result = promoter.register_best_model()

            progress.update(task, description="Complete!", total=1)
            progress.advance(task)

            console.print("✅ [bold green]Model registration completed successfully![/bold green]")

            # Display registration info
            table = Table(title="Model Registration Summary")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")

            for key, value in registration_result.items():
                table.add_row(key, str(value))

            console.print(table)

        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            console.print(f"❌ [bold red]Model registration failed: {e}[/bold red]")
            raise typer.Exit(1)


@app.command()
def promote(
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    model_name: str | None = typer.Option(
        None, "--model", "-m", help="Specific model to promote (default: best model)"
    ),
    stage: str = typer.Option(
        "Production", "--stage", "-s", help="Target stage (Staging, Production)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Promote model to specified stage in MLflow Model Registry."""
    console.print("🚀 [bold blue]Model Promotion[/bold blue]")

    # Load configuration
    config = Config.from_env(config_path)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(
        config_path="mlops/config/logging.yaml",
        log_level=log_level,
    )

    logger = get_logger(__name__)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Promoting model...", total=None)

        try:
            # Promote model
            promoter = ModelPromoter(config)

            if model_name:
                # Promote specific model
                promotion_result = promoter.promote_model(model_name, stage)
            else:
                # Promote best model
                promotion_result = promoter.promote_best_model(stage)

            progress.update(task, description="Complete!", total=1)
            progress.advance(task)

            console.print("✅ [bold green]Model promotion completed successfully![/bold green]")

            # Display promotion info
            table = Table(title="Model Promotion Summary")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="magenta")

            for key, value in promotion_result.items():
                table.add_row(key, str(value))

            console.print(table)

        except Exception as e:
            logger.error(f"Model promotion failed: {e}")
            console.print(f"❌ [bold red]Model promotion failed: {e}[/bold red]")
            raise typer.Exit(1)


@app.command()
def serve(
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Start model serving server."""
    console.print("🌐 [bold blue]Model Serving[/bold blue]")

    # Load configuration
    config = Config.from_env(config_path)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(
        config_path="mlops/config/logging.yaml",
        log_level=log_level,
    )

    logger = get_logger(__name__)

    try:
        # Start server
        server = ModelServer(config)
        server.start(host=host, port=port, reload=reload)

    except Exception as e:
        logger.error(f"Model serving failed: {e}")
        console.print(f"❌ [bold red]Model serving failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def batch_predict(
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    input_file: Path = typer.Argument(..., help="Input CSV file for predictions"),
    output_file: Path = typer.Option(
        "predictions.csv", "--output", "-o", help="Output CSV file for predictions"
    ),
    model_name: str | None = typer.Option(
        None, "--model", "-m", help="Specific model to use (default: Production)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Run batch predictions on input data."""
    console.print("🔮 [bold blue]Batch Prediction[/bold blue]")

    # Load configuration
    config = Config.from_env(config_path)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(
        config_path="mlops/config/logging.yaml",
        log_level=log_level,
    )

    logger = get_logger(__name__)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model and data...", total=None)

        try:
            # Load model
            server = ModelServer(config)
            model = server.load_model(model_name)

            # Load input data
            import pandas as pd
            input_data = pd.read_csv(input_file)

            progress.update(task, description="Making predictions...")

            # Make predictions
            predictions = server.predict_batch(model, input_data)

            # Save predictions
            output_df = input_data.copy()
            output_df["prediction"] = predictions

            output_df.to_csv(output_file, index=False)

            progress.update(task, description="Complete!", total=1)
            progress.advance(task)

            console.print("✅ [bold green]Batch prediction completed successfully![/bold green]")
            console.print(f"📁 Predictions saved to: {output_file}")

            # Display prediction summary
            table = Table(title="Prediction Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Input Rows", str(len(input_data)))
            table.add_row("Output File", str(output_file))
            table.add_row("Model Used", model_name or "Production")

            console.print(table)

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            console.print(f"❌ [bold red]Batch prediction failed: {e}[/bold red]")
            raise typer.Exit(1)


@app.command()
def info(
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
):
    """Display project information and configuration."""
    console.print("ℹ️  [bold blue]Project Information[/bold blue]")

    try:
        # Load configuration
        config = Config.from_env(config_path)

        # Display configuration summary
        table = Table(title="Configuration Summary")
        table.add_column("Section", style="cyan")
        table.add_column("Key", style="magenta")
        table.add_column("Value", style="green")

        # Data configuration
        table.add_row("Data", "Paths", str(config.data.paths))
        table.add_row("Data", "Task Type", config.data.task_type)
        table.add_row("Data", "Target Column", str(config.data.target_column))

        # Models configuration
        table.add_row("Models", "Roster", str(config.models.roster))

        # MLflow configuration
        table.add_row("MLflow", "Tracking URI", config.mlflow.tracking_uri)
        table.add_row("MLflow", "Experiment Name", config.mlflow.experiment_name)

        # System configuration
        table.add_row("System", "Random Seed", str(config.system.random_seed))
        table.add_row("System", "GPU Enabled", str(config.system.gpu.enabled))

        console.print(table)

        # Display paths
        paths_table = Table(title="Project Paths")
        paths_table.add_column("Path Type", style="cyan")
        paths_table.add_column("Path", style="magenta")

        for path_attr in dir(config.paths):
            if not path_attr.startswith('_'):
                path_value = getattr(config.paths, path_attr)
                paths_table.add_row(path_attr.replace('_', ' ').title(), str(path_value))

        console.print(paths_table)

    except Exception as e:
        console.print(f"❌ [bold red]Failed to load configuration: {e}[/bold red]")
        raise typer.Exit(1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
