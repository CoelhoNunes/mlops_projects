"""Logging utilities for the MLOps project."""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any

import yaml


def setup_logging(
    config_path: str | None = None,
    log_level: str | None = None,
    log_file: str | None = None,
) -> None:
    """
    Set up logging configuration from YAML file or with defaults.
    
    Args:
        config_path: Path to logging configuration YAML file
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Override log file path
    """
    if config_path and Path(config_path).exists():
        # Load from YAML config
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Override log level if specified
        if log_level:
            config['root']['level'] = log_level.upper()
            for logger_config in config.get('loggers', {}).values():
                if 'level' in logger_config:
                    logger_config['level'] = log_level.upper()

        # Override log file if specified
        if log_file:
            for handler in config.get('handlers', {}).values():
                if 'filename' in handler:
                    handler['filename'] = log_file

        # Ensure log directories exist
        for handler in config.get('handlers', {}).values():
            if 'filename' in handler:
                Path(handler['filename']).parent.mkdir(parents=True, exist_ok=True)

        # Apply configuration
        logging.config.dictConfig(config)

    else:
        # Default configuration
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        # Create formatter
        formatter = logging.Formatter(log_format, date_format)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Create file handler if log_file specified
        handlers = [console_handler]
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=handlers,
            force=True
        )

    # Set specific logger levels
    logging.getLogger('mlflow').setLevel(logging.INFO)
    logging.getLogger('optuna').setLevel(logging.INFO)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('pandas').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)

    print(f"Logging configured (level: {logging.getLogger().getEffectiveLevel()})")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def log_mlflow_metrics(
    logger: logging.Logger,
    metrics: dict[str, float],
    step: int | None = None,
) -> None:
    """
    Log metrics in a format suitable for MLflow.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        step: Optional step number for training metrics
    """
    for metric_name, value in metrics.items():
        if step is not None:
            logger.info(f"Step {step} - {metric_name}: {value:.6f}")
        else:
            logger.info(f"{metric_name}: {value:.6f}")


def log_mlflow_params(
    logger: logging.Logger,
    params: dict[str, Any],
) -> None:
    """
    Log parameters in a format suitable for MLflow.
    
    Args:
        logger: Logger instance
        params: Dictionary of parameter names and values
    """
    for param_name, value in params.items():
        logger.info(f"Parameter {param_name}: {value}")


def log_mlflow_artifacts(
    logger: logging.Logger,
    artifacts: dict[str, str],
) -> None:
    """
    Log artifact paths in a format suitable for MLflow.
    
    Args:
        logger: Logger instance
        artifacts: Dictionary of artifact names and file paths
    """
    for artifact_name, file_path in artifacts.items():
        if Path(file_path).exists():
            logger.info(f"Artifact {artifact_name}: {file_path}")
        else:
            logger.warning(f"Artifact {artifact_name} not found: {file_path}")


def log_training_start(
    logger: logging.Logger,
    model_name: str,
    config: dict[str, Any],
) -> None:
    """
    Log training start information.
    
    Args:
        logger: Logger instance
        model_name: Name of the model being trained
        config: Model configuration dictionary
    """
    logger.info("=" * 60)
    logger.info(f"Starting training for {model_name}")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config}")


def log_training_end(
    logger: logging.Logger,
    model_name: str,
    metrics: dict[str, float],
    duration: float,
) -> None:
    """
    Log training completion information.
    
    Args:
        logger: Logger instance
        model_name: Name of the model that finished training
        metrics: Final training metrics
        duration: Training duration in seconds
    """
    logger.info("=" * 60)
    logger.info(f"Training completed for {model_name}")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info("Final metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.6f}")
    logger.info("=" * 60)


def log_evaluation_results(
    logger: logging.Logger,
    model_name: str,
    metrics: dict[str, float],
    task_type: str,
) -> None:
    """
    Log evaluation results.
    
    Args:
        logger: Logger instance
        model_name: Name of the evaluated model
        metrics: Evaluation metrics
        task_type: Task type (classification or regression)
    """
    logger.info("=" * 60)
    logger.info(f"Evaluation results for {model_name} ({task_type})")
    logger.info("=" * 60)

    if task_type == "classification":
        logger.info("Classification Metrics:")
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            if metric in metrics:
                logger.info(f"  {metric.upper()}: {metrics[metric]:.6f}")
    else:
        logger.info("Regression Metrics:")
        for metric in ["rmse", "mae", "r2", "mape"]:
            if metric in metrics:
                logger.info(f"  {metric.upper()}: {metrics[metric]:.6f}")

    logger.info("=" * 60)


def log_model_registration(
    logger: logging.Logger,
    model_name: str,
    model_version: str,
    stage: str,
) -> None:
    """
    Log model registration information.
    
    Args:
        logger: Logger instance
        model_name: Name of the registered model
        model_version: Model version
        stage: Model stage (None, Staging, Production, Archived)
    """
    logger.info("=" * 60)
    logger.info(f"Model registered: {model_name}")
    logger.info(f"Version: {model_version}")
    logger.info(f"Stage: {stage}")
    logger.info("=" * 60)


def log_model_promotion(
    logger: logging.Logger,
    model_name: str,
    model_version: str,
    from_stage: str,
    to_stage: str,
    reason: str = None,
) -> None:
    """
    Log model promotion information.
    
    Args:
        logger: Logger instance
        model_name: Name of the promoted model
        model_version: Model version
        from_stage: Previous stage
        to_stage: New stage
        reason: Reason for promotion (optional)
    """
    logger.info("=" * 60)
    logger.info(f"Model promoted: {model_name}")
    logger.info(f"Version: {model_version}")
    logger.info(f"From stage: {from_stage}")
    logger.info(f"To stage: {to_stage}")
    if reason:
        logger.info(f"Reason: {reason}")
    logger.info("=" * 60)


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: str,
    additional_info: dict[str, Any] | None = None,
) -> None:
    """
    Log error with context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Context where the error occurred
        additional_info: Additional information to log
    """
    logger.error(f"Error in {context}: {str(error)}")
    if additional_info:
        logger.error(f"Additional context: {additional_info}")
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error details: {error}")
