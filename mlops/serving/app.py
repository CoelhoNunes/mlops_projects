"""
FastAPI model serving application for MLOps project.

Provides REST API endpoints for model inference, health checks,
and comprehensive monitoring with MLflow integration.
"""

import os
import time
from typing import Any

import mlflow
import pandas as pd
import prometheus_client
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field, validator

from ..utils.config import Config
from ..utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions', ['model_name', 'status'])
PREDICTION_DURATION = Histogram('model_prediction_duration_seconds', 'Time spent processing prediction', ['model_name'])
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time spent loading model', ['model_name'])
MODEL_MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Memory usage of loaded model', ['model_name'])


class PredictionRequest(BaseModel):
    """Request schema for model predictions."""

    data: list[dict[str, Any]] = Field(..., description="List of feature dictionaries for prediction")

    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data list cannot be empty")
        if len(v) > 1000:  # Limit batch size
            raise ValueError("Batch size cannot exceed 1000")
        return v


class PredictionResponse(BaseModel):
    """Response schema for model predictions."""

    predictions: list[float | int | str] = Field(..., description="Model predictions")
    probabilities: list[list[float]] | None = Field(None, description="Prediction probabilities (if available)")
    model_name: str = Field(..., description="Name of the model used for prediction")
    model_version: str = Field(..., description="Version of the model used for prediction")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Service status")
    model_name: str = Field(..., description="Name of the loaded model")
    model_version: str = Field(..., description="Version of the loaded model")
    model_stage: str = Field(..., description="Stage of the loaded model")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    timestamp: str = Field(..., description="Health check timestamp")


class ModelServer:
    """
    FastAPI-based model server with MLflow integration.
    
    Provides REST API endpoints for model inference, health monitoring,
    and comprehensive logging.
    """

    def __init__(self, config: Config):
        """
        Initialize the model server.
        
        Args:
            config: Configuration object containing server parameters
        """
        self.config = config
        self.mlflow_config = config.mlflow
        self.serving_config = config.serving if hasattr(config, 'serving') else {}

        # Initialize FastAPI app
        self.app = FastAPI(
            title="MLOps Model Server",
            description="Production model serving API with MLflow integration",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize model and metadata
        self.model = None
        self.model_name = None
        self.model_version = None
        self.model_stage = None
        self.feature_engineer = None
        self.start_time = time.time()

        # Setup MLflow
        self._setup_mlflow()

        # Load production model
        self._load_production_model()

        # Setup routes
        self._setup_routes()

        # Setup middleware
        self._setup_middleware()

        logger.info("Model server initialized successfully")

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking and registry."""
        try:
            # Set tracking URI
            tracking_uri = self.mlflow_config.tracking_uri
            mlflow.set_tracking_uri(tracking_uri)

            # Set registry URI if different
            registry_uri = getattr(self.mlflow_config, 'registry_uri', tracking_uri)
            if registry_uri != tracking_uri:
                mlflow.set_registry_uri(registry_uri)

            logger.info(f"MLflow setup complete: tracking_uri={tracking_uri}")

        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise

    def _load_production_model(self) -> None:
        """Load the production model from MLflow registry."""
        try:
            logger.info("Loading production model from MLflow registry")

            # Get model name from config or environment
            model_name = self.serving_config.get("model_name", os.getenv("MLOPS_MODEL_NAME", "best-model"))

            # Load model from registry
            model_uri = f"models:/{model_name}/Production"

            start_time = time.time()
            self.model = mlflow.pyfunc.load_model(model_uri)
            load_time = time.time() - start_time

            # Get model metadata
            client = mlflow.tracking.MlflowClient()
            model_info = client.get_registered_model(model_name)
            latest_production = client.get_latest_versions(model_name, stages=["Production"])

            if latest_production:
                self.model_version = str(latest_production[0].version)
                self.model_stage = "Production"
            else:
                self.model_version = "unknown"
                self.model_stage = "unknown"

            self.model_name = model_name

            # Update Prometheus metrics
            MODEL_LOAD_TIME.labels(model_name=model_name).set(load_time)

            # Try to load feature engineer
            self._load_feature_engineer()

            logger.info(f"Successfully loaded model: {model_name} v{self.model_version} ({self.model_stage})")
            logger.info(f"Model load time: {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            # Don't raise here - server can start without model for health checks
            self.model = None

    def _load_feature_engineer(self) -> None:
        """Load the feature engineering pipeline if available."""
        try:
            # Try to load feature engineer from MLflow artifacts
            feature_engineer_path = f"models:/{self.model_name}/Production"

            # This is a simplified approach - in practice you might need to load
            # the feature engineer separately or include it in the model pipeline
            logger.info("Feature engineer loading not implemented - using raw features")

        except Exception as e:
            logger.warning(f"Failed to load feature engineer: {e}")

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/", response_class=JSONResponse)
        async def root():
            """Root endpoint with service information."""
            return {
                "service": "MLOps Model Server",
                "version": "1.0.0",
                "status": "running",
                "model": self.model_name,
                "docs": "/docs"
            }

        @self.app.get("/healthz", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            import psutil

            # Get memory usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB

            # Update memory metric
            if self.model_name:
                MODEL_MEMORY_USAGE.labels(model_name=self.model_name).set(memory_usage * 1024 * 1024)

            return HealthResponse(
                status="healthy" if self.model is not None else "unhealthy",
                model_name=self.model_name or "none",
                model_version=self.model_version or "none",
                model_stage=self.model_stage or "none",
                uptime_seconds=time.time() - self.start_time,
                memory_usage_mb=memory_usage,
                timestamp=pd.Timestamp.now().isoformat()
            )

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Model prediction endpoint."""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            start_time = time.time()

            try:
                # Convert request data to DataFrame
                input_df = pd.DataFrame(request.data)

                # Validate input features
                self._validate_input_features(input_df)

                # Make predictions
                predictions = self.model.predict(input_df)

                # Get probabilities if available
                probabilities = None
                if hasattr(self.model, 'predict_proba'):
                    try:
                        probabilities = self.model.predict_proba(input_df).tolist()
                    except Exception:
                        pass

                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Update Prometheus metrics
                PREDICTION_COUNTER.labels(model_name=self.model_name, status="success").inc()
                PREDICTION_DURATION.labels(model_name=self.model_name).observe(processing_time / 1000)

                # Log prediction
                logger.info(f"Prediction successful: {len(predictions)} samples, {processing_time:.2f}ms")

                return PredictionResponse(
                    predictions=predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                    probabilities=probabilities,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    prediction_timestamp=pd.Timestamp.now().isoformat(),
                    processing_time_ms=processing_time
                )

            except Exception as e:
                # Update error metrics
                PREDICTION_COUNTER.labels(model_name=self.model_name, status="error").inc()

                logger.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            return Response(
                content=prometheus_client.generate_latest(),
                media_type="text/plain"
            )

        @self.app.get("/model-info")
        async def model_info():
            """Get information about the loaded model."""
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            return {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "model_stage": self.model_stage,
                "model_type": type(self.model).__name__,
                "loaded_at": pd.Timestamp.fromtimestamp(self.start_time).isoformat(),
                "uptime_seconds": time.time() - self.start_time
            }

        @self.app.post("/reload-model")
        async def reload_model():
            """Reload the production model."""
            try:
                logger.info("Reloading production model")
                self._load_production_model()

                if self.model is not None:
                    return {"status": "success", "message": "Model reloaded successfully"}
                else:
                    return {"status": "error", "message": "Failed to reload model"}

            except Exception as e:
                logger.error(f"Model reload failed: {e}")
                raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

    def _setup_middleware(self) -> None:
        """Setup request/response middleware."""

        @self.app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            """Add processing time header to responses."""
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response

        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """Log all requests."""
            start_time = time.time()

            # Log request
            logger.info(f"Request: {request.method} {request.url.path}")

            response = await call_next(request)

            # Log response
            process_time = time.time() - start_time
            logger.info(f"Response: {response.status_code} ({process_time:.3f}s)")

            return response

    def _validate_input_features(self, input_df: pd.DataFrame) -> None:
        """Validate input features for prediction."""
        try:
            # Check for required columns if feature names are known
            if hasattr(self.model, 'feature_names_in_'):
                required_features = set(self.model.feature_names_in_)
                input_features = set(input_df.columns)

                missing_features = required_features - input_features
                if missing_features:
                    raise ValueError(f"Missing required features: {missing_features}")

                extra_features = input_features - required_features
                if extra_features:
                    logger.warning(f"Extra features provided: {extra_features}")

            # Check for null values
            null_counts = input_df.isnull().sum()
            if null_counts.sum() > 0:
                logger.warning(f"Input contains null values: {null_counts[null_counts > 0].to_dict()}")

            # Check data types
            for col in input_df.columns:
                if input_df[col].dtype == 'object':
                    logger.warning(f"Column {col} has object dtype - may cause issues")

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValueError(f"Input validation failed: {str(e)}")

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the FastAPI server."""
        logger.info(f"Starting model server on {host}:{port}")

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            **kwargs
        )


def create_app(config: Config = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config: Configuration object (optional)
        
    Returns:
        Configured FastAPI application
    """
    if config is None:
        # Load default config
        config = Config.from_yaml("mlops/config/settings.yaml")

    # Setup logging
    setup_logging(config.logging)

    # Create model server
    server = ModelServer(config)

    return server.app


def main():
    """Main entry point for the model server."""
    import argparse

    parser = argparse.ArgumentParser(description="MLOps Model Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", default="mlops/config/settings.yaml", help="Configuration file path")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config.from_yaml(args.config)

        # Create and run server
        server = ModelServer(config)
        server.run(
            host=args.host,
            port=args.port,
            reload=args.reload
        )

    except Exception as e:
        logger.error(f"Failed to start model server: {e}")
        raise


if __name__ == "__main__":
    main()
