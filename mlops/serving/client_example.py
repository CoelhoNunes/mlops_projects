"""
Client example for MLOps model serving API.

Demonstrates how to interact with the FastAPI model server
including health checks, predictions, and error handling.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
import requests


class ModelServerClient:
    """
    Client for interacting with the MLOps model serving API.
    
    Provides methods for health checks, predictions, and model information
    with comprehensive error handling and retry logic.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the model server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'MLOps-Client/1.0.0'
        })

    def health_check(self) -> dict[str, Any]:
        """
        Check the health of the model server.
        
        Returns:
            Health status information
        """
        try:
            response = self.session.get(
                f"{self.base_url}/healthz",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"Health check failed: {str(e)}",
                "timestamp": pd.Timestamp.now().isoformat()
            }

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information
        """
        try:
            response = self.session.get(
                f"{self.base_url}/model-info",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"Failed to get model info: {str(e)}"
            }

    def predict(
        self,
        data: list[dict[str, Any]] | pd.DataFrame,
        retry_count: int = 3,
        retry_delay: float = 1.0
    ) -> dict[str, Any]:
        """
        Make predictions using the model server.
        
        Args:
            data: Input data for prediction (list of dicts or DataFrame)
            retry_count: Number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Prediction results
        """
        # Convert DataFrame to list of dicts if needed
        if isinstance(data, pd.DataFrame):
            input_data = data.to_dict('records')
        else:
            input_data = data

        # Validate input data
        if not input_data:
            return {
                "status": "error",
                "error": "Input data cannot be empty"
            }

        # Prepare request payload
        payload = {"data": input_data}

        # Retry logic
        for attempt in range(retry_count):
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt == retry_count - 1:
                    return {
                        "status": "error",
                        "error": f"Prediction failed after {retry_count} attempts: {str(e)}"
                    }

                # Wait before retry
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    def reload_model(self) -> dict[str, Any]:
        """
        Reload the production model.
        
        Returns:
            Reload status
        """
        try:
            response = self.session.post(
                f"{self.base_url}/reload-model",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": f"Model reload failed: {str(e)}"
            }

    def get_metrics(self) -> str:
        """
        Get Prometheus metrics from the server.
        
        Returns:
            Metrics in Prometheus format
        """
        try:
            response = self.session.get(
                f"{self.base_url}/metrics",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.text

        except requests.exceptions.RequestException as e:
            return f"# Failed to get metrics: {str(e)}"

    def batch_predict(
        self,
        data: list[dict[str, Any]] | pd.DataFrame,
        batch_size: int = 100,
        max_workers: int = 4
    ) -> list[dict[str, Any]]:
        """
        Make predictions in batches for large datasets.
        
        Args:
            data: Input data for prediction
            batch_size: Size of each batch
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of prediction results
        """
        # Convert DataFrame to list of dicts if needed
        if isinstance(data, pd.DataFrame):
            input_data = data.to_dict('records')
        else:
            input_data = data

        # Split into batches
        batches = [
            input_data[i:i + batch_size]
            for i in range(0, len(input_data), batch_size)
        ]

        results = []

        # Process batches sequentially (can be parallelized with ThreadPoolExecutor)
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} samples)")

            batch_result = self.predict(batch)
            results.append(batch_result)

            # Small delay between batches to avoid overwhelming the server
            time.sleep(0.1)

        return results

    def test_connection(self) -> bool:
        """
        Test the connection to the model server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/",
                timeout=5
            )
            return response.status_code == 200

        except requests.exceptions.RequestException:
            return False


def create_sample_data(n_samples: int = 10, n_features: int = 5) -> list[dict[str, Any]]:
    """
    Create sample data for testing the model server.
    
    Args:
        n_samples: Number of samples to create
        n_features: Number of features per sample
        
    Returns:
        List of sample data dictionaries
    """
    np.random.seed(42)  # For reproducibility

    sample_data = []
    for i in range(n_samples):
        sample = {}
        for j in range(n_features):
            feature_name = f"feature_{j}"

            # Mix of numeric and categorical features
            if j % 3 == 0:
                # Numeric feature
                sample[feature_name] = float(np.random.normal(0, 1))
            elif j % 3 == 1:
                # Integer feature
                sample[feature_name] = int(np.random.randint(0, 10))
            else:
                # Categorical feature
                sample[feature_name] = np.random.choice(['A', 'B', 'C'])

        sample_data.append(sample)

    return sample_data


def print_prediction_results(results: dict[str, Any]) -> None:
    """
    Print prediction results in a formatted way.
    
    Args:
        results: Prediction results from the server
    """
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)

    if results.get("status") == "error":
        print(f"❌ Error: {results['error']}")
        return

    print(f"✅ Model: {results['model_name']} v{results['model_version']}")
    print(f"📊 Predictions: {len(results['predictions'])} samples")
    print(f"⏱️  Processing time: {results['processing_time_ms']:.2f}ms")
    print(f"🕐 Timestamp: {results['prediction_timestamp']}")

    if results.get('probabilities'):
        print("🎯 Probabilities available: Yes")

    # Show first few predictions
    print("\n📈 Sample Predictions:")
    for i, pred in enumerate(results['predictions'][:5]):
        prob_str = ""
        if results.get('probabilities'):
            prob_str = f" (prob: {results['probabilities'][i][0]:.3f})"
        print(f"  Sample {i+1}: {pred}{prob_str}")

    if len(results['predictions']) > 5:
        print(f"  ... and {len(results['predictions']) - 5} more")


def main():
    """Main example demonstrating client usage."""
    print("🚀 MLOps Model Server Client Example")
    print("="*50)

    # Initialize client
    client = ModelServerClient()

    # Test connection
    print("\n🔌 Testing connection...")
    if not client.test_connection():
        print("❌ Failed to connect to model server. Make sure it's running on http://localhost:8000")
        return

    print("✅ Connected to model server")

    # Health check
    print("\n🏥 Health check...")
    health = client.health_check()
    print(f"Status: {health.get('status', 'unknown')}")
    if health.get('status') == 'healthy':
        print(f"Model: {health.get('model_name')} v{health.get('model_version')}")
        print(f"Uptime: {health.get('uptime_seconds', 0):.1f}s")
        print(f"Memory: {health.get('memory_usage_mb', 0):.1f}MB")
    else:
        print(f"Health check failed: {health.get('error', 'Unknown error')}")

    # Get model info
    print("\n📋 Model information...")
    model_info = client.get_model_info()
    if 'error' not in model_info:
        print(f"Model: {model_info.get('model_name')}")
        print(f"Version: {model_info.get('model_version')}")
        print(f"Stage: {model_info.get('model_stage')}")
        print(f"Type: {model_info.get('model_type')}")
    else:
        print(f"Failed to get model info: {model_info['error']}")

    # Create sample data
    print("\n📊 Creating sample data...")
    sample_data = create_sample_data(n_samples=5, n_features=4)
    print(f"Created {len(sample_data)} samples with {len(sample_data[0])} features each")

    # Show sample data
    print("\n📝 Sample data:")
    for i, sample in enumerate(sample_data):
        print(f"  Sample {i+1}: {sample}")

    # Make predictions
    print("\n🎯 Making predictions...")
    results = client.predict(sample_data)

    # Print results
    print_prediction_results(results)

    # Test batch prediction
    print("\n📦 Testing batch prediction...")
    large_sample_data = create_sample_data(n_samples=25, n_features=4)
    batch_results = client.batch_predict(large_sample_data, batch_size=10)

    successful_batches = sum(1 for r in batch_results if r.get('status') != 'error')
    print(f"✅ Successfully processed {successful_batches}/{len(batch_results)} batches")

    # Get metrics
    print("\n📊 Getting metrics...")
    metrics = client.get_metrics()
    print(f"Retrieved {len(metrics.splitlines())} metric lines")

    # Show some key metrics
    metric_lines = metrics.splitlines()
    for line in metric_lines:
        if any(key in line for key in ['model_predictions_total', 'model_prediction_duration_seconds']):
            print(f"  {line}")

    print("\n🎉 Client example completed successfully!")


def example_with_real_data():
    """Example using real CSV data if available."""
    print("\n📁 Example with real data...")

    # Try to load data from common locations
    data_paths = [
        "data/olist_customers_dataset.csv",
        "data/*.csv",
        "*.csv"
    ]

    import glob
    csv_files = []
    for path in data_paths:
        csv_files.extend(glob.glob(path))

    if not csv_files:
        print("❌ No CSV files found. Using sample data instead.")
        return

    print(f"📂 Found CSV files: {csv_files}")

    # Load first CSV file
    try:
        df = pd.read_csv(csv_files[0])
        print(f"✅ Loaded {csv_files[0]} with shape {df.shape}")

        # Select numeric columns for features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 5:
            numeric_cols = numeric_cols[:5]  # Limit to 5 features

        if numeric_cols:
            # Create sample data from real dataset
            sample_df = df[numeric_cols].head(10)
            sample_data = sample_df.to_dict('records')

            print(f"📊 Using {len(numeric_cols)} numeric features: {numeric_cols}")
            print(f"📝 Sample data shape: {sample_df.shape}")

            # Make predictions
            client = ModelServerClient()
            results = client.predict(sample_data)

            print_prediction_results(results)

        else:
            print("❌ No numeric columns found in CSV")

    except Exception as e:
        print(f"❌ Failed to load CSV data: {e}")


if __name__ == "__main__":
    # Run main example
    main()

    # Run example with real data if available
    example_with_real_data()

    print("\n" + "="*50)
    print("📚 Usage Examples:")
    print("="*50)
    print("""
# Basic usage
client = ModelServerClient("http://localhost:8000")

# Health check
health = client.health_check()

# Make predictions
data = [{"feature_1": 1.0, "feature_2": "A"}]
results = client.predict(data)

# Batch predictions
large_data = [{"feature_1": i, "feature_2": "B"} for i in range(100)]
batch_results = client.batch_predict(large_data, batch_size=20)

# Get metrics
metrics = client.get_metrics()
    """)
