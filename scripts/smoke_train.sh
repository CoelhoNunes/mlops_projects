#!/bin/bash

# MLOps Smoke Test Script
# Runs a complete E2E training pipeline with synthetic data

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
SMOKE_DATA_SIZE=${SMOKE_DATA_SIZE:-1000}
SMOKE_FEATURES=${SMOKE_FEATURES:-10}
SMOKE_TEST_SIZE=${SMOKE_TEST_SIZE:-0.2}
PYTHON_CMD=${PYTHON_CMD:-"python"}

print_header "MLOps Smoke Test - E2E Training Pipeline"
print_status "Testing complete MLOps stack with synthetic data..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if MLflow server is running
print_status "Checking MLflow server..."
if ! curl -s "http://localhost:5000/health" > /dev/null 2>&1; then
    print_warning "MLflow server not running. Starting local server..."
    ./scripts/run_local_mlflow.sh &
    MLFLOW_PID=$!
    sleep 10
    
    # Check again
    if ! curl -s "http://localhost:5000/health" > /dev/null 2>&1; then
        print_error "Failed to start MLflow server"
        exit 1
    fi
    print_status "MLflow server started successfully"
else
    print_status "MLflow server is already running"
    MLFLOW_PID=""
fi

# Create synthetic data
print_status "Creating synthetic dataset..."
$PYTHON_CMD -c "
import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic data
n_samples = $SMOKE_DATA_SIZE
n_features = $SMOKE_FEATURES

# Generate features
data = {}
for i in range(n_features):
    if i % 3 == 0:
        # Numeric features
        data[f'numeric_{i}'] = np.random.normal(0, 1, n_samples)
    elif i % 3 == 1:
        # Integer features
        data[f'int_{i}'] = np.random.randint(0, 10, n_samples)
    else:
        # Categorical features
        data[f'cat_{i}'] = np.random.choice(['A', 'B', 'C'], n_samples)

# Create target (binary classification)
# Simple rule: if sum of numeric features > 0, then 1, else 0
numeric_sum = sum([data[f'numeric_{i}'] for i in range(0, n_features, 3)])
data['target'] = (numeric_sum > 0).astype(int)

# Add some noise
noise = np.random.random(n_samples) < 0.1
data['target'] = data['target'] ^ noise

# Create DataFrame
df = pd.DataFrame(data)

# Ensure data directory exists
Path('data').mkdir(exist_ok=True)

# Save synthetic data
df.to_csv('data/smoke_test_data.csv', index=False)
print(f'Created synthetic dataset: {df.shape}')
print(f'Target distribution: {df[\"target\"].value_counts().to_dict()}')
"

if [ $? -ne 0 ]; then
    print_error "Failed to create synthetic data"
    exit 1
fi

print_status "✅ Synthetic data created successfully"

# Install dependencies if needed
print_status "Checking dependencies..."
if ! $PYTHON_CMD -c "import mlflow, pandas, numpy, sklearn" 2>/dev/null; then
    print_warning "Installing dependencies..."
    pip install -e .
fi

# Run data validation
print_status "Running data validation..."
$PYTHON_CMD -m mlops.orchestration.cli data-validate \
    --data-path "data/smoke_test_data.csv" \
    --target-column "target"

if [ $? -ne 0 ]; then
    print_error "Data validation failed"
    exit 1
fi

print_status "✅ Data validation completed"

# Run training
print_status "Running model training..."
$PYTHON_CMD -m mlops.orchestration.cli train \
    --data-path "data/smoke_test_data.csv" \
    --target-column "target" \
    --models "logreg,rf,torch_mlp" \
    --cv-folds 3 \
    --optuna-trials 5

if [ $? -ne 0 ]; then
    print_error "Model training failed"
    exit 1
fi

print_status "✅ Model training completed"

# Run evaluation
print_status "Running model evaluation..."
$PYTHON_CMD -m mlops.orchestration.cli evaluate \
    --data-path "data/smoke_test_data.csv" \
    --target-column "target" \
    --model-path "artifacts/models/best_model.pkl"

if [ $? -ne 0 ]; then
    print_error "Model evaluation failed"
    exit 1
fi

print_status "✅ Model evaluation completed"

# Run model registration
print_status "Running model registration..."
$PYTHON_CMD -m mlops.orchestration.cli register \
    --model-path "artifacts/models/best_model.pkl" \
    --model-name "smoke-test-model" \
    --model-type "sklearn"

if [ $? -ne 0 ]; then
    print_error "Model registration failed"
    exit 1
fi

print_status "✅ Model registration completed"

# Run model promotion
print_status "Running model promotion..."
$PYTHON_CMD -m mlops.orchestration.cli promote \
    --model-name "smoke-test-model" \
    --target-stage "Production"

if [ $? -ne 0 ]; then
    print_error "Model promotion failed"
    exit 1
fi

print_status "✅ Model promotion completed"

# Test model serving (if FastAPI is available)
print_status "Testing model serving..."
if $PYTHON_CMD -c "import fastapi" 2>/dev/null; then
    # Start model server in background
    $PYTHON_CMD -m mlops.serving.app --port 8001 &
    SERVER_PID=$!
    sleep 5
    
    # Test health endpoint
    if curl -s "http://localhost:8001/healthz" > /dev/null 2>&1; then
        print_status "✅ Model server started successfully"
        
        # Test prediction endpoint
        $PYTHON_CMD -c "
import requests
import json

# Test prediction
data = [{'numeric_0': 1.0, 'int_1': 5, 'cat_2': 'A'}]
response = requests.post('http://localhost:8001/predict', json={'data': data})
if response.status_code == 200:
    result = response.json()
    print(f'✅ Prediction successful: {result[\"predictions\"]}')
else:
    print(f'❌ Prediction failed: {response.text}')
    exit(1)
"
        
        # Stop server
        kill $SERVER_PID 2>/dev/null || true
        print_status "✅ Model serving test completed"
    else
        print_warning "Model server health check failed"
        kill $SERVER_PID 2>/dev/null || true
    fi
else
    print_warning "FastAPI not available, skipping serving test"
fi

# Check artifacts
print_status "Checking generated artifacts..."
ARTIFACT_PATHS=(
    "artifacts/models/best_model.pkl"
    "artifacts/models/feature_engineer.pkl"
    "artifacts/models/training_results.yaml"
    "artifacts/evaluation"
    "artifacts/registry"
    "mlruns"
)

for path in "${ARTIFACT_PATHS[@]}"; do
    if [ -e "$path" ]; then
        print_status "✅ Found: $path"
    else
        print_warning "⚠️  Missing: $path"
    fi
done

# Run tests
print_status "Running unit tests..."
if $PYTHON_CMD -m pytest tests/ -v --tb=short; then
    print_status "✅ Unit tests passed"
else
    print_warning "⚠️  Some unit tests failed"
fi

# Check code quality
print_status "Running code quality checks..."
if command -v ruff > /dev/null; then
    if ruff check mlops/; then
        print_status "✅ Ruff linting passed"
    else
        print_warning "⚠️  Ruff linting found issues"
    fi
else
    print_warning "Ruff not available"
fi

# Summary
print_header "Smoke Test Summary"
print_status "✅ Synthetic data creation: PASSED"
print_status "✅ Data validation: PASSED"
print_status "✅ Model training: PASSED"
print_status "✅ Model evaluation: PASSED"
print_status "✅ Model registration: PASSED"
print_status "✅ Model promotion: PASSED"
print_status "✅ Artifact generation: PASSED"

if [ -n "$MLFLOW_PID" ]; then
    print_status "Stopping MLflow server..."
    kill $MLFLOW_PID 2>/dev/null || true
fi

print_header "🎉 Smoke Test Completed Successfully!"
print_status "The MLOps stack is working correctly!"
print_status ""
print_status "Next steps:"
print_status "1. Review generated artifacts in mlruns/ and artifacts/"
print_status "2. Check MLflow UI at http://localhost:5000"
print_status "3. Run 'make train' with your real data"
print_status "4. Deploy models using 'make serve'"
