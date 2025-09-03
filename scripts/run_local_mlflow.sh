#!/bin/bash

# MLOps Local MLflow Server Script
# Starts a local MLflow tracking server and registry for development

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Default configuration
MLFLOW_PORT=${MLFLOW_PORT:-5000}
MLFLOW_HOST=${MLFLOW_HOST:-"0.0.0.0"}
MLFLOW_BACKEND_STORE_URI=${MLFLOW_BACKEND_STORE_URI:-"sqlite:///mlruns.db"}
MLFLOW_DEFAULT_ARTIFACT_ROOT=${MLFLOW_DEFAULT_ARTIFACT_ROOT:-"./mlruns"}
MLFLOW_WORKERS=${MLFLOW_WORKERS:-4}

print_header "MLOps Local MLflow Server"
print_status "Starting local MLflow tracking server..."

# Check if MLflow is installed
if ! command -v mlflow &> /dev/null; then
    print_warning "MLflow not found. Installing..."
    pip install mlflow
fi

# Create necessary directories
print_status "Creating MLflow directories..."
mkdir -p mlruns
mkdir -p logs

# Check if port is already in use
if lsof -Pi :$MLFLOW_PORT -sTCP:LISTEN -t >/dev/null ; then
    print_warning "Port $MLFLOW_PORT is already in use. Attempting to kill existing process..."
    lsof -ti:$MLFLOW_PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start MLflow server
print_status "Starting MLflow server on http://$MLFLOW_HOST:$MLFLOW_PORT"
print_status "Backend store: $MLFLOW_BACKEND_STORE_URI"
print_status "Artifact root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"
print_status "Workers: $MLFLOW_WORKERS"

# Start MLflow server in background
mlflow server \
    --host $MLFLOW_HOST \
    --port $MLFLOW_PORT \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT \
    --workers $MLFLOW_WORKERS \
    > logs/mlflow_server.log 2>&1 &

MLFLOW_PID=$!

# Wait for server to start
print_status "Waiting for server to start..."
sleep 5

# Check if server is running
if curl -s "http://$MLFLOW_HOST:$MLFLOW_PORT/health" > /dev/null 2>&1; then
    print_status "✅ MLflow server started successfully!"
    print_status "Server PID: $MLFLOW_PID"
    print_status "Tracking UI: http://$MLFLOW_HOST:$MLFLOW_PORT"
    print_status "Log file: logs/mlflow_server.log"
    
    # Save PID to file for easy cleanup
    echo $MLFLOW_PID > .mlflow_server.pid
    
    # Display server info
    echo ""
    print_header "MLflow Server Information"
    echo "URL: http://$MLFLOW_HOST:$MLFLOW_PORT"
    echo "Backend: $MLFLOW_BACKEND_STORE_URI"
    echo "Artifacts: $MLFLOW_DEFAULT_ARTIFACT_ROOT"
    echo "PID: $MLFLOW_PID"
    echo ""
    echo "To stop the server, run:"
    echo "  kill $MLFLOW_PID"
    echo "  or"
    echo "  ./scripts/stop_local_mlflow.sh"
    echo ""
    echo "To view logs:"
    echo "  tail -f logs/mlflow_server.log"
    
else
    print_warning "❌ MLflow server failed to start. Check logs:"
    tail -n 20 logs/mlflow_server.log
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    print_status "Shutting down MLflow server..."
    if [ -f .mlflow_server.pid ]; then
        PID=$(cat .mlflow_server.pid)
        kill $PID 2>/dev/null || true
        rm -f .mlflow_server.pid
    fi
    print_status "MLflow server stopped."
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Keep script running and monitor server
print_status "Monitoring MLflow server... (Press Ctrl+C to stop)"
while true; do
    if ! curl -s "http://$MLFLOW_HOST:$MLFLOW_PORT/health" > /dev/null 2>&1; then
        print_warning "MLflow server appears to have stopped unexpectedly"
        break
    fi
    sleep 10
done
