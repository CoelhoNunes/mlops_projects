#!/usr/bin/env python3
"""
Script to view MLflow results in Chrome browser.
This script will start the MLflow UI server and open it in Chrome.
"""

import subprocess
import sys
import time
import webbrowser
import os
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

def start_mlflow_ui():
    """Start MLflow UI server and open in Chrome."""
    tracking_uri = get_tracking_uri()
    
    print("Starting MLflow UI...")
    print(f"Tracking URI: {tracking_uri}")
    print("=" * 50)
    
    # Start MLflow UI in the background
    try:
        # Start MLflow UI server
        mlflow_process = subprocess.Popen(
            ["mlflow", "ui", "--backend-store-uri", tracking_uri, "--port", "5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("MLflow UI server started successfully!")
        print("Waiting for server to be ready...")
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Open Chrome with MLflow UI
        mlflow_url = "http://localhost:5000"
        print(f"Opening MLflow UI in Chrome: {mlflow_url}")
        
        # Try to open in Chrome specifically
        try:
            # For Windows
            chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe"
            if os.path.exists(chrome_path):
                webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
                webbrowser.get('chrome').open(mlflow_url)
            else:
                # Fallback to default browser
                webbrowser.open(mlflow_url)
        except Exception as e:
            print(f"Could not open Chrome automatically: {e}")
            print(f"Please manually open: {mlflow_url}")
        
        print("\n" + "=" * 50)
        print("MLflow UI is now running!")
        print(f"URL: {mlflow_url}")
        print("\nTo view your pipeline results:")
        print("1. Look for the latest experiment run")
        print("2. Click on the run to see detailed metrics")
        print("3. Check the 'Metrics' tab for R2 Score and RMSE")
        print("4. Check the 'Artifacts' tab for model files and plots")
        print("\nPress Ctrl+C to stop the MLflow UI server")
        
        # Keep the server running
        try:
            mlflow_process.wait()
        except KeyboardInterrupt:
            print("\nStopping MLflow UI server...")
            mlflow_process.terminate()
            mlflow_process.wait()
            print("MLflow UI server stopped.")
            
    except Exception as e:
        print(f"Error starting MLflow UI: {e}")
        print("\nManual instructions:")
        print(f"1. Open terminal and run: mlflow ui --backend-store-uri '{tracking_uri}' --port 5000")
        print("2. Open Chrome and go to: http://localhost:5000")
        return False
    
    return True

def main():
    print("MLflow Results Viewer")
    print("=" * 50)
    
    # Check if MLflow is installed
    try:
        import mlflow
        print("✓ MLflow is installed")
    except ImportError:
        print("✗ MLflow is not installed. Please install it first:")
        print("  pip install mlflow")
        return False
    
    # Start the UI
    success = start_mlflow_ui()
    
    if not success:
        print("\nAlternative viewing options:")
        print("1. Use the simple web interface: python src/web_interface.py")
        print("2. Check the mlruns directory for raw results")
    
    return success

if __name__ == "__main__":
    main()
