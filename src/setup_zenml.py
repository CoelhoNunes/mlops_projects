#!/usr/bin/env python3
"""
Setup script to configure ZenML with MLflow experiment tracker.
"""

import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("Setting up ZenML with MLflow experiment tracker...")
    print("=" * 50)
    
    # Check if ZenML is installed
    if not run_command("zenml version", "Checking ZenML installation"):
        print("ZenML is not installed or not in PATH. Please install it first.")
        return False
    
    # Register MLflow experiment tracker
    if not run_command("zenml experiment-tracker register mlflow_tracker --type=mlflow", 
                      "Registering MLflow experiment tracker"):
        print("Failed to register MLflow tracker. It might already exist.")
    
    # Register a stack with the MLflow tracker
    if not run_command("zenml stack register default_stack -e mlflow_tracker", 
                      "Registering default stack with MLflow tracker"):
        print("Failed to register stack. It might already exist.")
    
    # Set the default stack
    if not run_command("zenml stack set default_stack", 
                      "Setting default stack"):
        print("Failed to set default stack.")
    
    # Verify the setup
    if run_command("zenml stack describe", "Verifying stack configuration"):
        print("\n✓ ZenML setup completed successfully!")
        print("You can now run the pipeline with: python src/run_pipeline.py")
        return True
    else:
        print("\n✗ ZenML setup failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
