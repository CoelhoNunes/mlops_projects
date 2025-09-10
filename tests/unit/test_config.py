"""Unit tests for basic project functionality."""

import os
import tempfile
from unittest.mock import patch

import pytest


class TestBasicFunctionality:
    """Test basic project functionality."""

    def test_python_imports(self):
        """Test that basic Python imports work."""
        import sys
        from pathlib import Path
        
        # Add src to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        # Test that we can import the training modules
        try:
            import src.train
            assert src.train is not None
        except ImportError as e:
            pytest.fail(f"Failed to import training script: {e}")

    def test_training_script_functions(self):
        """Test that training script has expected functions."""
        import sys
        from pathlib import Path
        
        # Add src to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        import src.train
        
        # Check that main functions exist
        assert hasattr(src.train, "load_data")
        assert hasattr(src.train, "split_data")
        assert hasattr(src.train, "preprocess_data")
        assert hasattr(src.train, "train_model")
        assert hasattr(src.train, "evaluate_model")
        assert hasattr(src.train, "log_to_mlflow")
        assert hasattr(src.train, "main")

    def test_customer_training_script_functions(self):
        """Test that customer training script has expected functions."""
        import sys
        from pathlib import Path
        
        # Add src to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
        
        import src.train_customer
        
        # Check that main functions exist
        assert hasattr(src.train_customer, "load_customer_data")
        assert hasattr(src.train_customer, "engineer_features")
        assert hasattr(src.train_customer, "split_data")
        assert hasattr(src.train_customer, "train_models")
        assert hasattr(src.train_customer, "evaluate_model")
        assert hasattr(src.train_customer, "log_to_mlflow")
        assert hasattr(src.train_customer, "main")

    def test_environment_variables(self):
        """Test that environment variables can be set."""
        test_var = "TEST_VARIABLE"
        test_value = "test_value"
        
        with patch.dict(os.environ, {test_var: test_value}):
            assert os.environ[test_var] == test_value

    def test_file_operations(self):
        """Test basic file operations."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            assert os.path.exists(temp_path)
            with open(temp_path, "r") as f:
                content = f.read()
            assert content == "test content"
        finally:
            os.unlink(temp_path)