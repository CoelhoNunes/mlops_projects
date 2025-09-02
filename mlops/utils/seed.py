"""Seed management for reproducible results."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to set deterministic flags for PyTorch
    """
    # Set environment variable for external libraries
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        # Set deterministic flags
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set seed for all CUDA operations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(seed)

    print(f"Random seed set to {seed} (deterministic: {deterministic})")


def get_seed() -> int | None:
    """Get the current random seed from environment."""
    seed_str = os.environ.get("PYTHONHASHSEED")
    if seed_str:
        try:
            return int(seed_str)
        except ValueError:
            return None
    return None


def set_numpy_seed(seed: int) -> None:
    """Set NumPy random seed."""
    np.random.seed(seed)


def set_torch_seed(seed: int, deterministic: bool = True) -> None:
    """Set PyTorch random seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_random_seed(seed: int) -> None:
    """Set Python random seed."""
    random.seed(seed)


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, CPU otherwise).
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    return device


def set_torch_deterministic() -> None:
    """Set PyTorch to deterministic mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("PyTorch set to deterministic mode")


def set_torch_benchmark() -> None:
    """Set PyTorch to benchmark mode (faster but non-deterministic)."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("PyTorch set to benchmark mode (non-deterministic)")
