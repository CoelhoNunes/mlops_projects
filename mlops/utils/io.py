"""I/O utilities for the MLOps project."""

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path: Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, filepath: str | Path) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save the file
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str | Path) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Any: Loaded object
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")

    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj: Any, filepath: str | Path, indent: int = 2) -> None:
    """
    Save object to JSON file.
    
    Args:
        obj: Object to save (must be JSON serializable)
        filepath: Path to save the file
        indent: JSON indentation
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)


def load_json(filepath: str | Path) -> Any:
    """
    Load object from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Any: Loaded object
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, encoding='utf-8') as f:
        return json.load(f)


def save_yaml(obj: Any, filepath: str | Path, default_flow_style: bool = False) -> None:
    """
    Save object to YAML file.
    
    Args:
        obj: Object to save
        filepath: Path to save the file
        default_flow_style: YAML flow style
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(obj, f, default_flow_style=default_flow_style, indent=2, sort_keys=False)


def load_yaml(filepath: str | Path) -> Any:
    """
    Load object from YAML file.
    
    Args:
        filepath: Path to the YAML file
        
    Returns:
        Any: Loaded object
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"YAML file not found: {filepath}")

    with open(filepath, encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_csv(
    df: pd.DataFrame,
    filepath: str | Path,
    index: bool = False,
    **kwargs
) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Path to save the file
        index: Whether to save index
        **kwargs: Additional arguments for pd.to_csv
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    df.to_csv(filepath, index=index, **kwargs)


def load_csv(
    filepath: str | Path,
    **kwargs
) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    return pd.read_csv(filepath, **kwargs)


def save_model_artifacts(
    model: Any,
    filepath: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save model and metadata to artifacts directory.
    
    Args:
        model: Model object to save
        filepath: Base path for artifacts
        metadata: Optional metadata to save
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    # Save model
    model_path = filepath.with_suffix('.pkl')
    save_pickle(model, model_path)

    # Save metadata if provided
    if metadata:
        metadata_path = filepath.with_suffix('.json')
        save_json(metadata, metadata_path)


def load_model_artifacts(
    filepath: str | Path,
    load_metadata: bool = True,
) -> Any | tuple[Any, dict[str, Any]]:
    """
    Load model and optionally metadata from artifacts directory.
    
    Args:
        filepath: Base path for artifacts
        load_metadata: Whether to load metadata
        
    Returns:
        Union[Any, tuple]: Model object or (model, metadata) tuple
    """
    filepath = Path(filepath)

    # Load model
    model_path = filepath.with_suffix('.pkl')
    model = load_pickle(model_path)

    if not load_metadata:
        return model

    # Load metadata if available
    metadata_path = filepath.with_suffix('.json')
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        return model, metadata

    return model, {}


def save_training_artifacts(
    model: Any,
    transformer: Any,
    metrics: dict[str, float],
    config: dict[str, Any],
    artifacts_dir: str | Path,
    model_name: str,
) -> dict[str, str]:
    """
    Save all training artifacts.
    
    Args:
        model: Trained model
        transformer: Feature transformer
        metrics: Training metrics
        config: Model configuration
        artifacts_dir: Directory to save artifacts
        model_name: Name of the model
        
    Returns:
        Dict[str, str]: Dictionary mapping artifact names to file paths
    """
    artifacts_dir = Path(artifacts_dir)
    ensure_dir(artifacts_dir)

    artifacts = {}

    # Save model
    model_path = artifacts_dir / f"{model_name}_model.pkl"
    save_pickle(model, model_path)
    artifacts['model'] = str(model_path)

    # Save transformer
    transformer_path = artifacts_dir / f"{model_name}_transformer.pkl"
    save_pickle(transformer, transformer_path)
    artifacts['transformer'] = str(transformer_path)

    # Save metrics
    metrics_path = artifacts_dir / f"{model_name}_metrics.json"
    save_json(metrics, metrics_path)
    artifacts['metrics'] = str(metrics_path)

    # Save configuration
    config_path = artifacts_dir / f"{model_name}_config.json"
    save_json(config, config_path)
    artifacts['config'] = str(config_path)

    return artifacts


def list_artifacts(artifacts_dir: str | Path, pattern: str = "*") -> list[Path]:
    """
    List all artifacts in a directory matching a pattern.
    
    Args:
        artifacts_dir: Directory to search
        pattern: Glob pattern to match
        
    Returns:
        List[Path]: List of matching artifact paths
    """
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        return []

    return list(artifacts_dir.glob(pattern))


def cleanup_old_artifacts(
    artifacts_dir: str | Path,
    max_files: int = 10,
    pattern: str = "*",
) -> list[Path]:
    """
    Clean up old artifacts, keeping only the most recent ones.
    
    Args:
        artifacts_dir: Directory containing artifacts
        max_files: Maximum number of files to keep
        pattern: Glob pattern to match
        
    Returns:
        List[Path]: List of removed file paths
    """
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        return []

    # Get all matching files with their modification times
    files = []
    for file_path in artifacts_dir.glob(pattern):
        if file_path.is_file():
            mtime = file_path.stat().st_mtime
            files.append((file_path, mtime))

    # Sort by modification time (newest first)
    files.sort(key=lambda x: x[1], reverse=True)

    # Remove old files
    removed_files = []
    for file_path, _ in files[max_files:]:
        try:
            file_path.unlink()
            removed_files.append(file_path)
        except OSError as e:
            print(f"Warning: Could not remove {file_path}: {e}")

    return removed_files


def get_file_size_mb(filepath: str | Path) -> float:
    """
    Get file size in megabytes.
    
    Args:
        filepath: Path to the file
        
    Returns:
        float: File size in MB
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return 0.0

    size_bytes = filepath.stat().st_size
    return size_bytes / (1024 * 1024)


def get_directory_size_mb(directory: str | Path) -> float:
    """
    Get total size of directory in megabytes.
    
    Args:
        directory: Path to the directory
        
    Returns:
        float: Total directory size in MB
    """
    directory = Path(directory)
    if not directory.exists():
        return 0.0

    total_size = 0
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    return total_size / (1024 * 1024)


def save_html(html_content: str, filepath: str | Path) -> None:
    """
    Save HTML content to a file.
    
    Args:
        html_content: HTML string content to save
        filepath: Path to save the HTML file
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
