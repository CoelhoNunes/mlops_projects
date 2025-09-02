"""Data loading and preprocessing for the MLOps project."""

import glob
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils import get_logger, save_csv, save_json
from ..utils.config import Config


class DataLoader:
    """Data loader for CSV files with automatic splitting and MLflow logging."""

    def __init__(self, config: Config):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.data_info = {}

    def load_data(self) -> tuple[pd.DataFrame, str, str]:
        """
        Load data from CSV files and determine task type.
        
        Returns:
            Tuple: (DataFrame, target_column, task_type)
        """
        self.logger.info("Starting data loading process...")

        # Find CSV files
        csv_files = self._find_csv_files()
        if not csv_files:
            raise FileNotFoundError("No CSV files found in specified paths")

        self.logger.info(f"Found {len(csv_files)} CSV file(s): {csv_files}")

        # Load and merge data
        df = self._load_and_merge_csvs(csv_files)

        # Determine target column
        target_column = self._determine_target_column(df)

        # Determine task type
        task_type = self._determine_task_type(df, target_column)

        # Store data info
        self.data_info = {
            "csv_files": csv_files,
            "target_column": target_column,
            "task_type": task_type,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }

        self.logger.info(f"Data loaded successfully: {df.shape}")
        self.logger.info(f"Target column: {target_column}")
        self.logger.info(f"Task type: {task_type}")

        return df, target_column, task_type

    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            task_type: Task type (classification or regression)
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Starting data splitting process...")

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Get split ratios
        train_ratio = self.config.data.split.train_ratio
        val_ratio = self.config.data.split.val_ratio
        test_ratio = self.config.data.split.test_ratio

        # Calculate actual ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            self.logger.warning(f"Split ratios don't sum to 1.0: {total_ratio}")
            # Normalize ratios
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio

        # First split: train vs (val + test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=self.config.data.split.random_state,
            stratify=y if task_type == "classification" and self.config.data.split.stratified else None,
        )

        # Second split: train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=self.config.data.split.random_state,
            stratify=y_temp if task_type == "classification" and self.config.data.split.stratified else None,
        )

        # Log split information
        split_info = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "train_ratio": len(X_train) / len(df),
            "val_ratio": len(X_val) / len(df),
            "test_ratio": len(X_test) / len(df),
            "total_samples": len(df),
        }

        self.logger.info("Data split completed:")
        for key, value in split_info.items():
            self.logger.info(f"  {key}: {value}")

        # Store split info
        self.data_info["split"] = split_info

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_splits(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        target_column: str,
    ) -> dict[str, str]:
        """
        Save data splits to artifacts directory.
        
        Args:
            X_train, X_val, X_test: Feature DataFrames
            y_train, y_val, y_test: Target Series
            target_column: Name of target column
            
        Returns:
            Dict[str, str]: Dictionary mapping split names to file paths
        """
        self.logger.info("Saving data splits...")

        artifacts_dir = Path(self.config.paths.data_artifacts)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save feature splits
        for name, X_split in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
            filepath = artifacts_dir / f"{name}.csv"
            save_csv(X_split, filepath)
            saved_files[name] = str(filepath)

        # Save target splits
        for name, y_split in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
            filepath = artifacts_dir / f"{name}.csv"
            save_csv(y_split.to_frame(name=target_column), filepath)
            saved_files[name] = str(filepath)

        # Save data info
        info_path = artifacts_dir / "data_info.json"
        save_json(self.data_info, info_path)
        saved_files["data_info"] = str(info_path)

        self.logger.info(f"Data splits saved to {artifacts_dir}")

        return saved_files

    def _find_csv_files(self) -> list[str]:
        """Find CSV files matching the configured paths."""
        csv_files = []

        for path_pattern in self.config.data.paths:
            # Handle glob patterns
            if "*" in path_pattern:
                matches = glob.glob(path_pattern)
                csv_files.extend([f for f in matches if f.endswith('.csv')])
            else:
                # Single file path
                if path_pattern.endswith('.csv') and Path(path_pattern).exists():
                    csv_files.append(path_pattern)

        return sorted(csv_files)

    def _load_and_merge_csvs(self, csv_files: list[str]) -> pd.DataFrame:
        """Load and merge multiple CSV files if needed."""
        if len(csv_files) == 1:
            # Single file
            df = pd.read_csv(csv_files[0])
            self.logger.info(f"Loaded single CSV file: {csv_files[0]}")
        else:
            # Multiple files - merge them
            dfs = []
            for csv_file in csv_files:
                df_temp = pd.read_csv(csv_file)
                dfs.append(df_temp)
                self.logger.info(f"Loaded CSV file: {csv_file} (shape: {df_temp.shape})")

            # Merge files
            if self.config.data.merge_key:
                # Merge on specified key
                df = dfs[0]
                for df_temp in dfs[1:]:
                    df = df.merge(df_temp, on=self.config.data.merge_key, how='inner')
                self.logger.info(f"Merged {len(dfs)} files on key: {self.config.data.merge_key}")
            else:
                # Concatenate files
                df = pd.concat(dfs, ignore_index=True)
                self.logger.info(f"Concatenated {len(dfs)} files")

        return df

    def _determine_target_column(self, df: pd.DataFrame) -> str:
        """Determine target column from configuration or auto-detect."""
        if self.config.data.target_column:
            if self.config.data.target_column in df.columns:
                return self.config.data.target_column
            else:
                raise ValueError(f"Target column '{self.config.data.target_column}' not found in data")

        # Auto-detect target column (last column by default)
        target_column = df.columns[-1]
        self.logger.info(f"Auto-detected target column: {target_column}")

        return target_column

    def _determine_task_type(self, df: pd.DataFrame, target_column: str) -> str:
        """Determine task type from configuration or auto-detect."""
        if self.config.data.task_type != "auto":
            return self.config.data.task_type

        # Auto-detect based on target column data type
        target_dtype = df[target_column].dtype

        if np.issubdtype(target_dtype, np.number):
            # Check if it's classification (integer with few unique values)
            unique_values = df[target_column].nunique()
            if unique_values <= 20:  # Threshold for classification
                task_type = "classification"
            else:
                task_type = "regression"
        else:
            task_type = "classification"

        self.logger.info(f"Auto-detected task type: {task_type} (target dtype: {target_dtype})")

        return task_type

    def log_to_mlflow(self, run_id: str) -> None:
        """Log data information to MLflow."""
        try:
            # Log data info as parameters
            mlflow.log_params({
                "data_shape": f"{self.data_info['shape'][0]}x{self.data_info['shape'][1]}",
                "target_column": self.data_info["target_column"],
                "task_type": self.data_info["task_type"],
                "csv_files_count": len(self.data_info["csv_files"]),
                "memory_usage_mb": round(self.data_info["memory_usage_mb"], 2),
            })

            # Log data info as artifact
            info_path = Path(self.config.paths.data_artifacts) / "data_info.json"
            if info_path.exists():
                mlflow.log_artifact(str(info_path), "data_info")

            # Log split information if available
            if "split" in self.data_info:
                mlflow.log_params({
                    "train_samples": self.data_info["split"]["train_samples"],
                    "val_samples": self.data_info["split"]["val_samples"],
                    "test_samples": self.data_info["split"]["test_samples"],
                })

            self.logger.info("Data information logged to MLflow")

        except Exception as e:
            self.logger.warning(f"Could not log data info to MLflow: {e}")

    def get_data_summary(self) -> dict[str, any]:
        """Get summary of loaded data."""
        return self.data_info.copy()

    def validate_data_quality(self, df: pd.DataFrame) -> dict[str, any]:
        """Perform basic data quality checks."""
        quality_info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_percentage": (df.duplicated().sum() / len(df) * 100),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        }

        # Add data type information
        quality_info["dtypes"] = df.dtypes.to_dict()
        quality_info["numeric_columns"] = list(df.select_dtypes(include=[np.number]).columns)
        quality_info["categorical_columns"] = list(df.select_dtypes(include=['object', 'category']).columns)

        # Add target column specific info
        if hasattr(self, 'data_info') and 'target_column' in self.data_info:
            target_col = self.data_info['target_column']
            if target_col in df.columns:
                target_series = df[target_col]
                quality_info["target_info"] = {
                    "unique_values": target_series.nunique(),
                    "value_counts": target_series.value_counts().to_dict(),
                    "missing_values": target_series.isnull().sum(),
                }

        return quality_info
