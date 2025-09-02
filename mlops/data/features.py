"""
Feature engineering module for MLOps project.

Provides comprehensive feature preprocessing using sklearn ColumnTransformer
for numeric and categorical features with MLflow logging.
"""

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from ..utils.io import ensure_dir, load_pickle, save_pickle
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline using sklearn ColumnTransformer.
    
    Handles numeric and categorical preprocessing, feature selection,
    and provides sklearn-compatible transformer interface.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary containing feature engineering parameters
        """
        self.config = config
        self.feature_config = config.get("features", {})
        self.transformer = None
        self.feature_names = None
        self.is_fitted = False

        # Extract configuration
        self.numeric_strategy = self.feature_config.get("numeric", {})
        self.categorical_strategy = self.feature_config.get("categorical", {})
        self.feature_selection = self.feature_config.get("selection", {})
        self.output_path = self.feature_config.get("output_path", "artifacts/features")

        # Ensure output directory exists
        ensure_dir(self.output_path)

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: str = None
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Fit the feature transformer and transform the data.
        
        Args:
            df: Input DataFrame
            target_column: Target column for feature selection (optional)
            
        Returns:
            Tuple of (transformed_data, feature_names)
        """
        logger.info("Fitting and transforming features")

        # Remove target column if present
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column] if target_column else None
        else:
            X = df
            y = None

        # Create and fit transformer
        self.transformer = self._create_transformer(X)

        # Fit and transform
        X_transformed = self.transformer.fit_transform(X)

        # Get feature names
        self.feature_names = self._get_feature_names(X)

        # Apply feature selection if configured
        if self.feature_selection.get("enabled", False):
            X_transformed, self.feature_names = self._apply_feature_selection(
                X_transformed, y, self.feature_names
            )

        # Convert to DataFrame
        X_df = pd.DataFrame(
            X_transformed,
            columns=self.feature_names,
            index=df.index
        )

        self.is_fitted = True

        # Save transformer
        self._save_transformer()

        # Log feature engineering info
        self._log_feature_info(X_df, y)

        return X_df, self.feature_names

    def transform(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """
        Transform new data using fitted transformer.
        
        Args:
            df: Input DataFrame
            target_column: Target column to remove (optional)
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transforming")

        # Remove target column if present
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
        else:
            X = df

        # Transform
        X_transformed = self.transformer.transform(X)

        # Apply feature selection if configured
        if self.feature_selection.get("enabled", False) and hasattr(self, 'feature_selector'):
            X_transformed = self.feature_selector.transform(X_transformed)

        # Convert to DataFrame
        X_df = pd.DataFrame(
            X_transformed,
            columns=self.feature_names,
            index=df.index
        )

        return X_df

    def fit(self, df: pd.DataFrame, target_column: str = None) -> 'FeatureEngineer':
        """
        Fit the feature transformer without transforming.
        
        Args:
            df: Input DataFrame
            target_column: Target column to remove (optional)
            
        Returns:
            Self for chaining
        """
        logger.info("Fitting feature transformer")

        # Remove target column if present
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
        else:
            X = df

        # Create and fit transformer
        self.transformer = self._create_transformer(X)
        self.transformer.fit(X)

        # Get feature names
        self.feature_names = self._get_feature_names(X)

        # Apply feature selection if configured
        if self.feature_selection.get("enabled", False):
            _, self.feature_names = self._apply_feature_selection(
                self.transformer.transform(X), None, self.feature_names
            )

        self.is_fitted = True

        # Save transformer
        self._save_transformer()

        return self

    def _create_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create the ColumnTransformer with configured preprocessing."""
        transformers = []

        # Numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_features:
            numeric_transformer = self._create_numeric_transformer()
            transformers.append(('numeric', numeric_transformer, numeric_features))
            logger.info(f"Added numeric transformer for {len(numeric_features)} features")

        # Categorical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_features:
            categorical_transformer = self._create_categorical_transformer()
            transformers.append(('categorical', categorical_transformer, categorical_features))
            logger.info(f"Added categorical transformer for {len(categorical_features)} features")

        # Handle remaining features (pass through)
        remaining_features = [col for col in X.columns
                            if col not in numeric_features + categorical_features]
        if remaining_features:
            transformers.append(('remaining', 'passthrough', remaining_features))
            logger.info(f"Added passthrough for {len(remaining_features)} remaining features")

        if not transformers:
            raise ValueError("No features found for transformation")

        return ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            sparse_threshold=0.0,
            n_jobs=-1
        )

    def _create_numeric_transformer(self) -> Pipeline:
        """Create numeric feature preprocessing pipeline."""
        steps = []

        # Imputation
        imputation_strategy = self.numeric_strategy.get("imputation", "median")
        if imputation_strategy == "mean":
            imputer = SimpleImputer(strategy="mean")
        elif imputation_strategy == "median":
            imputer = SimpleImputer(strategy="median")
        elif imputation_strategy == "constant":
            constant_value = self.numeric_strategy.get("constant_value", 0)
            imputer = SimpleImputer(strategy="constant", fill_value=constant_value)
        else:
            imputer = SimpleImputer(strategy="median")

        steps.append(('imputer', imputer))

        # Scaling
        scaling_strategy = self.numeric_strategy.get("scaling", "standard")
        if scaling_strategy == "standard":
            scaler = StandardScaler()
        elif scaling_strategy == "robust":
            scaler = RobustScaler()
        elif scaling_strategy == "minmax":
            scaler = MinMaxScaler()
        elif scaling_strategy == "power":
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            scaler = StandardScaler()

        steps.append(('scaler', scaler))

        return Pipeline(steps)

    def _create_categorical_transformer(self) -> Pipeline:
        """Create categorical feature preprocessing pipeline."""
        steps = []

        # Imputation
        imputation_strategy = self.categorical_strategy.get("imputation", "constant")
        if imputation_strategy == "constant":
            constant_value = self.categorical_strategy.get("constant_value", "missing")
            imputer = SimpleImputer(strategy="constant", fill_value=constant_value)
        elif imputation_strategy == "most_frequent":
            imputer = SimpleImputer(strategy="most_frequent")
        else:
            imputer = SimpleImputer(strategy="constant", fill_value="missing")

        steps.append(('imputer', imputer))

        # Encoding
        encoding_strategy = self.categorical_strategy.get("encoding", "onehot")
        if encoding_strategy == "onehot":
            handle_unknown = self.categorical_strategy.get("handle_unknown", "ignore")
            sparse_output = self.categorical_strategy.get("sparse_output", False)
            encoder = OneHotEncoder(
                handle_unknown=handle_unknown,
                sparse_output=sparse_output,
                drop=self.categorical_strategy.get("drop", None)
            )
        elif encoding_strategy == "label":
            encoder = LabelEncoder()
        else:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        steps.append(('encoder', encoder))

        return Pipeline(steps)

    def _apply_feature_selection(
        self,
        X: np.ndarray,
        y: pd.Series | None,
        feature_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Apply feature selection if configured."""
        if not self.feature_selection.get("enabled", False):
            return X, feature_names

        selection_method = self.feature_selection.get("method", "kbest")
        k = self.feature_selection.get("k", min(100, X.shape[1]))

        if selection_method == "kbest":
            if y is not None:
                # Determine scoring function based on task type
                if y.dtype in ['object', 'category']:
                    score_func = f_classif
                else:
                    score_func = f_regression

                selector = SelectKBest(score_func=score_func, k=k)
                X_selected = selector.fit_transform(X, y)

                # Get selected feature names
                selected_indices = selector.get_support()
                selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_indices[i]]

                # Store selector for later use
                self.feature_selector = selector

                logger.info(f"Applied k-best feature selection: {len(selected_features)} features selected from {len(feature_names)}")

                return X_selected, selected_features

        # If no selection applied, return original
        return X, feature_names

    def _get_feature_names(self, X: pd.DataFrame) -> list[str]:
        """Extract feature names from the fitted transformer."""
        if not self.transformer:
            return []

        feature_names = []

        for name, trans, columns in self.transformer.transformers_:
            if name == "remaining":
                feature_names.extend(columns)
            elif hasattr(trans, 'get_feature_names_out'):
                # For sklearn transformers with get_feature_names_out
                if hasattr(trans, 'named_steps') and 'encoder' in trans.named_steps:
                    # Handle OneHotEncoder specifically
                    encoder = trans.named_steps['encoder']
                    if hasattr(encoder, 'get_feature_names_out'):
                        feature_names.extend(encoder.get_feature_names_out(columns))
                    else:
                        # Fallback for older sklearn versions
                        feature_names.extend([f"{col}_{i}" for col in columns for i in range(trans.transform(X[columns]).shape[1])])
                else:
                    feature_names.extend(trans.get_feature_names_out(columns))
            else:
                # Fallback for transformers without feature names
                if hasattr(trans, 'transform'):
                    n_features = trans.transform(X[columns]).shape[1]
                    feature_names.extend([f"{col}_{i}" for col in columns for i in range(n_features)])
                else:
                    feature_names.extend(columns)

        return feature_names

    def _save_transformer(self) -> None:
        """Save the fitted transformer to disk."""
        if not self.transformer:
            return

        try:
            # Save transformer
            transformer_path = Path(self.output_path) / "feature_transformer.pkl"
            save_pickle(self.transformer, transformer_path)

            # Save feature names
            feature_names_path = Path(self.output_path) / "feature_names.json"
            import json
            with open(feature_names_path, 'w') as f:
                json.dump(self.feature_names, f)

            # Save configuration
            config_path = Path(self.output_path) / "feature_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.feature_config, f)

            logger.info(f"Saved feature transformer to {self.output_path}")

        except Exception as e:
            logger.warning(f"Failed to save feature transformer: {e}")

    def load_transformer(self, path: str = None) -> 'FeatureEngineer':
        """Load a previously fitted transformer."""
        if path is None:
            path = self.output_path

        try:
            # Load transformer
            transformer_path = Path(path) / "feature_transformer.pkl"
            self.transformer = load_pickle(transformer_path)

            # Load feature names
            feature_names_path = Path(path) / "feature_names.json"
            import json
            with open(feature_names_path) as f:
                self.feature_names = json.load(f)

            # Load configuration
            config_path = Path(path) / "feature_config.json"
            with open(config_path) as f:
                loaded_config = json.load(f)

            self.is_fitted = True
            logger.info(f"Loaded feature transformer from {path}")

        except Exception as e:
            logger.error(f"Failed to load feature transformer: {e}")
            raise

        return self

    def _log_feature_info(self, X: pd.DataFrame, y: pd.Series | None) -> None:
        """Log feature engineering information to MLflow."""
        try:
            # Log feature engineering parameters
            mlflow.log_params({
                "feature_engineering_numeric_features": len([col for col in X.columns if col.startswith('numeric')]),
                "feature_engineering_categorical_features": len([col for col in X.columns if col.startswith('categorical')]),
                "feature_engineering_total_features": len(X.columns),
                "feature_engineering_numeric_strategy": self.numeric_strategy.get("scaling", "standard"),
                "feature_engineering_categorical_strategy": self.categorical_strategy.get("encoding", "onehot"),
                "feature_engineering_selection_enabled": self.feature_selection.get("enabled", False)
            })

            # Log feature names as artifact
            feature_names_file = Path(self.output_path) / "feature_names.txt"
            with open(feature_names_file, 'w') as f:
                for i, name in enumerate(X.columns):
                    f.write(f"{i}: {name}\n")

            mlflow.log_artifact(str(feature_names_file))

            # Log feature statistics
            if y is not None:
                mlflow.log_metrics({
                    "feature_engineering_features_to_samples_ratio": len(X.columns) / len(X),
                    "feature_engineering_target_correlation_count": len([col for col in X.columns if col.startswith('numeric')])
                })

            logger.info("Successfully logged feature engineering info to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log feature engineering info to MLflow: {e}")

    def get_feature_importance(self, model: BaseEstimator) -> dict[str, float]:
        """Get feature importance from a fitted model."""
        if not self.is_fitted or not self.feature_names:
            return {}

        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, model.feature_importances_, strict=False))
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importance_dict = dict(zip(self.feature_names, np.abs(model.coef_.flatten()), strict=False))
        else:
            logger.warning("Model does not have feature_importances_ or coef_ attribute")
            return {}

        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_importance

    def get_transformer_summary(self) -> dict[str, Any]:
        """Get a summary of the fitted transformer."""
        if not self.is_fitted:
            return {"status": "not_fitted"}

        summary = {
            "status": "fitted",
            "total_features": len(self.feature_names) if self.feature_names else 0,
            "numeric_strategy": self.numeric_strategy.get("scaling", "standard"),
            "categorical_strategy": self.categorical_strategy.get("encoding", "onehot"),
            "feature_selection_enabled": self.feature_selection.get("enabled", False),
            "output_path": self.output_path
        }

        if self.feature_names:
            summary["feature_names"] = self.feature_names[:10]  # First 10 features
            if len(self.feature_names) > 10:
                summary["feature_names"].append(f"... and {len(self.feature_names) - 10} more")

        return summary

    def reset(self) -> None:
        """Reset the feature engineer to unfitted state."""
        self.transformer = None
        self.feature_names = None
        self.is_fitted = False
        if hasattr(self, 'feature_selector'):
            delattr(self, 'feature_selector')

        logger.info("Feature engineer reset to unfitted state")


class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom feature transformer for advanced preprocessing.
    
    Can be used to add custom preprocessing steps to the feature engineering pipeline.
    """

    def __init__(self, custom_function=None):
        """
        Initialize custom transformer.
        
        Args:
            custom_function: Custom function to apply to features
        """
        self.custom_function = custom_function

    def fit(self, X, y=None):
        """Fit the transformer."""
        return self

    def transform(self, X):
        """Transform the features."""
        if self.custom_function:
            return self.custom_function(X)
        return X

    def get_feature_names_out(self, feature_names_in):
        """Get output feature names."""
        return feature_names_in
