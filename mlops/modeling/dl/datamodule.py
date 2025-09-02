"""
PyTorch DataModule for tabular data in MLOps project.

Provides efficient data loading and preprocessing for tabular datasets
with mixed numeric and categorical features, including class weights for imbalance.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.dataset import Subset

from ...utils.logging import get_logger

logger = get_logger(__name__)


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data with mixed numeric and categorical features.
    
    Handles both numeric and categorical features, providing proper tensor conversion
    and handling of missing values.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        numeric_features: list[str] = None,
        categorical_features: list[str] = None,
        target_encoder: Any = None,
        feature_encoder: Any = None
    ):
        """
        Initialize the tabular dataset.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            numeric_features: List of numeric feature column names
            categorical_features: List of categorical feature column names
            target_encoder: Optional target encoder (e.g., LabelEncoder)
            feature_encoder: Optional feature encoder (e.g., OneHotEncoder)
        """
        self.data = data.copy()
        self.target_column = target_column

        # Auto-detect feature types if not provided
        if numeric_features is None:
            self.numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in self.numeric_features:
                self.numeric_features.remove(target_column)
        else:
            self.numeric_features = numeric_features

        if categorical_features is None:
            self.categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if target_column in self.categorical_features:
                self.categorical_features.remove(target_column)
        else:
            self.categorical_features = categorical_features

        self.target_encoder = target_encoder
        self.feature_encoder = feature_encoder

        # Prepare features and target
        self._prepare_features()
        self._prepare_target()

        logger.info(f"Initialized TabularDataset with {len(self.numeric_features)} numeric and {len(self.categorical_features)} categorical features")

    def _prepare_features(self):
        """Prepare feature tensors."""
        # Handle numeric features
        if self.numeric_features:
            numeric_data = self.data[self.numeric_features].fillna(0).values
            self.numeric_tensor = torch.FloatTensor(numeric_data)
        else:
            self.numeric_tensor = torch.FloatTensor([])

        # Handle categorical features
        if self.categorical_features:
            if self.feature_encoder:
                # Use pre-fitted encoder
                categorical_data = self.feature_encoder.transform(self.data[self.categorical_features])
                self.categorical_tensor = torch.FloatTensor(categorical_data)
            else:
                # Simple label encoding
                categorical_data = self.data[self.categorical_features].fillna('missing')
                for col in self.categorical_features:
                    categorical_data[col] = categorical_data[col].astype('category').cat.codes
                self.categorical_tensor = torch.LongTensor(categorical_data.values)
        else:
            self.categorical_tensor = torch.LongTensor([])

    def _prepare_target(self):
        """Prepare target tensor."""
        target_data = self.data[self.target_column]

        if self.target_encoder:
            # Use pre-fitted encoder
            target_encoded = self.target_encoder.transform(target_data)
            self.target_tensor = torch.LongTensor(target_encoded)
        else:
            # Auto-detect target type
            if target_data.dtype in ['object', 'category']:
                # Classification: encode categories
                target_encoded = target_data.astype('category').cat.codes
                self.target_tensor = torch.LongTensor(target_encoded)
            else:
                # Regression: keep as float
                self.target_tensor = torch.FloatTensor(target_data.fillna(0).values)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, target)
        """
        features = []

        # Add numeric features
        if len(self.numeric_tensor) > 0:
            features.append(self.numeric_tensor[idx])

        # Add categorical features
        if len(self.categorical_tensor) > 0:
            features.append(self.categorical_tensor[idx])

        # Concatenate all features
        if len(features) > 1:
            # Mixed features: concatenate numeric and categorical
            if self.numeric_features and self.categorical_features:
                # Convert categorical to one-hot if needed
                if self.feature_encoder is None:
                    # Simple one-hot encoding for categorical features
                    cat_features = F.one_hot(self.categorical_tensor[idx], num_classes=len(self.categorical_features))
                    cat_features = cat_features.float()
                    features = [self.numeric_tensor[idx], cat_features]

                # Concatenate along feature dimension
                combined_features = torch.cat(features, dim=0)
            else:
                combined_features = torch.cat(features, dim=0)
        else:
            combined_features = features[0]

        return combined_features, self.target_tensor[idx]

    def get_feature_dim(self) -> int:
        """Get the total feature dimension."""
        total_dim = 0

        if len(self.numeric_tensor) > 0:
            total_dim += self.numeric_tensor.shape[1]

        if len(self.categorical_tensor) > 0:
            if self.feature_encoder:
                total_dim += self.categorical_tensor.shape[1]
            else:
                # For simple label encoding, we'll convert to one-hot
                total_dim += len(self.categorical_features)

        return total_dim

    def get_num_classes(self) -> int:
        """Get the number of classes for classification tasks."""
        if self.target_tensor.dtype == torch.long:
            return len(torch.unique(self.target_tensor))
        else:
            return 1  # Regression task

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced classification."""
        if self.target_tensor.dtype != torch.long:
            return torch.ones(1)  # No class weights for regression

        # Calculate class counts
        unique, counts = torch.unique(self.target_tensor, return_counts=True)
        total_samples = len(self.target_tensor)

        # Calculate weights (inverse frequency)
        weights = total_samples / (len(unique) * counts.float())

        # Normalize weights
        weights = weights / weights.sum() * len(unique)

        return weights


class TabularDataModule:
    """
    PyTorch DataModule for tabular data with train/validation/test splits.
    
    Provides efficient data loading, preprocessing, and class weight calculation
    for training deep learning models on tabular data.
    """

    def __init__(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        numeric_features: list[str] = None,
        categorical_features: list[str] = None,
        target_encoder: Any = None,
        feature_encoder: Any = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_class_weights: bool = True,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_test: bool = False
    ):
        """
        Initialize the tabular data module.
        
        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            test_data: Test data DataFrame
            target_column: Name of the target column
            numeric_features: List of numeric feature column names
            categorical_features: List of categorical feature column names
            target_encoder: Optional target encoder (e.g., LabelEncoder)
            feature_encoder: Optional feature encoder (e.g., OneHotEncoder)
            batch_size: Batch size for data loading
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            use_class_weights: Whether to use class weights for imbalanced data
            shuffle_train: Whether to shuffle training data
            shuffle_val: Whether to shuffle validation data
            shuffle_test: Whether to shuffle test data
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_class_weights = use_class_weights
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test

        # Create datasets
        self.train_dataset = TabularDataset(
            train_data, target_column, numeric_features, categorical_features,
            target_encoder, feature_encoder
        )

        self.val_dataset = TabularDataset(
            val_data, target_column, numeric_features, categorical_features,
            target_encoder, feature_encoder
        )

        self.test_dataset = TabularDataset(
            test_data, target_column, numeric_features, categorical_features,
            target_encoder, feature_encoder
        )

        # Calculate class weights if needed
        self.class_weights = None
        if self.use_class_weights and self.train_dataset.get_num_classes() > 1:
            self.class_weights = self.train_dataset.get_class_weights()
            logger.info(f"Calculated class weights: {self.class_weights}")

        # Create samplers
        self._create_samplers()

        logger.info(f"Initialized TabularDataModule with {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test samples")

    def _create_samplers(self):
        """Create data samplers for training."""
        if self.use_class_weights and self.class_weights is not None:
            # Use weighted random sampling for imbalanced data
            sample_weights = self.class_weights[self.train_dataset.target_tensor]
            self.train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            logger.info("Created weighted random sampler for training")
        else:
            self.train_sampler = None

    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=self.shuffle_train if self.train_sampler is None else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_test,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    def get_feature_dim(self) -> int:
        """Get the total feature dimension."""
        return self.train_dataset.get_feature_dim()

    def get_num_classes(self) -> int:
        """Get the number of classes for classification tasks."""
        return self.train_dataset.get_num_classes()

    def get_class_weights(self) -> torch.Tensor | None:
        """Get class weights if available."""
        return self.class_weights

    def get_data_info(self) -> dict[str, Any]:
        """Get information about the data."""
        return {
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "test_samples": len(self.test_dataset),
            "feature_dim": self.get_feature_dim(),
            "num_classes": self.get_num_classes(),
            "batch_size": self.batch_size,
            "use_class_weights": self.use_class_weights,
            "numeric_features": self.train_dataset.numeric_features,
            "categorical_features": self.train_dataset.categorical_features
        }

    def create_subset(self, dataset_name: str, indices: list[int]) -> 'TabularDataModule':
        """
        Create a subset of the data module.
        
        Args:
            dataset_name: Name of the dataset to subset ('train', 'val', 'test')
            indices: Indices to include in the subset
            
        Returns:
            New TabularDataModule with subset data
        """
        if dataset_name == 'train':
            subset_data = Subset(self.train_dataset, indices)
            subset_df = self.train_dataset.data.iloc[indices]
        elif dataset_name == 'val':
            subset_data = Subset(self.val_dataset, indices)
            subset_df = self.val_dataset.data.iloc[indices]
        elif dataset_name == 'test':
            subset_data = Subset(self.test_dataset, indices)
            subset_df = self.test_dataset.data.iloc[indices]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        # Create new data module with subset
        subset_module = TabularDataModule(
            train_data=subset_df,
            val_data=subset_df,  # Use same data for both
            test_data=subset_df,  # Use same data for both
            target_column=self.train_dataset.target_column,
            numeric_features=self.train_dataset.numeric_features,
            categorical_features=self.train_dataset.categorical_features,
            target_encoder=self.train_dataset.target_encoder,
            feature_encoder=self.train_dataset.feature_encoder,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            use_class_weights=self.use_class_weights,
            shuffle_train=self.shuffle_train,
            shuffle_val=self.shuffle_val,
            shuffle_test=self.shuffle_test
        )

        return subset_module

    def save_state(self, path: str):
        """Save the data module state."""
        import pickle

        state = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'use_class_weights': self.use_class_weights,
            'shuffle_train': self.shuffle_train,
            'shuffle_val': self.shuffle_val,
            'shuffle_test': self.shuffle_test,
            'class_weights': self.class_weights,
            'feature_dim': self.get_feature_dim(),
            'num_classes': self.get_num_classes()
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved data module state to {path}")

    def load_state(self, path: str):
        """Load the data module state."""
        import pickle

        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Update attributes
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)

        logger.info(f"Loaded data module state from {path}")


class TabularDataModuleBuilder:
    """
    Builder class for creating TabularDataModule instances with flexible configuration.
    
    Provides a fluent interface for configuring and building data modules.
    """

    def __init__(self):
        """Initialize the builder."""
        self.config = {
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
            'use_class_weights': True,
            'shuffle_train': True,
            'shuffle_val': False,
            'shuffle_test': False
        }
        self.data = {}
        self.features = {}
        self.encoders = {}

    def with_data(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target_column: str) -> 'TabularDataModuleBuilder':
        """Set the data."""
        self.data = {
            'train': train,
            'val': val,
            'test': test,
            'target_column': target_column
        }
        return self

    def with_features(self, numeric: list[str] = None, categorical: list[str] = None) -> 'TabularDataModuleBuilder':
        """Set the feature columns."""
        self.features = {
            'numeric': numeric,
            'categorical': categorical
        }
        return self

    def with_encoders(self, target_encoder: Any = None, feature_encoder: Any = None) -> 'TabularDataModuleBuilder':
        """Set the encoders."""
        self.encoders = {
            'target_encoder': target_encoder,
            'feature_encoder': feature_encoder
        }
        return self

    def with_batch_size(self, batch_size: int) -> 'TabularDataModuleBuilder':
        """Set the batch size."""
        self.config['batch_size'] = batch_size
        return self

    def with_num_workers(self, num_workers: int) -> 'TabularDataModuleBuilder':
        """Set the number of workers."""
        self.config['num_workers'] = num_workers
        return self

    def with_pin_memory(self, pin_memory: bool) -> 'TabularDataModuleBuilder':
        """Set pin memory."""
        self.config['pin_memory'] = pin_memory
        return self

    def with_class_weights(self, use_class_weights: bool) -> 'TabularDataModuleBuilder':
        """Set whether to use class weights."""
        self.config['use_class_weights'] = use_class_weights
        return self

    def with_shuffling(self, train: bool = True, val: bool = False, test: bool = False) -> 'TabularDataModuleBuilder':
        """Set shuffling for different splits."""
        self.config['shuffle_train'] = train
        self.config['shuffle_val'] = val
        self.config['shuffle_test'] = test
        return self

    def build(self) -> TabularDataModule:
        """Build the TabularDataModule."""
        if not self.data:
            raise ValueError("Data must be set before building")

        return TabularDataModule(
            train_data=self.data['train'],
            val_data=self.data['val'],
            test_data=self.data['test'],
            target_column=self.data['target_column'],
            numeric_features=self.features.get('numeric'),
            categorical_features=self.features.get('categorical'),
            target_encoder=self.encoders.get('target_encoder'),
            feature_encoder=self.encoders.get('feature_encoder'),
            **self.config
        )
