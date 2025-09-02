"""PyTorch MLP estimator with sklearn-compatible API for tabular data."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from ...utils import get_device, get_logger, set_torch_seed


class MLP(nn.Module):
    """Multi-layer perceptron for tabular data."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.2,
        activation: str = "relu",
        batch_norm: bool = True,
        task_type: str = "classification",
    ):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            activation: Activation function
            batch_norm: Whether to use batch normalization
            task_type: Task type (classification or regression)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.task_type = task_type

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class TorchMLPEstimator(BaseEstimator):
    """PyTorch MLP estimator with sklearn-compatible API."""

    def __init__(
        self,
        hidden_dims: list[int] = [128, 64, 32],
        dropout: float = 0.2,
        activation: str = "relu",
        batch_norm: bool = True,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        patience: int = 10,
        min_delta: float = 0.001,
        amp: bool = True,
        gradient_clip: float = 1.0,
        scheduler: str = "reduce_lr_on_plateau",
        random_state: int = 42,
        device: str | None = None,
        task_type: str = "auto",
        **kwargs
    ):
        """
        Initialize TorchMLPEstimator.
        
        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function
            batch_norm: Whether to use batch normalization
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            amp: Whether to use automatic mixed precision
            gradient_clip: Gradient clipping value
            scheduler: Learning rate scheduler
            random_state: Random seed
            device: Device to use (auto, cuda, cpu)
            task_type: Task type (auto, classification, regression)
            **kwargs: Additional arguments
        """
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.amp = amp
        self.gradient_clip = gradient_clip
        self.scheduler = scheduler
        self.random_state = random_state
        self.device = device
        self.task_type = task_type

        # Set random seed
        set_torch_seed(random_state)

        # Get device
        if device == "auto" or device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        # Initialize model and training components
        self.model = None
        self.optimizer = None
        self.scheduler_obj = None
        self.scaler = None
        self.is_fitted = False

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metric": [],
            "val_metric": [],
        }

        # Best model state
        self.best_model_state = None
        self.best_metric = None

        self.logger = get_logger(__name__)

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        X_val: np.ndarray | pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "TorchMLPEstimator":
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features
            y_val: Validation targets
            sample_weight: Sample weights
            
        Returns:
            self: Fitted estimator
        """
        # Validate inputs
        X, y = check_X_y(X, y, dtype=np.float32)

        # Determine task type if auto
        if self.task_type == "auto":
            if np.issubdtype(y.dtype, np.number):
                unique_values = len(np.unique(y))
                self.task_type = "classification" if unique_values <= 20 else "regression"
            else:
                self.task_type = "classification"

        # Set output dimension
        if self.task_type == "classification":
            if len(np.unique(y)) == 2:
                output_dim = 1  # Binary classification
            else:
                output_dim = len(np.unique(y))  # Multi-class
        else:
            output_dim = 1  # Regression

        # Create model
        input_dim = X.shape[1]
        self.model = MLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
            task_type=self.task_type,
        ).to(self.device)

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Create scheduler
        if self.scheduler == "reduce_lr_on_plateau":
            self.scheduler_obj = ReduceLROnPlateau(
                self.optimizer,
                mode="min" if self.task_type == "regression" else "max",
                factor=0.5,
                patience=self.patience // 2,
                verbose=True,
            )

        # Create scaler for automatic mixed precision
        if self.amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()

        # Prepare data
        train_dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y) if self.task_type == "regression" else torch.LongTensor(y)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Prepare validation data
        if X_val is not None and y_val is not None:
            X_val, y_val = check_array(X_val, dtype=np.float32), np.array(y_val)
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val) if self.task_type == "regression" else torch.LongTensor(y_val)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True if torch.cuda.is_available() else False,
            )
        else:
            val_loader = None

        # Training loop
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Task type: {self.task_type}")
        self.logger.info(f"Output dimension: {output_dim}")

        start_time = time.time()
        best_metric = float('inf') if self.task_type == "regression" else float('-inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            train_loss, train_metric = self._train_epoch(train_loader, sample_weight)

            # Validation
            if val_loader is not None:
                val_loss, val_metric = self._validate_epoch(val_loader)

                # Update scheduler
                if self.scheduler_obj is not None:
                    if self.task_type == "regression":
                        self.scheduler_obj.step(val_loss)
                    else:
                        self.scheduler_obj.step(-val_metric)  # Maximize metric

                # Early stopping
                if self.task_type == "regression":
                    if val_loss < best_metric - self.min_delta:
                        best_metric = val_loss
                        patience_counter = 0
                        self.best_model_state = self.model.state_dict().copy()
                        self.best_metric = best_metric
                    else:
                        patience_counter += 1
                else:
                    if val_metric > best_metric + self.min_delta:
                        best_metric = val_metric
                        patience_counter = 0
                        self.best_model_state = self.model.state_dict().copy()
                        self.best_metric = best_metric
                    else:
                        patience_counter += 1

                # Log progress
                if epoch % 10 == 0 or epoch == self.epochs - 1:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}"
                    )

                # Store history
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["train_metric"].append(train_metric)
                self.history["val_metric"].append(val_metric)

                # Early stopping check
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # No validation data
                if epoch % 10 == 0 or epoch == self.epochs - 1:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}"
                    )

                self.history["train_loss"].append(train_loss)
                self.history["train_metric"].append(train_metric)

        # Restore best model if validation was used
        if val_loader is not None and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(f"Restored best model with metric: {self.best_metric:.4f}")

        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        self.is_fitted = True
        return self

    def _train_epoch(self, train_loader: DataLoader, sample_weight: np.ndarray | None = None) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            if self.amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self._compute_loss(output, target, sample_weight, batch_idx)

                self.scaler.scale(loss).backward()

                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self._compute_loss(output, target, sample_weight, batch_idx)

                loss.backward()

                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()

            # Compute metric
            metric = self._compute_metric(output, target)

            total_loss += loss.item()
            total_metric += metric
            num_batches += 1

        return total_loss / num_batches, total_metric / num_batches

    def _validate_epoch(self, val_loader: DataLoader) -> tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                if self.amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self._compute_loss(output, target)
                else:
                    output = self.model(data)
                    loss = self._compute_loss(output, target)

                metric = self._compute_metric(output, target)

                total_loss += loss.item()
                total_metric += metric
                num_batches += 1

        return total_loss / num_batches, total_metric / num_batches

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor, sample_weight: np.ndarray | None = None, batch_idx: int = 0) -> torch.Tensor:
        """Compute loss."""
        if self.task_type == "classification":
            if output.shape[1] == 1:
                # Binary classification
                loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.float())
            else:
                # Multi-class classification
                loss = F.cross_entropy(output, target)
        else:
            # Regression
            loss = F.mse_loss(output.squeeze(), target.float())

        return loss

    def _compute_metric(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Compute metric for monitoring."""
        if self.task_type == "classification":
            if output.shape[1] == 1:
                # Binary classification
                pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
            else:
                # Multi-class classification
                pred = torch.argmax(output, dim=1)

            # Accuracy
            metric = (pred == target).float().mean().item()
        else:
            # Regression - R² score approximation
            pred = output.squeeze()
            ss_res = torch.sum((target.float() - pred) ** 2)
            ss_tot = torch.sum((target.float() - target.float().mean()) ** 2)
            metric = 1 - (ss_res / (ss_tot + 1e-8))
            metric = metric.item()

        return metric

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        check_is_fitted(self, "is_fitted")
        X = check_array(X, dtype=np.float32)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i + self.batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)

                if self.amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(batch_tensor)
                else:
                    output = self.model(batch_tensor)

                if self.task_type == "classification":
                    if output.shape[1] == 1:
                        # Binary classification
                        pred = (torch.sigmoid(output.squeeze()) > 0.5).float()
                    else:
                        # Multi-class classification
                        pred = torch.argmax(output, dim=1)
                else:
                    # Regression
                    pred = output.squeeze()

                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")

        check_is_fitted(self, "is_fitted")
        X = check_array(X, dtype=np.float32)

        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i + self.batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)

                if self.amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(batch_tensor)
                else:
                    output = self.model(batch_tensor)

                if output.shape[1] == 1:
                    # Binary classification
                    prob = torch.sigmoid(output.squeeze())
                    prob = torch.stack([1 - prob, prob], dim=1)
                else:
                    # Multi-class classification
                    prob = F.softmax(output, dim=1)

                probabilities.append(prob.cpu().numpy())

        return np.concatenate(probabilities)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters."""
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "amp": self.amp,
            "gradient_clip": self.gradient_clip,
            "scheduler": self.scheduler,
            "random_state": self.random_state,
            "device": str(self.device),
            "task_type": self.task_type,
        }

    def set_params(self, **params) -> "TorchMLPEstimator":
        """Set parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def save_model(self, filepath: str | Path) -> None:
        """Save the model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.get_params(),
            "history": self.history,
            "best_metric": self.best_metric,
        }, filepath)

        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str | Path) -> "TorchMLPEstimator":
        """Load the model."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore parameters
        params = checkpoint["model_config"]
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Create and load model
        input_dim = 1  # Will be set during fit
        output_dim = 1 if self.task_type == "regression" else 2

        self.model = MLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
            task_type=self.task_type,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore other attributes
        self.history = checkpoint.get("history", {})
        self.best_metric = checkpoint.get("best_metric", None)
        self.is_fitted = True

        self.logger.info(f"Model loaded from {filepath}")
        return self


# Create sklearn-compatible classifier and regressor
class TorchMLPClassifier(TorchMLPEstimator, ClassifierMixin):
    """PyTorch MLP classifier."""

    def __init__(self, **kwargs):
        super().__init__(task_type="classification", **kwargs)

    def fit(self, X, y, **kwargs):
        return super().fit(X, y, **kwargs)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)


class TorchMLPRegressor(TorchMLPEstimator, RegressorMixin):
    """PyTorch MLP regressor."""

    def __init__(self, **kwargs):
        super().__init__(task_type="regression", **kwargs)

    def fit(self, X, y, **kwargs):
        return super().fit(X, y, **kwargs)

    def predict(self, X):
        return super().predict(X)
