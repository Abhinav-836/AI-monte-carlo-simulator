"""
PEMC Neural Predictor Trainer
"""

import copy
import logging
from dataclasses import asdict
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .estimator import NeuralPredictor, PEMCConfig

logger = logging.getLogger(__name__)


class PEMCTrainer:
    def __init__(self, predictor: NeuralPredictor, config: PEMCConfig, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = predictor.to(self.device)
        self.config = config
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.criterion = nn.MSELoss()

    @staticmethod
    def _validate(features: np.ndarray, payoffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fail early with a clear message instead of a PyTorch size assertion."""
        features = np.asarray(features, dtype=np.float32)
        payoffs = np.asarray(payoffs, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D (n_samples, n_features), got shape {features.shape}. "
                "Did you swap features and payoffs? Samplers return (payoffs, features).")
        if payoffs.ndim == 2 and payoffs.shape[1] == 1:
            payoffs = payoffs.ravel()  # accept (n, 1)
        if payoffs.ndim != 1:
            raise ValueError(f"payoffs must be 1-D (n_samples,), got shape {payoffs.shape}")
        if len(features) != len(payoffs):
            raise ValueError(f"features has {len(features)} rows but payoffs has {len(payoffs)}")
        if len(features) < 10:
            raise ValueError("need at least 10 samples to train")
        if not (np.isfinite(features).all() and np.isfinite(payoffs).all()):
            raise ValueError("features/payoffs contain NaN or inf")
        return features, payoffs

    def train(self, features: np.ndarray, payoffs: np.ndarray, validation_split: float = 0.2,
              epochs: int = 100, batch_size: int = 64, early_stopping_patience: int = 10) -> Dict:
        features, payoffs = self._validate(features, payoffs)

        idx = np.random.permutation(len(features))
        features, payoffs = features[idx], payoffs[idx]
        n_train = max(1, int(len(features) * (1 - validation_split)))

        # targets are 1-D (B,) to match the predictor output; (B,1) vs (B,) broadcast MSE to (B,B)
        X_train = torch.as_tensor(features[:n_train])
        y_train = torch.as_tensor(payoffs[:n_train])
        X_val = torch.as_tensor(features[n_train:]).to(self.device)
        y_val = torch.as_tensor(payoffs[n_train:]).to(self.device)
        has_val = len(X_val) > 0

        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_val, patience, best_weights = float("inf"), 0, None

        for epoch in tqdm(range(epochs), desc="Training PEMC"):
            self.predictor.train()
            losses = []
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.predictor(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                self.optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))

            self.predictor.eval()
            if has_val:
                with torch.no_grad():
                    val_loss = self.criterion(self.predictor(X_val), y_val).item()
            else:
                val_loss = train_loss
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if val_loss < best_val:
                best_val, patience = val_loss, 0
                best_weights = copy.deepcopy(self.predictor.state_dict())
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_weights:
            self.predictor.load_state_dict(best_weights)
        self.predictor.eval()
        return history

    def generate_training_data(self, sampler: Callable, n_samples: int = 100000,
                               option_params: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calls sampler(n, params) -> (payoffs, features); returns (features, payoffs)."""
        feats, pays = [], []
        remaining, batch = n_samples, 10000
        for _ in tqdm(range(int(np.ceil(n_samples / batch))), desc="Generating data"):
            m = min(batch, remaining)
            p, f = sampler(m, option_params)
            feats.append(f)
            pays.append(p)
            remaining -= m
        return np.vstack(feats), np.concatenate(pays)

    def save_predictor(self, path: str):
        # plain dicts only: torch.load(weights_only=True) rejects dataclasses
        torch.save({
            "model_state_dict": self.predictor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
        }, path)

    def load_predictor(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.predictor.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])