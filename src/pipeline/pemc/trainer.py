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
        # (verbose=... was removed: no longer accepted by recent PyTorch)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.criterion = nn.MSELoss()

    def train(self, features: np.ndarray, payoffs: np.ndarray, validation_split: float = 0.2,
              epochs: int = 100, batch_size: int = 64, early_stopping_patience: int = 10) -> Dict:
        idx = np.random.permutation(len(features))
        features, payoffs = np.asarray(features)[idx], np.asarray(payoffs)[idx]
        n_train = int(len(features) * (1 - validation_split))

        # targets are 1-D (B,) to match the predictor output; (B,1) vs (B,) made MSE broadcast to (B,B)
        X_train = torch.as_tensor(features[:n_train], dtype=torch.float32)
        y_train = torch.as_tensor(payoffs[:n_train], dtype=torch.float32).view(-1)
        X_val = torch.as_tensor(features[n_train:], dtype=torch.float32).to(self.device)
        y_val = torch.as_tensor(payoffs[n_train:], dtype=torch.float32).view(-1).to(self.device)

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

            self.predictor.eval()
            with torch.no_grad():
                val_loss = self.criterion(self.predictor(X_val), y_val).item()
            self.scheduler.step(val_loss)

            history["train_loss"].append(float(np.mean(losses)))
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
        # plain dicts only: torch.load(weights_only=True) (the default in new PyTorch) rejects dataclasses
        torch.save({
            "model_state_dict": self.predictor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
        }, path)

    def load_predictor(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.predictor.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])