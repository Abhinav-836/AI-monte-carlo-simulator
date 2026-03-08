"""
PEMC Neural Predictor Trainer (Production Ready)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Callable, Dict
from tqdm import tqdm
import logging
import copy

from .estimator import NeuralPredictor, PEMCConfig

logger = logging.getLogger(__name__)


class PEMCTrainer:
    """Trains neural predictors for PEMC control variates"""

    def __init__(
        self,
        predictor: NeuralPredictor,
        config: PEMCConfig,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.predictor = predictor.to(self.device)
        self.config = config

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, verbose=True
        )

        self.criterion = nn.MSELoss()

    # =========================
    # TRAIN
    # =========================
    def train(
        self,
        features: np.ndarray,
        payoffs: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 10
    ) -> Dict:

        # 🔥 Shuffle to prevent data leakage
        indices = np.random.permutation(len(features))
        features = features[indices]
        payoffs = payoffs[indices]

        n_train = int(len(features) * (1 - validation_split))

        X_train = torch.FloatTensor(features[:n_train])
        y_train = torch.FloatTensor(payoffs[:n_train]).view(-1, 1)

        X_val = torch.FloatTensor(features[n_train:])
        y_val = torch.FloatTensor(payoffs[n_train:]).view(-1, 1)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )

        history = {'train_loss': [], 'val_loss': [], 'lr': []}

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in tqdm(range(epochs), desc="Training PEMC"):

            # ---- TRAIN ----
            self.predictor.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()

                preds = self.predictor(batch_X)
                loss = self.criterion(preds, batch_y)

                loss.backward()

                # 🔥 Gradient clipping (important)
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)

                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # ---- VALIDATION ----
            self.predictor.eval()
            with torch.no_grad():
                X_val_device = X_val.to(self.device)
                y_val_device = y_val.to(self.device)

                val_preds = self.predictor(X_val_device)
                val_loss = self.criterion(val_preds, y_val_device).item()

            # Scheduler step
            self.scheduler.step(val_loss)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # ---- EARLY STOPPING ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = copy.deepcopy(self.predictor.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}"
                )

        # Restore best model
        if best_weights:
            self.predictor.load_state_dict(best_weights)

        return history

    # =========================
    # DATA GENERATION
    # =========================
    def generate_training_data(
        self,
        sampler: Callable,
        n_samples: int = 100000,
        option_params: dict = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        logger.info(f"Generating {n_samples} samples...")

        features_list = []
        payoffs_list = []

        batch_size = 10000
        n_batches = int(np.ceil(n_samples / batch_size))

        for _ in tqdm(range(n_batches), desc="Generating data"):
            current_batch = min(batch_size, n_samples)

            payoffs, features = sampler(current_batch, option_params)

            features_list.append(features)
            payoffs_list.append(payoffs)

            n_samples -= current_batch

        features = np.vstack(features_list)
        payoffs = np.concatenate(payoffs_list)

        logger.info(f"Generated dataset: {features.shape}")

        return features, payoffs

    # =========================
    # SAVE / LOAD
    # =========================
    def save_predictor(self, path: str):
        torch.save({
            'model_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

        logger.info(f"Saved model → {path}")

    def load_predictor(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.predictor.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded model ← {path}")
