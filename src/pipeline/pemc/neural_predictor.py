"""
Production-grade Neural Predictors for PEMC
Stable, fast, and Monte Carlo safe
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


# =========================
# 🔧 UTIL: Enable Dropout Only
# =========================
def enable_mc_dropout(model):
    """Enable dropout during inference without affecting LayerNorm"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# =========================
# 🧠 MAIN PREDICTOR
# =========================
class NeuralPredictor(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim

        # Activation
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU(0.1)
        else:
            act = nn.ReLU()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),   # ✅ FIXED
                act,
                nn.Dropout(dropout_rate)
            ])
            prev = h

        layers.append(nn.Linear(prev, 1))

        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any():
            raise ValueError("Input contains NaNs")

        return self.network(x).squeeze(-1)

    # =========================
    # 🔮 MC DROPOUT
    # =========================
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self.eval()
        enable_mc_dropout(self)

        preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x))

        preds = torch.stack(preds)

        return preds.mean(0), preds.std(0)


# =========================
# 🧠 ENSEMBLE (OPTIMIZED)
# =========================
class EnsemblePredictor(nn.Module):

    def __init__(self, input_dim: int, n_models: int = 5):
        super().__init__()

        self.models = nn.ModuleList([
            NeuralPredictor(input_dim) for _ in range(n_models)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([m(x) for m in self.models])
        return preds.mean(0)

    def predict_with_uncertainty(self, x: torch.Tensor):
        with torch.no_grad():
            preds = torch.stack([m(x) for m in self.models])

        return preds.mean(0), preds.std(0)


# =========================
# 🔁 RESIDUAL MODEL (IMPROVED)
# =========================
class ResidualPredictor(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_blocks: int = 3
    ):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(n_blocks)
        ])

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        out = torch.relu(self.input_layer(x))

        for block in self.blocks:
            out = torch.relu(out + block(out))

        return self.output(out).squeeze(-1)


# =========================
# 🧬 SIMPLE FEATURE EMBEDDING (FIXED)
# =========================
class FeatureEmbedding(nn.Module):
    """
    Lightweight embedding (better than attention for tabular data)
    """

    def __init__(self, input_dim: int, embed_dim: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
