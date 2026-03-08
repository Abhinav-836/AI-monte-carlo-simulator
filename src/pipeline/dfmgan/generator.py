"""
DFMGAN - Discriminator-Filtered Monte Carlo GAN
Fully fixed version (stable, memory-safe, production ready)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass


# =========================
# CONFIG
# =========================
@dataclass
class DFMGANConfig:
    n_assets: int = 3
    seq_len: int = 60
    latent_dim: int = 32
    hidden_dim: int = 64
    n_simulations: int = 1000
    filter_top_k: float = 0.1
    batch_size_paths: int = 16   # increased slightly


# =========================
# GENERATOR
# =========================
class ConditionalMCGenerator(nn.Module):
    """Generator with stable sequence handling"""

    def __init__(self, config: DFMGANConfig):
        super().__init__()
        self.config = config
        self.actual_n_assets = None

        self.input_proj = None
        self.output_proj = None

        self.lstm = nn.LSTM(
            input_size=config.latent_dim,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.register_buffer('hist_mean', torch.zeros(1))
        self.register_buffer('hist_std', torch.ones(1))

    def _build_adaptive_layers(self, n_assets, device):
        """Build layers dynamically"""
        self.actual_n_assets = n_assets

        self.input_proj = nn.Linear(n_assets, self.config.latent_dim).to(device)
        self.output_proj = nn.Linear(self.config.hidden_dim, n_assets).to(device)

        self.hist_mean = torch.zeros(n_assets, device=device)
        self.hist_std = torch.ones(n_assets, device=device)

    def forward(self, historical_data: torch.Tensor, n_paths: Optional[int] = None) -> torch.Tensor:
        """Generate realistic Monte Carlo paths"""

        if n_paths is None:
            n_paths = min(self.config.n_simulations, 500)

        n_paths = min(n_paths, 500)

        B, T, A = historical_data.shape
        device = historical_data.device

        # Build layers dynamically
        if self.actual_n_assets != A:
            self._build_adaptive_layers(A, device)

        # Compute statistics safely
        with torch.no_grad():
            self.hist_mean = historical_data.mean(dim=(0, 1))
            self.hist_std = historical_data.std(dim=(0, 1)).clamp(min=1e-6)

        # Fix sequence length
        target_T = self.config.seq_len

        if T > target_T:
            historical_data = historical_data[:, -target_T:, :]
        elif T < target_T:
            pad_size = target_T - T
            last_val = historical_data[:, -1:, :].repeat(1, pad_size, 1)
            historical_data = torch.cat([historical_data, last_val], dim=1)

        # Normalize
        hist_norm = (historical_data - self.hist_mean) / self.hist_std

        # Project
        hist_latent = self.input_proj(hist_norm)

        all_paths = []
        batch_size = min(self.config.batch_size_paths, n_paths)

        for i in range(0, n_paths, batch_size):
            current_batch = min(batch_size, n_paths - i)

            noise = torch.randn(
                current_batch,
                hist_latent.size(1),
                hist_latent.size(2),
                device=device
            ) * 0.05

            # Repeat base sequence for batch
            base = hist_latent[:1].repeat(current_batch, 1, 1)

            noisy_input = base + noise

            lstm_out, _ = self.lstm(noisy_input)

            path_norm = self.output_proj(lstm_out)

            # Denormalize
            path = path_norm * self.hist_std + self.hist_mean

            # Ensure positivity (financial prices)
            path = torch.clamp(path, min=1e-3)

            all_paths.append(path)

        return torch.cat(all_paths, dim=0)


# =========================
# DISCRIMINATOR
# =========================
class TimeGANDiscriminator(nn.Module):
    """Stable discriminator"""

    def __init__(self, config: DFMGANConfig):
        super().__init__()
        self.config = config
        self.actual_n_assets = None

        self.conv = None
        self.fc = None

        self.pool = nn.AdaptiveAvgPool1d(1)

    def _build_adaptive_layers(self, n_assets, device):
        self.actual_n_assets = n_assets

        self.conv = nn.Conv1d(n_assets, 16, kernel_size=3, padding=1).to(device)
        self.fc = nn.Linear(16, 1).to(device)

    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        B, T, A = paths.shape
        device = paths.device

        if self.actual_n_assets != A:
            self._build_adaptive_layers(A, device)

        x = paths.transpose(1, 2)  # (B, A, T)

        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)

        logits = self.fc(x)

        return torch.sigmoid(logits)


# =========================
# PATH FILTER
# =========================
class PathFilter:
    """Filters top realistic paths"""

    def __init__(self, discriminator: TimeGANDiscriminator):
        self.discriminator = discriminator
        self.discriminator.eval()

    def filter_paths(
        self,
        generated_paths: torch.Tensor,
        top_k: float = 0.1,
        batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        n_paths = generated_paths.size(0)
        scores_list = []

        with torch.no_grad():
            for i in range(0, n_paths, batch_size):
                batch = generated_paths[i:i + batch_size]
                scores = self.discriminator(batch)
                scores_list.append(scores.squeeze().cpu())

        scores = torch.cat(scores_list)

        k = max(1, int(n_paths * top_k))

        top_indices = torch.topk(scores, k=k).indices

        return generated_paths[top_indices], scores
