"""
DFMGAN - Discriminator-Filtered Monte Carlo GAN
Layers are built in __init__ (the lazy build happened AFTER the optimizers were created,
so the projection layers and the discriminator were never trained).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class DFMGANConfig:
    n_assets: int = 3
    seq_len: int = 60
    latent_dim: int = 32
    hidden_dim: int = 64
    n_simulations: int = 1000
    filter_top_k: float = 0.1
    batch_size_paths: int = 16
    noise_std: float = 0.05


class ConditionalMCGenerator(nn.Module):
    def __init__(self, config: DFMGANConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.n_assets, config.latent_dim)
        self.lstm = nn.LSTM(config.latent_dim, config.hidden_dim, num_layers=1, batch_first=True)
        self.output_proj = nn.Linear(config.hidden_dim, config.n_assets)

    def forward(self, historical_data: torch.Tensor, n_paths: Optional[int] = None) -> torch.Tensor:
        n_paths = min(n_paths or self.config.n_simulations, 500)
        B, T, A = historical_data.shape
        if A != self.config.n_assets:
            raise ValueError(f"Generator built for {self.config.n_assets} assets, got {A}. "
                             "Rebuild the models with the right n_assets.")
        device = historical_data.device

        with torch.no_grad():
            hist_mean = historical_data.mean(dim=(0, 1))
            hist_std = historical_data.std(dim=(0, 1), unbiased=False).clamp(min=1e-6)

        target_T = self.config.seq_len
        if T > target_T:
            historical_data = historical_data[:, -target_T:, :]
        elif T < target_T:
            pad = historical_data[:, -1:, :].repeat(1, target_T - T, 1)
            historical_data = torch.cat([historical_data, pad], dim=1)

        hist_latent = self.input_proj((historical_data - hist_mean) / hist_std)

        out = []
        step = min(self.config.batch_size_paths, n_paths)
        for i in range(0, n_paths, step):
            cb = min(step, n_paths - i)
            idx = torch.arange(i, i + cb, device=device) % B   # condition on different real windows
            base = hist_latent[idx]
            noisy = base + torch.randn_like(base) * self.config.noise_std
            lstm_out, _ = self.lstm(noisy)
            path = self.output_proj(lstm_out) * hist_std + hist_mean
            out.append(torch.clamp(path, min=1e-3))
        return torch.cat(out, dim=0)


class TimeGANDiscriminator(nn.Module):
    """WGAN critic: raw score, NO sigmoid (a sigmoid breaks the Wasserstein loss / gradient penalty)."""

    def __init__(self, config: DFMGANConfig):
        super().__init__()
        self.config = config
        self.conv = nn.Conv1d(config.n_assets, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        if paths.size(-1) != self.config.n_assets:
            raise ValueError(f"Critic built for {self.config.n_assets} assets, got {paths.size(-1)}")
        # scale-free: compare shapes of paths, not price levels
        rel = paths / (paths[:, :1, :].abs() + 1e-8) - 1.0
        x = torch.relu(self.conv(rel.transpose(1, 2)))
        return self.fc(self.pool(x).squeeze(-1))


class PathFilter:
    def __init__(self, discriminator: TimeGANDiscriminator):
        self.discriminator = discriminator
        self.discriminator.eval()

    def filter_paths(self, generated_paths: torch.Tensor, top_k: float = 0.1,
                     batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = []
        device = next(self.discriminator.parameters()).device
        with torch.no_grad():
            for i in range(0, generated_paths.size(0), batch_size):
                s = self.discriminator(generated_paths[i:i + batch_size].to(device))
                scores.append(s.flatten().cpu())   # flatten (not squeeze): batch of 1 would crash torch.cat
        scores = torch.cat(scores)
        k = max(1, int(generated_paths.size(0) * top_k))
        return generated_paths[torch.topk(scores, k=k).indices], scores