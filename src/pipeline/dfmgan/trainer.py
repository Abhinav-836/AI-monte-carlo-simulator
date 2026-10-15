"""
DFMGAN Trainer - WGAN-GP
"""

import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DFMGANTrainer:
    def __init__(self, generator, discriminator, config, device="cpu"):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config
        self.device = device
        # layers now exist at construction, so the optimizers see ALL parameters
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    def train(self, real_data, epochs=10, batch_size=32, lambda_gp=10.0, n_critic=3):
        real_data = torch.as_tensor(real_data, dtype=torch.float32)
        batch_size = min(batch_size, len(real_data))
        loader = DataLoader(TensorDataset(real_data), batch_size=batch_size, shuffle=True, drop_last=True)
        if len(loader) == 0:
            raise ValueError("Not enough real sequences to form a batch")

        history = {"d_loss": [], "g_loss": []}
        for epoch in tqdm(range(epochs), desc="Training"):
            d_losses, g_losses = [], []
            for (batch,) in loader:
                real = batch.to(self.device)
                bsz = real.size(0)
                cond = real[:, : min(30, real.size(1)), :]

                for _ in range(n_critic):
                    self.d_optimizer.zero_grad()
                    fake = self._match_len(self.generator(cond, bsz).detach(), real)
                    d_loss = -(self.discriminator(real).mean() - self.discriminator(fake).mean())
                    d_loss = d_loss + lambda_gp * self._gradient_penalty(real, fake)
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    self.d_optimizer.step()

                self.g_optimizer.zero_grad()
                fake = self._match_len(self.generator(cond, bsz), real)
                g_loss = -self.discriminator(fake).mean()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.g_optimizer.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            history["d_loss"].append(float(np.mean(d_losses)))
            history["g_loss"].append(float(np.mean(g_losses)))
            logger.info(f"Epoch {epoch + 1}: D={history['d_loss'][-1]:.4f}, G={history['g_loss'][-1]:.4f}")
        return history

    def save(self, path: str):
        """Checkpoint format expected by MonteCarloPipeline.load_models(checkpoint_path=...)."""
        torch.save({"generator": self.generator.state_dict(),
                    "discriminator": self.discriminator.state_dict()}, path)

    @staticmethod
    def _match_len(fake, real):
        if fake.size(1) < real.size(1):
            pad = fake[:, -1:, :].repeat(1, real.size(1) - fake.size(1), 1)
            return torch.cat([fake, pad], dim=1)
        return fake[:, : real.size(1), :]

    def _gradient_penalty(self, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, device=self.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        scores = self.discriminator(interp)
        grads = torch.autograd.grad(scores, interp, torch.ones_like(scores),
                                    create_graph=True, retain_graph=True)[0]
        return ((grads.reshape(real.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()