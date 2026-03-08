"""
DFMGAN Trainer - Stable WGAN-GP version
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import logging

from .generator import ConditionalMCGenerator, TimeGANDiscriminator, DFMGANConfig

logger = logging.getLogger(__name__)


class DFMGANTrainer:
    """Stable trainer using WGAN-GP"""

    def __init__(self, generator, discriminator, config, device="cpu"):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config
        self.device = device

        # WGAN optimizers (better stability)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    def train(self, real_data, epochs=10, batch_size=32, lambda_gp=10.0, n_critic=3):
        """
        WGAN-GP Training
        n_critic: discriminator updates per generator update
        """

        dataset = TensorDataset(real_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        history = {'d_loss': [], 'g_loss': []}

        for epoch in tqdm(range(epochs), desc="Training"):
            d_losses, g_losses = [], []

            for i, (batch,) in enumerate(dataloader):
                real_batch = batch.to(self.device)
                current_batch_size = real_batch.size(0)

                # =========================
                # Train Discriminator
                # =========================
                for _ in range(n_critic):
                    self.d_optimizer.zero_grad()

                    fake_data = self.generator(
                        real_batch[:, :min(30, real_batch.size(1)), :],
                        current_batch_size
                    ).detach()

                    fake_data = self._match_sequence_length(fake_data, real_batch)

                    real_scores = self.discriminator(real_batch)
                    fake_scores = self.discriminator(fake_data)

                    # WGAN loss
                    d_loss = -(real_scores.mean() - fake_scores.mean())

                    # Gradient penalty
                    gp = self._gradient_penalty(real_batch, fake_data)
                    d_loss = d_loss + lambda_gp * gp

                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    self.d_optimizer.step()

                # =========================
                # Train Generator
                # =========================
                self.g_optimizer.zero_grad()

                fake_data = self.generator(
                    real_batch[:, :min(30, real_batch.size(1)), :],
                    current_batch_size
                )

                fake_data = self._match_sequence_length(fake_data, real_batch)

                fake_scores = self.discriminator(fake_data)

                g_loss = -fake_scores.mean()

                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.g_optimizer.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            history['d_loss'].append(np.mean(d_losses))
            history['g_loss'].append(np.mean(g_losses))

            logger.info(f"Epoch {epoch+1}: D={history['d_loss'][-1]:.4f}, G={history['g_loss'][-1]:.4f}")

        return history

    # =========================
    # Helpers
    # =========================
    def _match_sequence_length(self, fake, real):
        """Ensure fake and real sequences match"""
        if fake.size(1) != real.size(1):
            if fake.size(1) < real.size(1):
                pad = fake[:, -1:, :].repeat(1, real.size(1) - fake.size(1), 1)
                fake = torch.cat([fake, pad], dim=1)
            else:
                fake = fake[:, :real.size(1), :]
        return fake

    def _gradient_penalty(self, real_data, fake_data):
        """WGAN-GP gradient penalty"""

        batch_size = real_data.size(0)

        alpha = torch.rand(batch_size, 1, 1, device=self.device)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        scores = self.discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)

        gp = ((grad_norm - 1) ** 2).mean()

        return gp
