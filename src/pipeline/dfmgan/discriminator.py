"""
TimeGAN Discriminator for path realism scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeGANDiscriminator(nn.Module):
    """
    Discriminator that distinguishes real from generated paths
    Used for filtering most realistic paths
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, seq_len: int = 252):
        super().__init__()
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Temporal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=seq_len,
            num_heads=4,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + seq_len, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        
        Returns:
            scores: (batch_size, 1) realism scores
        """
        # Transpose for conv: (batch, input_dim, seq_len)
        x_conv = x.transpose(1, 2)
        conv_features = self.conv_layers(x_conv).squeeze(-1)
        
        # Attention over time dimension
        x_attn, _ = self.attention(x, x, x)
        attn_features = x_attn.mean(dim=1)
        
        # Combine features
        combined = torch.cat([conv_features, attn_features], dim=1)
        
        # Classify
        scores = self.classifier(combined)
        
        return scores
    
    def compute_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP
        
        Args:
            real_data: Real data samples
            fake_data: Generated data samples
        
        Returns:
            gradient_penalty: GP value
        """
        batch_size = real_data.size(0)
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, device=real_data.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Get critic scores
        scores = self.forward(interpolated)
        
        # Compute gradients
        grad = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute gradient penalty
        grad = grad.view(batch_size, -1)
        grad_norm = grad.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        
        return gp