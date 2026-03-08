"""
Path filtering using discriminator scores
"""

import torch
import numpy as np
from typing import Tuple, Optional

class PathFilter:
    """
    Filters generated paths to keep only the most realistic ones
    Uses discriminator scores to select top K% paths
    """
    
    def __init__(self, discriminator, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.discriminator = discriminator
        self.device = device
        self.discriminator.eval()
        
    def filter_paths(
        self,
        generated_paths: torch.Tensor,
        top_k: float = 0.01,
        batch_size: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top K% most realistic paths
        
        Args:
            generated_paths: Shape (n_paths, seq_len, n_assets)
            top_k: Fraction to keep (e.g., 0.01 = top 1%)
            batch_size: Batch size for scoring
            
        Returns:
            filtered_paths: Top K% paths
            scores: Realism scores for all paths
        """
        n_paths = generated_paths.size(0)
        all_scores = []
        
        # Score in batches
        with torch.no_grad():
            for i in range(0, n_paths, batch_size):
                batch = generated_paths[i:i+batch_size].to(self.device)
                batch_scores = self.discriminator(batch)
                all_scores.append(batch_scores.cpu())
        
        scores = torch.cat(all_scores).flatten()
        
        # Sort by score (higher = more realistic)
        sorted_indices = torch.argsort(scores, descending=True)
        
        # Select top K%
        k = max(1, int(n_paths * top_k))
        top_indices = sorted_indices[:k]
        
        filtered = generated_paths[top_indices]
        
        return filtered, scores
    
    def filter_with_diversity(
        self,
        generated_paths: torch.Tensor,
        top_k: float = 0.01,
        diversity_weight: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select paths balancing realism and diversity
        
        Args:
            generated_paths: Shape (n_paths, seq_len, n_assets)
            top_k: Fraction to keep
            diversity_weight: Weight for diversity vs realism
            
        Returns:
            filtered_paths: Selected paths
            scores: Combined scores
        """
        n_paths = generated_paths.size(0)
        
        # Get realism scores
        with torch.no_grad():
            scores = self.discriminator(generated_paths.to(self.device)).cpu().flatten()
        
        # Calculate diversity (pairwise distances)
        paths_flat = generated_paths.view(n_paths, -1)
        distances = torch.cdist(paths_flat, paths_flat, p=2)
        diversity_scores = distances.mean(dim=1)
        
        # Normalize scores
        realism_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        diversity_norm = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-8)
        
        # Combine
        combined = (1 - diversity_weight) * realism_norm + diversity_weight * diversity_norm
        
        # Select top K%
        k = max(1, int(n_paths * top_k))
        top_indices = torch.argsort(combined, descending=True)[:k]
        
        filtered = generated_paths[top_indices]
        
        return filtered, combined
    
    def compute_path_statistics(self, paths: torch.Tensor) -> dict:
        """
        Compute statistics for filtered paths
        
        Args:
            paths: Generated paths (n_paths, seq_len, n_assets)
        
        Returns:
            Dictionary of statistics
        """
        with torch.no_grad():
            n_paths, seq_len, n_assets = paths.shape
            
            # Final prices
            final_prices = paths[:, -1, :]
            
            # Returns
            returns = torch.diff(torch.log(paths + 1e-8), dim=1)
            
            statistics = {
                'mean_final_price': final_prices.mean(dim=0).tolist(),
                'std_final_price': final_prices.std(dim=0).tolist(),
                'mean_return': returns.mean(dim=(0, 1)).tolist(),
                'volatility': returns.std(dim=(0, 1)).tolist(),
                'max_price': paths.max(dim=1)[0].max(dim=0)[0].tolist(),
                'min_price': paths.min(dim=1)[0].min(dim=0)[0].tolist(),
                'correlation': torch.corrcoef(final_prices.T).tolist()
            }
            
        return statistics