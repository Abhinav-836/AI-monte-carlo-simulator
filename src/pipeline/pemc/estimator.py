"""
PEMC - Prediction-Enhanced Monte Carlo
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


@dataclass
class PEMCConfig:
    n_coupled: int = 1000
    n_independent: int = 100000
    confidence_level: float = 0.95
    batch_size: int = 4096
    sampler_chunk: int = 10000  # simulate independent samples in chunks (100k paths at once OOMs on cloud)


class NeuralPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PEMCEstimator:
    def __init__(self, predictor: nn.Module, config: PEMCConfig):
        self.predictor = predictor
        self.config = config
        self.predictor.eval()
        self.device = next(predictor.parameters()).device

    def _predict_in_batches(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features)
        preds, bs = [], self.config.batch_size
        with torch.no_grad():
            for i in range(0, len(features), bs):
                batch = torch.as_tensor(features[i:i + bs], dtype=torch.float32, device=self.device)
                preds.append(self.predictor(batch).cpu().numpy())
        out = np.concatenate(preds) if preds else np.array([])
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def estimate(self, expensive_sampler: Callable, cheap_sampler: Callable) -> Tuple[float, float, dict]:
        # 1. coupled samples (true payoff + prediction)
        payoffs, feats = expensive_sampler(self.config.n_coupled)
        payoffs = np.asarray(payoffs, float)
        pred_c = self._predict_in_batches(feats)

        mask = np.isfinite(payoffs) & np.isfinite(pred_c)
        payoffs, pred_c = payoffs[mask], pred_c[mask]
        if len(payoffs) < 2:
            raise ValueError("Not enough valid coupled samples after filtering.")

        # 2. optimal beta and correction
        cov_fg = np.mean((payoffs - payoffs.mean()) * (pred_c - pred_c.mean()))
        beta = cov_fg / (np.var(pred_c, ddof=1) + 1e-8)
        correction = np.mean(payoffs - beta * pred_c)

        # 3. independent samples, streamed in chunks (running mean / variance)
        total, s1, s2 = 0, 0.0, 0.0
        remaining = self.config.n_independent
        while remaining > 0:
            m = min(self.config.sampler_chunk, remaining)
            _, f = cheap_sampler(m)
            p = self._predict_in_batches(f)
            total += len(p)
            s1 += float(p.sum())
            s2 += float((p ** 2).sum())
            remaining -= m
        mean_ind = s1 / total
        var_ind_pred = max((s2 - total * mean_ind ** 2) / max(total - 1, 1), 0.0)

        estimate = correction + beta * mean_ind

        # 4. variance
        var_coupled = np.var(payoffs - beta * pred_c, ddof=1) / len(payoffs)
        var_independent = (beta ** 2) * var_ind_pred / total
        variance = var_coupled + var_independent
        var_standard = np.var(payoffs, ddof=1) / len(payoffs)
        reduction = (var_standard - variance) / (var_standard + 1e-12)

        # 5. CI
        z = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
        ci = z * np.sqrt(variance)

        metrics = {
            "estimate": float(estimate), "variance": float(variance),
            "variance_reduction": float(reduction), "beta": float(beta),
            "ci_lower": float(estimate - ci), "ci_upper": float(estimate + ci),
            "correction_term": float(correction), "predictor_mean": float(mean_ind),
            "samples_used": {"coupled": int(len(payoffs)), "independent": int(total)},
        }
        return float(estimate), float(variance), metrics