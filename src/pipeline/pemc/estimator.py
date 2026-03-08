"""
PEMC - Prediction-Enhanced Monte Carlo (Production Ready)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Tuple
from dataclasses import dataclass
from scipy import stats


# =========================
# CONFIG
# =========================
@dataclass
class PEMCConfig:
    n_coupled: int = 1000
    n_independent: int = 100000
    confidence_level: float = 0.95
    batch_size: int = 4096


# =========================
# MODEL
# =========================
class NeuralPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# =========================
# ESTIMATOR
# =========================
class PEMCEstimator:

    def __init__(self, predictor: NeuralPredictor, config: PEMCConfig):
        self.predictor = predictor
        self.config = config
        self.predictor.eval()

        # auto-detect device
        self.device = next(predictor.parameters()).device

    # -------------------------
    # Batch Prediction
    # -------------------------
    def _predict_in_batches(self, features: np.ndarray):
        preds = []
        bs = self.config.batch_size

        with torch.no_grad():
            for i in range(0, len(features), bs):
                batch = torch.tensor(
                    features[i:i+bs], dtype=torch.float32
                ).to(self.device)

                out = self.predictor(batch).cpu().numpy()
                preds.append(out)

        preds = np.concatenate(preds)

        # Clean NaNs
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

        return preds

    # -------------------------
    # Main Estimation
    # -------------------------
    def estimate(
        self,
        expensive_sampler: Callable,
        cheap_sampler: Callable
    ) -> Tuple[float, float, dict]:

        # =========================
        # 1. Coupled Samples
        # =========================
        coupled_payoffs, coupled_features = expensive_sampler(self.config.n_coupled)

        coupled_payoffs = np.asarray(coupled_payoffs)
        coupled_features = np.asarray(coupled_features)

        pred_coupled = self._predict_in_batches(coupled_features)

        # Clean invalid values
        mask = np.isfinite(coupled_payoffs) & np.isfinite(pred_coupled)
        coupled_payoffs = coupled_payoffs[mask]
        pred_coupled = pred_coupled[mask]

        if len(coupled_payoffs) == 0:
            raise ValueError("No valid coupled samples after filtering.")

        # =========================
        # 2. Optimal Beta (fast)
        # =========================
        mean_f = np.mean(coupled_payoffs)
        mean_g = np.mean(pred_coupled)

        cov_fg = np.mean((coupled_payoffs - mean_f) * (pred_coupled - mean_g))
        var_g = np.var(pred_coupled, ddof=1)

        beta = cov_fg / (var_g + 1e-8)

        # =========================
        # 3. Correction Term
        # =========================
        correction = np.mean(coupled_payoffs - beta * pred_coupled)

        # =========================
        # 4. Independent Samples
        # =========================
        _, independent_features = cheap_sampler(self.config.n_independent)

        independent_features = np.asarray(independent_features)

        pred_independent = self._predict_in_batches(independent_features)

        # =========================
        # 5. Final Estimate
        # =========================
        estimate = correction + beta * np.mean(pred_independent)

        # =========================
        # 6. Variance
        # =========================
        var_coupled = np.var(
            coupled_payoffs - beta * pred_coupled, ddof=1
        ) / len(coupled_payoffs)

        var_independent = np.var(
            beta * pred_independent, ddof=1
        ) / len(pred_independent)

        variance = var_coupled + var_independent

        # Standard MC
        var_standard = np.var(coupled_payoffs, ddof=1) / len(coupled_payoffs)

        reduction = (var_standard - variance) / (var_standard + 1e-8)

        # =========================
        # 7. Confidence Interval
        # =========================
        z_score = stats.norm.ppf(
            1 - (1 - self.config.confidence_level) / 2
        )

        ci_width = z_score * np.sqrt(variance)

        metrics = {
            'estimate': float(estimate),
            'variance': float(variance),
            'variance_reduction': float(reduction),
            'beta': float(beta),
            'ci_lower': float(estimate - ci_width),
            'ci_upper': float(estimate + ci_width),
            'correction_term': float(correction),
            'predictor_mean': float(np.mean(pred_independent)),
            'samples_used': {
                'coupled': int(len(coupled_payoffs)),
                'independent': int(len(pred_independent))
            }
        }

        return float(estimate), float(variance), metrics
