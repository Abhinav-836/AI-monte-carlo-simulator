"""
Variance analysis for PEMC
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-12


@dataclass
class VarianceMetrics:
    mc_variance: float
    pemc_variance: float
    variance_reduction: float
    ci_width_reduction: float
    sample_efficiency: float
    optimal_allocation: Dict[str, float]


class VarianceAnalyzer:
    def __init__(self):
        self.metrics_history = []

    def calculate_variance_reduction(self, mc_estimates: np.ndarray, pemc_estimates: np.ndarray) -> float:
        mc_var = np.var(mc_estimates, ddof=1)
        pemc_var = np.var(pemc_estimates, ddof=1)
        return float(max(0.0, min(1.0, (mc_var - pemc_var) / (mc_var + EPS))))

    def calculate_confidence_interval_width(self, estimates: np.ndarray, confidence: float = 0.95) -> float:
        from scipy import stats
        se = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
        return float(2 * stats.norm.ppf(1 - (1 - confidence) / 2) * se)

    def analyze_variance_reduction(self, mc_estimates: np.ndarray, pemc_estimates: np.ndarray,
                                   confidence: float = 0.95) -> VarianceMetrics:
        mc_var = np.var(mc_estimates, ddof=1)
        pemc_var = np.var(pemc_estimates, ddof=1)
        mc_w = self.calculate_confidence_interval_width(mc_estimates, confidence)
        pemc_w = self.calculate_confidence_interval_width(pemc_estimates, confidence)
        efficiency = mc_var / (pemc_var + EPS)

        n = 1000
        N = int(n * efficiency)
        metrics = VarianceMetrics(
            mc_variance=float(mc_var), pemc_variance=float(pemc_var),
            variance_reduction=float((mc_var - pemc_var) / (mc_var + EPS)),
            ci_width_reduction=float((mc_w - pemc_w) / (mc_w + EPS)),
            sample_efficiency=float(efficiency),
            optimal_allocation={"n_coupled": n, "n_independent": N, "ratio": N / n},
        )
        self.metrics_history.append(metrics)
        return metrics

    def optimal_allocation(self, sigma_f: float, sigma_g: float, rho: float,
                           cost_ratio: float = 10.0, budget: float = 10000.0) -> Dict[str, float]:
        """
        Minimise  sigma_f^2 (1-rho^2)/n + rho^2 sigma_f^2 / N   s.t.  n*cost_ratio + N = budget
        (control-variate term is beta^2 sigma_g^2 = rho^2 sigma_f^2).
        Lagrange solution: N/n = rho*sqrt(cost_ratio)/sqrt(1-rho^2).
        """
        rho = float(np.clip(abs(rho), 0.0, 0.999999))
        ratio = rho * np.sqrt(cost_ratio) / np.sqrt(1 - rho ** 2)
        n_opt = budget / (cost_ratio + ratio)
        N_opt = n_opt * ratio
        expected_var = sigma_f ** 2 * (1 - rho ** 2) / max(n_opt, 1e-9)
        if N_opt > 0:
            expected_var += rho ** 2 * sigma_f ** 2 / N_opt
        return {
            "n_optimal": int(n_opt), "N_optimal": int(N_opt), "ratio": float(ratio),
            "expected_variance": float(expected_var), "expected_std": float(np.sqrt(expected_var)),
        }

    def plot_variance_comparison(self, mc_estimates: np.ndarray, pemc_estimates: np.ndarray,
                                 save_path: Optional[str] = None):
        """Returns a matplotlib Figure (use st.pyplot(fig)); no plt.show(), which doesn't work on Streamlit."""
        try:
            from matplotlib.figure import Figure
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None

        fig = Figure(figsize=(12, 10))
        ax = fig.subplots(2, 2)

        ax[0, 0].hist(mc_estimates, bins=50, alpha=0.5, label="Standard MC", density=True)
        ax[0, 0].hist(pemc_estimates, bins=50, alpha=0.5, label="PEMC", density=True)
        ax[0, 0].set(xlabel="Estimate", ylabel="Density", title="Distribution of Estimates")
        ax[0, 0].legend()

        ax[0, 1].boxplot([mc_estimates, pemc_estimates])
        ax[0, 1].set_xticklabels(["Standard MC", "PEMC"])
        ax[0, 1].set(ylabel="Estimate", title="Box Plot Comparison")

        k = np.arange(1, len(mc_estimates) + 1)
        k2 = np.arange(1, len(pemc_estimates) + 1)
        ax[1, 0].plot(np.cumsum(mc_estimates) / k, label="Standard MC", alpha=0.7)
        ax[1, 0].plot(np.cumsum(pemc_estimates) / k2, label="PEMC", alpha=0.7)
        ax[1, 0].set(xlabel="Number of Samples", ylabel="Cumulative Mean", title="Convergence")
        ax[1, 0].legend()

        mc_var, pemc_var = np.var(mc_estimates, ddof=1), np.var(pemc_estimates, ddof=1)
        red = (mc_var - pemc_var) / (mc_var + EPS) * 100
        ax[1, 1].bar(["Standard MC", "PEMC"], [mc_var, pemc_var], color=["red", "green"])
        ax[1, 1].set(ylabel="Variance", title=f"Variance Reduction: {red:.1f}%")

        for a in ax.flat:
            a.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def calculate_sample_efficiency(self, mc_var: float, pemc_var: float,
                                    target_variance: float) -> Dict[str, float]:
        mc_n = mc_var / target_variance
        pemc_n = pemc_var / target_variance
        eff = mc_n / (pemc_n + EPS)
        return {
            "mc_samples_needed": int(mc_n), "pemc_samples_needed": int(pemc_n),
            "efficiency_ratio": float(eff),
            "time_savings": float((1 - 1 / eff) * 100) if eff > 1 else 0.0,
        }

    def theoretical_variance_reduction(self, rho: float, n_coupled: int, n_independent: int,
                                       sigma_f: float = 1.0, sigma_g: float = 1.0) -> float:
        """PEMC variance = s_f^2(1-rho^2)/n + beta^2 s_g^2/N, with beta*s_g = rho*s_f.
        (sigma_g kept in the signature for compatibility; it cancels out.)"""
        var_mc = sigma_f ** 2 / n_coupled
        var_pemc = sigma_f ** 2 * (1 - rho ** 2) / n_coupled + (rho * sigma_f) ** 2 / n_independent
        return float(max(0.0, min(1.0, (var_mc - var_pemc) / var_mc)))

    def get_summary_stats(self) -> Dict:
        if not self.metrics_history:
            return {}
        red = [m.variance_reduction for m in self.metrics_history]
        return {
            "mean_reduction": float(np.mean(red)), "std_reduction": float(np.std(red)),
            "max_reduction": float(np.max(red)), "min_reduction": float(np.min(red)),
            "num_analyses": len(red),
        }