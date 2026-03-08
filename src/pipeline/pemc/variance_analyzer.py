"""
Variance analysis for PEMC
Calculates and visualizes variance reduction
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class VarianceMetrics:
    """Container for variance analysis results"""
    mc_variance: float
    pemc_variance: float
    variance_reduction: float
    ci_width_reduction: float
    sample_efficiency: float
    optimal_allocation: Dict[str, float]

class VarianceAnalyzer:
    """
    Analyzes variance reduction achieved by PEMC
    Provides optimal sample allocation and efficiency metrics
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def calculate_variance_reduction(
        self,
        mc_estimates: np.ndarray,
        pemc_estimates: np.ndarray
    ) -> float:
        """
        Calculate variance reduction percentage
        
        Args:
            mc_estimates: Estimates from standard Monte Carlo
            pemc_estimates: Estimates from PEMC
        
        Returns:
            Variance reduction (0-1)
        """
        mc_var = np.var(mc_estimates, ddof=1)
        pemc_var = np.var(pemc_estimates, ddof=1)
        
        reduction = (mc_var - pemc_var) / mc_var
        return max(0, min(1, reduction))  # Clip to [0,1]
    
    def calculate_confidence_interval_width(
        self,
        estimates: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate confidence interval width
        
        Args:
            estimates: Array of estimates
            confidence: Confidence level
        
        Returns:
            Width of confidence interval
        """
        from scipy import stats
        
        mean = np.mean(estimates)
        std = np.std(estimates, ddof=1)
        n = len(estimates)
        
        se = std / np.sqrt(n)
        z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        ci_width = 2 * z_score * se
        return ci_width
    
    def analyze_variance_reduction(
        self,
        mc_estimates: np.ndarray,
        pemc_estimates: np.ndarray,
        confidence: float = 0.95
    ) -> VarianceMetrics:
        """
        Comprehensive variance reduction analysis
        
        Args:
            mc_estimates: Standard MC estimates
            pemc_estimates: PEMC estimates
            confidence: Confidence level
        
        Returns:
            VarianceMetrics object with detailed analysis
        """
        # Calculate variances
        mc_var = np.var(mc_estimates, ddof=1)
        pemc_var = np.var(pemc_estimates, ddof=1)
        
        # Variance reduction
        reduction = (mc_var - pemc_var) / mc_var
        
        # Confidence interval width reduction
        mc_ci_width = self.calculate_confidence_interval_width(mc_estimates, confidence)
        pemc_ci_width = self.calculate_confidence_interval_width(pemc_estimates, confidence)
        ci_reduction = (mc_ci_width - pemc_ci_width) / mc_ci_width
        
        # Sample efficiency
        # How many fewer samples PEMC needs to achieve same variance
        sample_efficiency = mc_var / pemc_var
        
        # Optimal allocation (simplified)
        optimal_n = 1000  # Placeholder
        optimal_N = int(optimal_n * sample_efficiency)
        
        optimal_allocation = {
            'n_coupled': optimal_n,
            'n_independent': optimal_N,
            'ratio': optimal_N / optimal_n
        }
        
        metrics = VarianceMetrics(
            mc_variance=mc_var,
            pemc_variance=pemc_var,
            variance_reduction=reduction,
            ci_width_reduction=ci_reduction,
            sample_efficiency=sample_efficiency,
            optimal_allocation=optimal_allocation
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def optimal_allocation(
        self,
        sigma_f: float,
        sigma_g: float,
        rho: float,
        cost_ratio: float = 10.0
    ) -> Dict[str, float]:
        """
        Compute optimal sample allocation between coupled and independent samples
        
        Based on PEMC theory: 
        Optimal ratio = sqrt( (σ²_f|g / c_f) / (σ²_g / c_g) )
        
        Args:
            sigma_f: Standard deviation of target f(Y)
            sigma_g: Standard deviation of predictor g(θ,X)
            rho: Correlation between f and g
            cost_ratio: Cost of expensive simulation / cost of cheap ML evaluation
        
        Returns:
            Dictionary with optimal n, N, and expected variance
        """
        # Conditional variance of f given g
        sigma_f_given_g = sigma_f**2 * (1 - rho**2)
        
        # Optimal ratio n/N
        optimal_ratio = np.sqrt(
            (sigma_f_given_g / 1.0) / (sigma_g**2 / cost_ratio)
        )
        
        # For a given computational budget, solve for n and N
        # Total cost = n * c_f + N * c_g
        # With c_g = 1, c_f = cost_ratio
        
        budget = 10000  # Example budget
        
        n_optimal = budget / (cost_ratio + 1/optimal_ratio)
        N_optimal = n_optimal * optimal_ratio
        
        # Expected variance
        expected_var = sigma_f_given_g / n_optimal + sigma_g**2 / N_optimal
        
        return {
            'n_optimal': int(n_optimal),
            'N_optimal': int(N_optimal),
            'ratio': optimal_ratio,
            'expected_variance': expected_var,
            'expected_std': np.sqrt(expected_var)
        }
    
    def plot_variance_comparison(
        self,
        mc_estimates: np.ndarray,
        pemc_estimates: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot variance comparison
        
        Args:
            mc_estimates: Standard MC estimates
            pemc_estimates: PEMC estimates
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Histogram comparison
            axes[0, 0].hist(mc_estimates, bins=50, alpha=0.5, label='Standard MC', density=True)
            axes[0, 0].hist(pemc_estimates, bins=50, alpha=0.5, label='PEMC', density=True)
            axes[0, 0].set_xlabel('Estimate')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Distribution of Estimates')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Box plot
            axes[0, 1].boxplot([mc_estimates, pemc_estimates], labels=['Standard MC', 'PEMC'])
            axes[0, 1].set_ylabel('Estimate')
            axes[0, 1].set_title('Box Plot Comparison')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Convergence plot
            n_samples = len(mc_estimates)
            mc_cummean = np.cumsum(mc_estimates) / np.arange(1, n_samples + 1)
            pemc_cummean = np.cumsum(pemc_estimates) / np.arange(1, n_samples + 1)
            
            axes[1, 0].plot(mc_cummean, label='Standard MC', alpha=0.7)
            axes[1, 0].plot(pemc_cummean, label='PEMC', alpha=0.7)
            axes[1, 0].axhline(y=np.mean(mc_estimates), color='gray', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Number of Samples')
            axes[1, 0].set_ylabel('Cumulative Mean')
            axes[1, 0].set_title('Convergence')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Variance reduction bar chart
            mc_var = np.var(mc_estimates, ddof=1)
            pemc_var = np.var(pemc_estimates, ddof=1)
            reduction = (mc_var - pemc_var) / mc_var * 100
            
            axes[1, 1].bar(['Standard MC', 'PEMC'], [mc_var, pemc_var], color=['red', 'green'])
            axes[1, 1].set_ylabel('Variance')
            axes[1, 1].set_title(f'Variance Reduction: {reduction:.1f}%')
            
            # Add text on bars
            for i, v in enumerate([mc_var, pemc_var]):
                axes[1, 1].text(i, v + 0.001, f'{v:.4f}', ha='center')
            
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def calculate_sample_efficiency(
        self,
        mc_var: float,
        pemc_var: float,
        target_variance: float
    ) -> Dict[str, float]:
        """
        Calculate sample efficiency to achieve target variance
        
        Args:
            mc_var: Variance of standard MC
            pemc_var: Variance of PEMC
            target_variance: Desired variance level
        
        Returns:
            Dictionary with required samples for each method
        """
        mc_samples_needed = mc_var / target_variance
        pemc_samples_needed = pemc_var / target_variance
        
        efficiency = mc_samples_needed / pemc_samples_needed
        
        return {
            'mc_samples_needed': int(mc_samples_needed),
            'pemc_samples_needed': int(pemc_samples_needed),
            'efficiency_ratio': efficiency,
            'time_savings': (1 - 1/efficiency) * 100 if efficiency > 1 else 0
        }
    
    def theoretical_variance_reduction(
        self,
        rho: float,
        n_coupled: int,
        n_independent: int,
        sigma_f: float = 1.0,
        sigma_g: float = 1.0
    ) -> float:
        """
        Calculate theoretical variance reduction
        
        Args:
            rho: Correlation between f and g
            n_coupled: Number of coupled samples
            n_independent: Number of independent samples
            sigma_f: Std dev of f
            sigma_g: Std dev of g
        
        Returns:
            Theoretical variance reduction
        """
        # Standard MC variance
        var_mc = sigma_f**2 / n_coupled
        
        # PEMC variance
        var_pemc = sigma_f**2 * (1 - rho**2) / n_coupled + sigma_g**2 / n_independent
        
        reduction = (var_mc - var_pemc) / var_mc
        return max(0, min(1, reduction))
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics from analysis history
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {}
        
        reductions = [m.variance_reduction for m in self.metrics_history]
        
        return {
            'mean_reduction': np.mean(reductions),
            'std_reduction': np.std(reductions),
            'max_reduction': np.max(reductions),
            'min_reduction': np.min(reductions),
            'num_analyses': len(reductions)
        }