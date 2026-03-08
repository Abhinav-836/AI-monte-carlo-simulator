"""
Option pricing models for PEMC
Provides ground truth payoffs for training neural predictors
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class OptionPricer:
    """Base class for option pricing models"""
    
    @staticmethod
    def black_scholes(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call'
    ) -> float:
        """
        Black-Scholes option price
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    @staticmethod
    def monte_carlo_payoff(
        S0: float, K: float, T: float, r: float, sigma: float,
        n_sims: int, n_steps: int, option_type: str = 'call'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo simulation for option payoff
        
        Returns:
            payoffs: Option payoffs for each simulation
            paths: Price paths
        """
        dt = T / n_steps
        
        # Generate price paths
        Z = np.random.standard_normal((n_sims, n_steps))
        increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_returns = np.cumsum(increments, axis=1)
        
        S = S0 * np.exp(log_returns)
        
        # Calculate payoff
        if option_type == 'call':
            payoffs = np.maximum(S[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S[:, -1], 0)
        
        # Discount to present value
        payoffs = payoffs * np.exp(-r * T)
        
        return payoffs, S


class AsianOptionPricer(OptionPricer):
    """Asian option (average price)"""
    
    def payoff(
        self,
        n_sims: int,
        S0: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float,
        n_steps: int = 252,
        option_type: str = 'call'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Asian option payoff (average price)
        """
        dt = T / n_steps
        
        # Generate price paths
        Z = np.random.standard_normal((n_sims, n_steps))
        increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_returns = np.cumsum(increments, axis=1)
        
        S = S0 * np.exp(log_returns)
        
        # Average price
        avg_price = np.mean(S, axis=1)
        
        # Payoff based on average
        if option_type == 'call':
            payoffs = np.maximum(avg_price - K, 0)
        else:
            payoffs = np.maximum(K - avg_price, 0)
        
        payoffs = payoffs * np.exp(-r * T)
        
        # Features for predictor - ensure all are 1D arrays with same shape
        moneyness = np.full(n_sims, np.log(S0 / K))
        total_vol = np.full(n_sims, sigma * np.sqrt(T))
        discount = np.full(n_sims, r * T)
        
        # Early average (first 10 steps or all if less)
        if n_steps >= 10:
            early_avg = np.mean(S[:, :10], axis=1) / S0
        else:
            early_avg = np.ones(n_sims)
        
        realized_vol = np.std(log_returns, axis=1)
        max_price = np.max(S, axis=1) / S0
        min_price = np.min(S, axis=1) / S0
        
        features = np.column_stack([
            moneyness,
            total_vol,
            discount,
            early_avg,
            realized_vol,
            max_price,
            min_price
        ])
        
        return payoffs, features


class BarrierOptionPricer(OptionPricer):
    """Barrier option (up-and-out)"""
    
    def payoff(
        self,
        n_sims: int,
        S0: float, 
        K: float, 
        B: float, 
        T: float, 
        r: float, 
        sigma: float,
        n_steps: int = 252,
        option_type: str = 'call'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Up-and-out barrier option payoff
        Option knocks out if price hits barrier B
        """
        dt = T / n_steps
        
        # Generate price paths
        Z = np.random.standard_normal((n_sims, n_steps))
        increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_returns = np.cumsum(increments, axis=1)
        
        S = S0 * np.exp(log_returns)
        
        # Check if barrier was hit
        max_price = np.max(S, axis=1)
        barrier_hit = max_price >= B
        
        # Payoff
        if option_type == 'call':
            vanilla_payoff = np.maximum(S[:, -1] - K, 0)
        else:
            vanilla_payoff = np.maximum(K - S[:, -1], 0)
        
        # Knock out if barrier hit
        payoffs = np.where(barrier_hit, 0, vanilla_payoff)
        payoffs = payoffs * np.exp(-r * T)
        
        # Features for predictor - ensure all are 1D arrays with same shape
        moneyness = np.full(n_sims, np.log(S0 / K))
        barrier_distance = np.full(n_sims, np.log(B / S0))
        total_vol = np.full(n_sims, sigma * np.sqrt(T))
        discount = np.full(n_sims, r * T)
        max_to_barrier = max_price / B
        
        if n_steps >= 10:
            early_avg = np.mean(S[:, :10], axis=1) / S0
        else:
            early_avg = np.ones(n_sims)
            
        realized_vol = np.std(log_returns, axis=1)
        
        features = np.column_stack([
            moneyness,
            barrier_distance,
            total_vol,
            discount,
            max_to_barrier,
            early_avg,
            realized_vol
        ])
        
        return payoffs, features


class VarianceSwapPricer(OptionPricer):
    """Variance swap"""
    
    def payoff(
        self,
        n_sims: int,
        S0: float, 
        K_var: float, 
        T: float, 
        r: float, 
        sigma: float,
        n_steps: int = 252
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Variance swap payoff = realized variance - strike variance
        """
        dt = T / n_steps
        
        # Generate price paths
        Z = np.random.standard_normal((n_sims, n_steps))
        increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_returns = increments  # Daily log returns
        
        # Realized variance (annualized)
        realized_var = np.var(log_returns, axis=1) * 252
        
        # Payoff
        payoffs = (realized_var - K_var) * np.exp(-r * T)
        
        # Features for predictor - ensure all are 1D arrays with same shape
        implied_vs_strike = np.full(n_sims, sigma ** 2 / K_var)
        
        if n_steps >= 10:
            early_var = np.mean(log_returns[:, :10] ** 2, axis=1) * 252
        else:
            early_var = np.ones(n_sims)
            
        realized_std = np.std(log_returns, axis=1)
        max_daily = np.max(np.abs(log_returns), axis=1)
        jump_count = np.sum(log_returns < -2*sigma*np.sqrt(dt), axis=1)
        
        features = np.column_stack([
            implied_vs_strike,
            early_var,
            realized_std,
            max_daily,
            jump_count
        ])
        
        return payoffs, features


def create_sampler(option_type: str) -> Callable:
    """
    Factory function to create appropriate sampler for PEMC
    
    Args:
        option_type: 'asian', 'barrier', 'variance_swap', or 'european'
    
    Returns:
        Sampler function that takes (n_samples, params) and returns (payoffs, features)
    """
    pricers = {
        'asian': AsianOptionPricer(),
        'barrier': BarrierOptionPricer(),
        'variance_swap': VarianceSwapPricer(),
        'european': OptionPricer()
    }
    
    pricer = pricers.get(option_type)
    if pricer is None:
        raise ValueError(f"Unknown option type: {option_type}")
    
    def sampler(n_samples: int, params: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sampler function for PEMC
        
        Args:
            n_samples: Number of samples to generate
            params: Dictionary of option parameters
        
        Returns:
            payoffs: Array of option payoffs (n_samples,)
            features: Array of features (n_samples, n_features)
        """
        if params is None:
            params = {
                'S0': 100,
                'K': 105,
                'T': 1.0,
                'r': 0.05,
                'sigma': 0.2,
                'n_steps': 252
            }
        
        # Add barrier-specific parameter
        if option_type == 'barrier' and 'B' not in params:
            params['B'] = 120
        
        # Add variance swap-specific parameter
        if option_type == 'variance_swap' and 'K_var' not in params:
            params['K_var'] = 0.04
        
        return pricer.payoff(n_sims=n_samples, **params)
    
    return sampler