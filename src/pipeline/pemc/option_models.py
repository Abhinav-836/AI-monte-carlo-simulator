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
        """Black-Scholes option price"""
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
        """Monte Carlo simulation for option payoff"""
        dt = T / n_steps

        Z = np.random.standard_normal((n_sims, n_steps))
        increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_returns = np.cumsum(increments, axis=1)

        S = S0 * np.exp(log_returns)

        if option_type == 'call':
            payoffs = np.maximum(S[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S[:, -1], 0)

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
        """Asian option payoff (average price)"""
        dt = T / n_steps

        Z = np.random.standard_normal((n_sims, n_steps))
        increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_returns = np.cumsum(increments, axis=1)

        S = S0 * np.exp(log_returns)

        avg_price = np.mean(S, axis=1)

        if option_type == 'call':
            payoffs = np.maximum(avg_price - K, 0)
        else:
            payoffs = np.maximum(K - avg_price, 0)

        payoffs = payoffs * np.exp(-r * T)

        moneyness = np.full(n_sims, np.log(S0 / K))
        total_vol = np.full(n_sims, sigma * np.sqrt(T))
        discount = np.full(n_sims, r * T)

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

    # ============================================================
    # ✅ FIXED: max_price variable was shadowed (ratio vs absolute)
    #    → features were wrong → PEMC predictor trained on garbage
    # ============================================================
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
        """Up-and-out barrier option payoff"""
        dt = T / n_steps

        Z = np.random.standard_normal((n_sims, n_steps))
        increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_returns = np.cumsum(increments, axis=1)

        S = S0 * np.exp(log_returns)

        # ✅ FIX: keep absolute max price for barrier detection
        max_price_abs = np.max(S, axis=1)
        barrier_hit = max_price_abs >= B

        if option_type == 'call':
            vanilla_payoff = np.maximum(S[:, -1] - K, 0)
        else:
            vanilla_payoff = np.maximum(K - S[:, -1], 0)

        payoffs = np.where(barrier_hit, 0, vanilla_payoff)
        payoffs = payoffs * np.exp(-r * T)

        # ✅ FIX: use distinct variable names — no more shadowing
        moneyness = np.full(n_sims, np.log(S0 / K))
        barrier_distance = np.full(n_sims, np.log(B / S0))
        total_vol = np.full(n_sims, sigma * np.sqrt(T))
        discount = np.full(n_sims, r * T)
        max_to_barrier = max_price_abs / B  # ✅ now correct units

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
        """Variance swap payoff = realized variance - strike variance"""
        dt = T / n_steps

        Z = np.random.standard_normal((n_sims, n_steps))
        increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_returns = increments

        realized_var = np.var(log_returns, axis=1) * 252

        payoffs = (realized_var - K_var) * np.exp(-r * T)

        implied_vs_strike = np.full(n_sims, sigma ** 2 / K_var)

        if n_steps >= 10:
            early_var = np.mean(log_returns[:, :10] ** 2, axis=1) * 252
        else:
            early_var = np.ones(n_sims)

        realized_std = np.std(log_returns, axis=1)
        max_daily = np.max(np.abs(log_returns), axis=1)
        jump_count = np.sum(log_returns < -2 * sigma * np.sqrt(dt), axis=1)

        features = np.column_stack([
            implied_vs_strike,
            early_var,
            realized_std,
            max_daily,
            jump_count
        ])

        return payoffs, features


def create_sampler(option_type: str) -> Callable:
    """Factory function to create appropriate sampler for PEMC"""
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
        """Sampler function for PEMC"""
        if params is None:
            params = {
                'S0': 100,
                'K': 105,
                'T': 1.0,
                'r': 0.05,
                'sigma': 0.2,
                'n_steps': 252
            }

        if option_type == 'barrier' and 'B' not in params:
            params['B'] = 120

        if option_type == 'variance_swap' and 'K_var' not in params:
            params['K_var'] = 0.04

        return pricer.payoff(n_sims=n_samples, **params)

    return sampler