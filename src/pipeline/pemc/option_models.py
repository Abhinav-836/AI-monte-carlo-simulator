"""
Option pricing models for PEMC (ground-truth payoffs for training neural predictors)
"""

import logging
from typing import Callable, Tuple

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def _simulate(n_sims: int, S0: float, r: float, sigma: float, T: float, n_steps: int):
    dt = T / n_steps
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal((n_sims, n_steps))
    S = S0 * np.exp(np.cumsum(increments, axis=1))
    return increments, S, dt


def _path_features(S, increments, S0, dt, n_sims):
    early = np.mean(S[:, :10], axis=1) / S0 if S.shape[1] >= 10 else np.ones(n_sims)
    realized_vol = np.std(increments, axis=1) / np.sqrt(dt)  # annualised vol of increments
    return early, realized_vol, np.max(S, axis=1) / S0, np.min(S, axis=1) / S0


class OptionPricer:
    @staticmethod
    def black_scholes(S, K, T, r, sigma, option_type: str = "call") -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def monte_carlo_payoff(S0, K, T, r, sigma, n_sims, n_steps, option_type: str = "call"):
        _, S, _ = _simulate(n_sims, S0, r, sigma, T, n_steps)
        pay = np.maximum(S[:, -1] - K, 0) if option_type == "call" else np.maximum(K - S[:, -1], 0)
        return pay * np.exp(-r * T), S


class EuropeanOptionPricer(OptionPricer):
    """Was missing: create_sampler('european') crashed with AttributeError."""

    def payoff(self, n_sims, S0, K, T, r, sigma, n_steps: int = 252, option_type: str = "call"):
        inc, S, dt = _simulate(n_sims, S0, r, sigma, T, n_steps)
        pay = np.maximum(S[:, -1] - K, 0) if option_type == "call" else np.maximum(K - S[:, -1], 0)
        pay = pay * np.exp(-r * T)
        early, rv, mx, mn = _path_features(S, inc, S0, dt, n_sims)
        features = np.column_stack([
            np.full(n_sims, np.log(S0 / K)), np.full(n_sims, sigma * np.sqrt(T)),
            np.full(n_sims, r * T), early, rv, mx, mn])
        return pay, features


class AsianOptionPricer(OptionPricer):
    def payoff(self, n_sims, S0, K, T, r, sigma, n_steps: int = 252, option_type: str = "call"):
        inc, S, dt = _simulate(n_sims, S0, r, sigma, T, n_steps)
        avg = np.mean(S, axis=1)
        pay = np.maximum(avg - K, 0) if option_type == "call" else np.maximum(K - avg, 0)
        pay = pay * np.exp(-r * T)
        early, rv, mx, mn = _path_features(S, inc, S0, dt, n_sims)
        features = np.column_stack([
            np.full(n_sims, np.log(S0 / K)), np.full(n_sims, sigma * np.sqrt(T)),
            np.full(n_sims, r * T), early, rv, mx, mn])
        return pay, features


class BarrierOptionPricer(OptionPricer):
    """Up-and-out barrier option."""

    def payoff(self, n_sims, S0, K, B, T, r, sigma, n_steps: int = 252, option_type: str = "call"):
        inc, S, dt = _simulate(n_sims, S0, r, sigma, T, n_steps)
        max_abs = np.max(S, axis=1)
        hit = max_abs >= B
        vanilla = np.maximum(S[:, -1] - K, 0) if option_type == "call" else np.maximum(K - S[:, -1], 0)
        pay = np.where(hit, 0.0, vanilla) * np.exp(-r * T)

        early, rv, _, _ = _path_features(S, inc, S0, dt, n_sims)
        features = np.column_stack([
            np.full(n_sims, np.log(S0 / K)), np.full(n_sims, np.log(B / S0)),
            np.full(n_sims, sigma * np.sqrt(T)), np.full(n_sims, r * T),
            max_abs / B, early, rv])
        return pay, features


class VarianceSwapPricer(OptionPricer):
    def payoff(self, n_sims, S0, K_var, T, r, sigma, n_steps: int = 252):
        dt = T / n_steps
        lr = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal((n_sims, n_steps))
        realized_var = np.var(lr, axis=1) * 252
        pay = (realized_var - K_var) * np.exp(-r * T)

        early_var = np.mean(lr[:, :10] ** 2, axis=1) * 252 if n_steps >= 10 else np.ones(n_sims)
        features = np.column_stack([
            np.full(n_sims, sigma ** 2 / K_var), early_var, np.std(lr, axis=1),
            np.max(np.abs(lr), axis=1), np.sum(lr < -2 * sigma * np.sqrt(dt), axis=1)])
        return pay, features


def create_sampler(option_type: str) -> Callable:
    pricers = {
        "asian": AsianOptionPricer(), "barrier": BarrierOptionPricer(),
        "variance_swap": VarianceSwapPricer(), "european": EuropeanOptionPricer(),
    }
    pricer = pricers.get(option_type)
    if pricer is None:
        raise ValueError(f"Unknown option type: {option_type}")

    def sampler(n_samples: int, params: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        p = dict(params) if params else {
            "S0": 100, "K": 105, "T": 1.0, "r": 0.05, "sigma": 0.2, "n_steps": 252}
        if option_type == "variance_swap":
            p.pop("K", None)
            p.setdefault("K_var", 0.04)
        if option_type == "barrier":
            p.setdefault("B", 120)
        return pricer.payoff(n_sims=n_samples, **p)

    return sampler