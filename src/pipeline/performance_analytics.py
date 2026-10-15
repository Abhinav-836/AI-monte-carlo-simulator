"""
Advanced Performance Analytics Module
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


def _align(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align on the most recent observations (truncating from the start misaligns dates)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    return a[-n:], b[-n:]


def _drawdowns(returns: np.ndarray) -> np.ndarray:
    prices = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(prices)
    return (prices - peak) / (peak + 1e-8)


class PerformanceAnalytics:
    """Inputs are daily LOG returns (the drawdown helpers compound with exp(cumsum))."""

    @staticmethod
    def rolling_alpha_beta(returns: np.ndarray, benchmark_returns: np.ndarray,
                           window: int = 252) -> Dict[str, np.ndarray]:
        returns, benchmark_returns = _align(returns, benchmark_returns)
        n = len(returns)
        alpha = np.full(n, np.nan)
        beta = np.full(n, np.nan)
        r_squared = np.full(n, np.nan)

        for i in range(window, n):
            y = returns[i - window:i]
            x = benchmark_returns[i - window:i]
            var_x = np.var(x, ddof=1)
            if var_x <= 0:
                continue
            b = np.cov(x, y)[0, 1] / var_x
            a = np.mean(y) - b * np.mean(x)
            resid = y - (b * x + a)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            beta[i], alpha[i] = b, a
            r_squared[i] = 1 - np.sum(resid ** 2) / (ss_tot + 1e-12)
        return {"alpha": alpha, "beta": beta, "r_squared": r_squared}

    @staticmethod
    def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray,
                          annualize: bool = True) -> float:
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        r, b = _align(returns, benchmark_returns)
        active = r - b
        mean_a, te = np.mean(active), np.std(active, ddof=1)
        if annualize:
            mean_a *= TRADING_DAYS
            te *= np.sqrt(TRADING_DAYS)
        return float(mean_a / (te + 1e-8))

    @staticmethod
    def treynor_ratio(returns: np.ndarray, beta: float, risk_free_rate: float = 0.02) -> float:
        if len(returns) == 0 or beta == 0:
            return 0.0
        return float((np.mean(returns) * TRADING_DAYS - risk_free_rate) / beta)

    @staticmethod
    def jensens_alpha(returns: np.ndarray, benchmark_returns: np.ndarray, beta: float,
                      risk_free_rate: float = 0.02) -> float:
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        r, b = _align(returns, benchmark_returns)
        rf = risk_free_rate / TRADING_DAYS
        expected = rf + beta * (np.mean(b) - rf)
        return float((np.mean(r) - expected) * TRADING_DAYS)

    @staticmethod
    def capture_ratios(returns: np.ndarray, benchmark_returns: np.ndarray) -> Dict[str, float]:
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return {"up_capture": 0.0, "down_capture": 0.0}
        r, b = _align(returns, benchmark_returns)
        up, down = b > 0, b < 0
        up_c = r[up].sum() / (b[up].sum() + 1e-8) if up.any() else 0.0
        down_c = r[down].sum() / (b[down].sum() + 1e-8) if down.any() else 0.0
        return {"up_capture": float(up_c), "down_capture": float(down_c)}

    @staticmethod
    def ulcer_index(returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        return float(np.sqrt(np.mean(_drawdowns(returns) ** 2)))

    @staticmethod
    def martin_ratio(returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        return float(np.mean(returns) * TRADING_DAYS / (PerformanceAnalytics.ulcer_index(returns) + 1e-8))

    @staticmethod
    def downside_deviation(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Lower partial moment over ALL observations (the old version averaged only the losers)."""
        if len(returns) == 0:
            return 0.0
        t = threshold / TRADING_DAYS
        return float(np.sqrt(np.mean(np.minimum(returns - t, 0.0) ** 2)) * np.sqrt(TRADING_DAYS))

    @staticmethod
    def upside_potential(returns: np.ndarray, threshold: float = 0.0) -> float:
        if len(returns) == 0:
            return 0.0
        t = threshold / TRADING_DAYS
        return float(np.sqrt(np.mean(np.maximum(returns - t, 0.0) ** 2)) * np.sqrt(TRADING_DAYS))

    @staticmethod
    def kelly_criterion(returns: np.ndarray) -> float:
        """Continuous Kelly fraction mu/sigma^2, clipped to a long-only [0, 1]."""
        if len(returns) == 0:
            return 0.0
        var = np.var(returns, ddof=1)
        if var == 0:
            return 0.0
        return float(np.clip(np.mean(returns) / var, 0.0, 1.0))

    @staticmethod
    def pain_index(returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        dd = _drawdowns(returns)
        neg = dd[dd < 0]
        return float(np.mean(neg)) if len(neg) else 0.0

    @staticmethod
    def pain_ratio(returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        pain = abs(PerformanceAnalytics.pain_index(returns))
        return float(np.mean(returns) * TRADING_DAYS / (pain + 1e-8))

    @staticmethod
    def calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        if len(returns) == 0:
            return 0.0
        dd = PerformanceAnalytics.downside_deviation(returns)
        return float((np.mean(returns) * TRADING_DAYS - risk_free_rate) / (dd + 1e-8))

    @staticmethod
    def calculate_all_performance_metrics(returns: np.ndarray,
                                          benchmark_returns: Optional[np.ndarray] = None,
                                          risk_free_rate: float = 0.02) -> Dict[str, float]:
        returns = np.asarray(returns, float)
        if len(returns) == 0:
            return {}

        annual_return = float(np.mean(returns) * TRADING_DAYS)
        vol = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))
        dd = _drawdowns(returns)
        neg = dd[dd < 0]

        m = {
            "annual_return": annual_return,
            "volatility": vol,
            "sharpe": float((annual_return - risk_free_rate) / (vol + 1e-8)),
            "skewness": float(stats.skew(returns)),
            "kurtosis": float(stats.kurtosis(returns)),
            "max_drawdown": float(np.min(dd)),
            "avg_drawdown": float(np.mean(neg)) if len(neg) else 0.0,
        }
        m["ulcer_index"] = PerformanceAnalytics.ulcer_index(returns)
        m["pain_index"] = PerformanceAnalytics.pain_index(returns)
        m["sortino"] = PerformanceAnalytics.calculate_sortino(returns, risk_free_rate)
        m["calmar"] = annual_return / (abs(m["max_drawdown"]) + 1e-8)
        m["martin_ratio"] = annual_return / (m["ulcer_index"] + 1e-8)
        m["pain_ratio"] = annual_return / (abs(m["pain_index"]) + 1e-8)
        m["kelly_fraction"] = PerformanceAnalytics.kelly_criterion(returns)

        if benchmark_returns is not None and len(benchmark_returns) > 1:
            r, b = _align(returns, benchmark_returns)
            m["information_ratio"] = PerformanceAnalytics.information_ratio(r, b)
            var_b = np.var(b, ddof=1)
            if var_b > 0:
                beta = float(np.cov(r, b)[0, 1] / var_b)
                rf = risk_free_rate / TRADING_DAYS
                m["beta"] = beta
                m["alpha"] = float(((np.mean(r) - rf) - beta * (np.mean(b) - rf)) * TRADING_DAYS)
                m["treynor"] = float((annual_return - risk_free_rate) / beta) if beta != 0 else 0.0
            cap = PerformanceAnalytics.capture_ratios(r, b)
            m["up_capture"], m["down_capture"] = cap["up_capture"], cap["down_capture"]
        return m