"""
Confidence interval calculations for Monte Carlo results (PRODUCTION READY)
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Callable
import logging

logger = logging.getLogger(__name__)


class ConfidenceIntervalCalculator:
    """Robust and optimized confidence interval calculations"""

    # =========================
    # 🔥 FAST BOOTSTRAP (VECTORIZED)
    # =========================
    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic: str = 'mean',
        confidence: float = 0.95,
        n_bootstrap: int = 2000  # ✅ reduced for speed
    ) -> Tuple[float, float]:

        data = np.asarray(data)
        data = data[~np.isnan(data)]

        if len(data) < 2:
            return np.nan, np.nan

        # Select statistic
        if statistic == 'mean':
            stat_func = np.mean
        elif statistic == 'median':
            stat_func = np.median
        elif statistic == 'var':
            stat_func = np.var
        elif callable(statistic):
            stat_func = statistic
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        n = len(data)

        # ✅ Vectorized bootstrap (FAST)
        samples = np.random.choice(data, (n_bootstrap, n), replace=True)
        stats_vals = np.apply_along_axis(stat_func, 1, samples)

        alpha = 1 - confidence
        lower = np.percentile(stats_vals, 100 * alpha / 2)
        upper = np.percentile(stats_vals, 100 * (1 - alpha / 2))

        return float(lower), float(upper)

    # =========================
    # 📊 PARAMETRIC CI
    # =========================
    @staticmethod
    def parametric_ci(
        data: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:

        data = np.asarray(data)
        data = data[~np.isnan(data)]

        n = len(data)
        if n < 2:
            return np.nan, np.nan

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        se = std / np.sqrt(n)

        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        return float(mean - z * se), float(mean + z * se)

    # =========================
    # 🔮 PREDICTION INTERVAL
    # =========================
    @staticmethod
    def prediction_interval(
        data: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:

        data = np.asarray(data)
        data = data[~np.isnan(data)]

        n = len(data)
        if n < 2:
            return np.nan, np.nan

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        t = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
        se_pred = std * np.sqrt(1 + 1 / n)

        return float(mean - t * se_pred), float(mean + t * se_pred)

    # =========================
    # 📉 SIMULTANEOUS CI
    # =========================
    @staticmethod
    def simultaneous_ci(
        data: np.ndarray,
        confidence: float = 0.95
    ) -> List[Tuple[float, float]]:

        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError("Data must be 2D (n_points, n_samples)")

        n_points = data.shape[0]

        alpha = 1 - confidence
        point_conf = 1 - alpha / n_points

        cis = []
        for row in data:
            cis.append(
                ConfidenceIntervalCalculator.parametric_ci(row, point_conf)
            )

        return cis

    # =========================
    # 📈 FAN CHART
    # =========================
    @staticmethod
    def fan_chart(
        paths: np.ndarray,
        percentiles: List[float] = None
    ) -> dict:

        paths = np.asarray(paths)

        if paths.ndim != 2:
            raise ValueError("Paths must be 2D (n_paths, time_steps)")

        if percentiles is None:
            percentiles = [10, 25, 50, 75, 90]

        result = {
            'median': np.median(paths, axis=0),
            'mean': np.mean(paths, axis=0)
        }

        for p in percentiles:
            result[f'p{p}'] = np.percentile(paths, p, axis=0)

        return result

    # =========================
    # 🎯 PROBABILITY OF EXCEEDANCE (FIXED)
    # =========================
    @staticmethod
    def probability_of_exceedance(
        paths: np.ndarray,
        threshold: float,
        final_step: bool = True
    ) -> float:

        paths = np.asarray(paths)

        if paths.size == 0:
            return np.nan

        if final_step:
            values = paths[:, -1]
        else:
            values = paths.flatten()

        return float(np.mean(values > threshold))

    # =========================
    # 📉 EXPECTED SHORTFALL (SAFE)
    # =========================
    @staticmethod
    def expected_shortfall(
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:

        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return np.nan

        var = np.percentile(returns, 100 * (1 - confidence))

        tail_losses = returns[returns <= var]

        if len(tail_losses) == 0:
            return var  # fallback

        return float(np.mean(tail_losses))

    # =========================
    # 🧾 FORMATTER
    # =========================
    @staticmethod
    def confidence_interval_string(
        lower: float,
        upper: float,
        decimals: int = 2
    ) -> str:

        if np.isnan(lower) or np.isnan(upper):
            return "[N/A]"

        return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"
