"""
Advanced Risk Metrics
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")


@dataclass
class RiskMetrics:
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    volatility: float
    expected_return: float
    skewness: float
    kurtosis: float
    omega_ratio: float
    tail_ratio: float
    gain_to_pain_ratio: float
    value_at_risk_historical: float
    expected_shortfall: float


class RiskCalculator:

    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        if returns.size == 0:
            return 0.0
        return float(np.percentile(returns, (1 - confidence) * 100))

    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        if returns.size == 0:
            return 0.0
        var = RiskCalculator.calculate_var(returns, confidence)
        tail = returns[returns <= var]
        return float(np.mean(tail)) if len(tail) > 0 else var

    @staticmethod
    def calculate_var_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
        if returns.size == 0:
            return 0.0
        return float(np.mean(returns) + stats.norm.ppf(1 - confidence) * np.std(returns))

    @staticmethod
    def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.02, annualize: bool = True) -> float:
        if returns.size == 0 or np.std(returns) == 0:
            return 0.0
        mean_d, vol_d = np.mean(returns), np.std(returns)
        if annualize:
            return float((mean_d * 252 - risk_free_rate) / (vol_d * np.sqrt(252) + 1e-8))
        return float((mean_d - risk_free_rate / 252) / (vol_d + 1e-8))

    @staticmethod
    def calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Downside deviation is measured over ALL observations (target 0), not just the losers."""
        if returns.size == 0:
            return 0.0
        downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)) * np.sqrt(252)
        if downside == 0:
            return 0.0
        return float((np.mean(returns) * 252 - risk_free_rate) / downside)

    @staticmethod
    def calculate_max_drawdown(price_path: np.ndarray) -> float:
        if price_path.size == 0:
            return 0.0
        peak = np.maximum.accumulate(price_path)
        return float(np.min((price_path - peak) / (peak + 1e-8)))

    @staticmethod
    def calculate_calmar(returns: np.ndarray, max_drawdown: float = None) -> float:
        if returns.size == 0:
            return 0.0
        annual_return = np.mean(returns) * 252
        if max_drawdown is None:
            max_drawdown = RiskCalculator.calculate_max_drawdown(np.exp(np.cumsum(returns)))
        return float(annual_return / (abs(max_drawdown) + 1e-8))

    @staticmethod
    def calculate_drawdown_duration(price_path: np.ndarray) -> Dict:
        if price_path.size == 0:
            return {"max_duration": 0, "avg_duration": 0, "num_periods": 0}
        peak = np.maximum.accumulate(price_path)
        in_dd = (price_path - peak) / (peak + 1e-8) < 0
        durations, current = [], 0
        for flag in in_dd:
            if flag:
                current += 1
            elif current > 0:
                durations.append(current)
                current = 0
        if current > 0:
            durations.append(current)
        return {
            "max_duration": max(durations) if durations else 0,
            "avg_duration": float(np.mean(durations)) if durations else 0,
            "num_periods": len(durations),
        }

    @staticmethod
    def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        if returns.size == 0:
            return 0.0
        t = threshold / 252
        gains = np.sum(np.maximum(returns - t, 0))
        losses = np.sum(np.maximum(t - returns, 0))
        return float(gains / (losses + 1e-8))

    @staticmethod
    def calculate_tail_ratio(returns: np.ndarray, tail_percent: float = 0.05) -> float:
        if returns.size == 0:
            return 0.0
        left = np.percentile(returns, tail_percent * 100)
        right = np.percentile(returns, (1 - tail_percent) * 100)
        return float(abs(right) / (abs(left) + 1e-8))

    @staticmethod
    def calculate_gain_to_pain_ratio(returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        total_gain = np.sum(returns[returns > 0])
        total_loss = abs(np.sum(returns[returns < 0]))
        return float(total_gain / (total_loss + 1e-8))

    @staticmethod
    def calculate_skewness(returns: np.ndarray) -> float:
        return float(stats.skew(returns)) if returns.size >= 2 else 0.0

    @staticmethod
    def calculate_kurtosis(returns: np.ndarray) -> float:
        return float(stats.kurtosis(returns)) if returns.size >= 2 else 0.0

    @staticmethod
    def calculate_historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        return RiskCalculator.calculate_var(returns, confidence)

    @staticmethod
    def calculate_expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
        return RiskCalculator.calculate_cvar(returns, confidence)

    @staticmethod
    def _capture(returns: np.ndarray, benchmark: np.ndarray, up: bool) -> float:
        if returns.size == 0 or benchmark.size == 0:
            return 0.0
        n = min(len(returns), len(benchmark))
        r, b = returns[-n:], benchmark[-n:]  # align on the most recent observations
        mask = b > 0 if up else b < 0
        if not mask.any():
            return 0.0
        return float(r[mask].sum() / (b[mask].sum() + 1e-8))

    @staticmethod
    def calculate_upside_capture(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        return RiskCalculator._capture(returns, benchmark_returns, True)

    @staticmethod
    def calculate_downside_capture(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        return RiskCalculator._capture(returns, benchmark_returns, False)

    @classmethod
    def calculate_all_metrics(
        cls,
        paths: np.ndarray,
        asset_names: List[str],
        input_type: str = "returns",
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """paths: (n_paths, seq_len, n_assets)"""
        if paths is None or len(paths) == 0:
            return {}
        if paths.ndim != 3:
            raise ValueError("paths must be (n_paths, seq_len, n_assets)")

        metrics = {}
        for i, asset in enumerate(asset_names[: paths.shape[2]]):
            if input_type == "returns":
                asset_returns = paths[:, :, i].flatten()
                price_paths = np.exp(np.cumsum(paths[:, :, i], axis=1))
            else:
                price_paths = paths[:, :, i]
                asset_returns = np.diff(np.log(price_paths + 1e-8), axis=1).flatten()

            asset_returns = asset_returns[np.isfinite(asset_returns)]
            if asset_returns.size == 0:
                continue

            # Per-path drawdown (drawdown of the *average* path hides almost all the risk)
            peak = np.maximum.accumulate(price_paths, axis=1)
            per_path_dd = np.min((price_paths - peak) / (peak + 1e-8), axis=1)
            avg_dd = float(np.mean(per_path_dd))
            dd_duration = cls.calculate_drawdown_duration(np.mean(price_paths, axis=0))

            up = down = 0.0
            if benchmark_returns is not None:
                up = cls.calculate_upside_capture(asset_returns, benchmark_returns)
                down = cls.calculate_downside_capture(asset_returns, benchmark_returns)

            metrics[asset] = {
                "var_95": cls.calculate_var(asset_returns, 0.95),
                "cvar_95": cls.calculate_cvar(asset_returns, 0.95),
                "var_99": cls.calculate_var(asset_returns, 0.99),
                "cvar_99": cls.calculate_cvar(asset_returns, 0.99),
                "sharpe": cls.calculate_sharpe(asset_returns),
                "sortino": cls.calculate_sortino(asset_returns),
                "calmar": cls.calculate_calmar(asset_returns, avg_dd),
                "max_drawdown": avg_dd,
                "worst_drawdown": float(np.min(per_path_dd)),
                "max_drawdown_duration": float(dd_duration["max_duration"]),
                "avg_drawdown_duration": float(dd_duration["avg_duration"]),
                "volatility": float(np.std(asset_returns) * np.sqrt(252)),
                "expected_return": float(np.mean(asset_returns) * 252),
                "skewness": cls.calculate_skewness(asset_returns),
                "kurtosis": cls.calculate_kurtosis(asset_returns),
                "omega_ratio": cls.calculate_omega_ratio(asset_returns),
                "tail_ratio": cls.calculate_tail_ratio(asset_returns),
                "gain_to_pain_ratio": cls.calculate_gain_to_pain_ratio(asset_returns),
                "upside_capture": up,
                "downside_capture": down,
            }
        return metrics


class RollingRiskCalculator:
    def __init__(self, window: int = 252):
        self.window = window

    def calculate_rolling_var(self, returns: np.ndarray, confidence: float = 0.95) -> np.ndarray:
        out = np.full(len(returns), np.nan)
        for i in range(self.window, len(returns)):
            out[i] = np.percentile(returns[i - self.window:i], (1 - confidence) * 100)
        return out

    def calculate_rolling_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> np.ndarray:
        out = np.full(len(returns), np.nan)
        for i in range(self.window, len(returns)):
            w = returns[i - self.window:i]
            vol = np.std(w)
            if vol > 0:
                out[i] = (np.mean(w) * 252 - risk_free_rate) / (vol * np.sqrt(252))
        return out

    def calculate_rolling_volatility(self, returns: np.ndarray) -> np.ndarray:
        out = np.full(len(returns), np.nan)
        for i in range(self.window, len(returns)):
            out[i] = np.std(returns[i - self.window:i]) * np.sqrt(252)
        return out