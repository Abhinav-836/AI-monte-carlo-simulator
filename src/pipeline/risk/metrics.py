"""
Risk metrics calculation from simulated paths
Quant-grade implementation (correct math)
"""

import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class RiskCalculator:
    """Calculate risk metrics from simulated paths"""

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
    def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        if returns.size == 0 or np.std(returns) == 0:
            return 0.0

        mean_daily = np.mean(returns)
        vol_daily = np.std(returns)

        annual_return = mean_daily * 252
        annual_vol = vol_daily * np.sqrt(252)

        return float((annual_return - risk_free_rate) / (annual_vol + 1e-8))

    @staticmethod
    def calculate_max_drawdown(price_path: np.ndarray) -> float:
        if price_path.size == 0:
            return 0.0

        peak = np.maximum.accumulate(price_path)
        drawdown = (price_path - peak) / (peak + 1e-8)
        return float(np.min(drawdown))

    def calculate_all_metrics(
        self,
        paths: np.ndarray,
        asset_names: List[str],
        input_type: str = "returns"  # 🔥 NEW
    ) -> Dict[str, Dict[str, float]]:
        """
        paths shape: (n_paths, seq_len, n_assets)

        input_type:
            "returns" -> paths are log returns
            "prices"  -> paths are price paths
        """

        if paths is None or len(paths) == 0:
            return {}

        if paths.ndim != 3:
            raise ValueError("paths must be (n_paths, seq_len, n_assets)")

        n_paths, seq_len, n_assets = paths.shape

        metrics = {}

        for i, asset in enumerate(asset_names[:n_assets]):

            if input_type == "returns":
                # Use returns directly
                asset_returns = paths[:, :, i].flatten()

                # Convert to price paths for drawdown
                price_paths = np.exp(np.cumsum(paths[:, :, i], axis=1))

            else:
                # Convert price → returns
                prices = paths[:, :, i]
                asset_returns = np.diff(np.log(prices + 1e-8), axis=1).flatten()
                price_paths = prices

            # Clean data
            asset_returns = asset_returns[np.isfinite(asset_returns)]

            if asset_returns.size == 0:
                logger.warning(f"No valid returns for {asset}")
                continue

            # 🔥 Compute drawdown PER PATH
            drawdowns = [
                self.calculate_max_drawdown(price_paths[j])
                for j in range(len(price_paths))
            ]

            metrics[asset] = {
                "var_95": self.calculate_var(asset_returns, 0.95),
                "cvar_95": self.calculate_cvar(asset_returns, 0.95),
                "sharpe": self.calculate_sharpe(asset_returns),
                "max_drawdown": float(np.mean(drawdowns)),
                "expected_return": float(np.mean(asset_returns) * 252),
                "volatility": float(np.std(asset_returns) * np.sqrt(252)),
            }

        return metrics
