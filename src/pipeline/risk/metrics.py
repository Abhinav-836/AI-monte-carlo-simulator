"""
Advanced Risk Metrics with Full Quant Library
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics"""
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
    """
    Professional risk metrics calculator with full quant library
    """
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Value at Risk using historical method"""
        if returns.size == 0:
            return 0.0
        return float(np.percentile(returns, (1 - confidence) * 100))

    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall)"""
        if returns.size == 0:
            return 0.0
        var = RiskCalculator.calculate_var(returns, confidence)
        tail = returns[returns <= var]
        return float(np.mean(tail)) if len(tail) > 0 else var

    @staticmethod
    def calculate_var_parametric(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Parametric VaR assuming normal distribution"""
        if returns.size == 0:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence)
        return float(mean + z_score * std)

    @staticmethod
    def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.02, annualize: bool = True) -> float:
        """Sharpe ratio with annualization"""
        if returns.size == 0 or np.std(returns) == 0:
            return 0.0
        
        mean_daily = np.mean(returns)
        vol_daily = np.std(returns)
        
        if annualize:
            annual_return = mean_daily * 252
            annual_vol = vol_daily * np.sqrt(252)
            return float((annual_return - risk_free_rate) / (annual_vol + 1e-8))
        else:
            return float((mean_daily - risk_free_rate/252) / (vol_daily + 1e-8))

    @staticmethod
    def calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Sortino ratio (uses downside deviation)"""
        if returns.size == 0:
            return 0.0
        
        mean_return = np.mean(returns) * 252
        downside_returns = returns[returns < 0]
        downside_risk = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.01
        
        return float((mean_return - risk_free_rate) / (downside_risk + 1e-8))

    @staticmethod
    def calculate_calmar(returns: np.ndarray, max_drawdown: float = None) -> float:
        """Calmar ratio (return / max drawdown)"""
        if returns.size == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        if max_drawdown is None:
            max_dd = RiskCalculator.calculate_max_drawdown(np.exp(np.cumsum(returns)))
        else:
            max_dd = max_drawdown
        
        return float(annual_return / (abs(max_dd) + 1e-8))

    @staticmethod
    def calculate_max_drawdown(price_path: np.ndarray) -> float:
        """Maximum drawdown from price path"""
        if price_path.size == 0:
            return 0.0
        
        peak = np.maximum.accumulate(price_path)
        drawdown = (price_path - peak) / (peak + 1e-8)
        return float(np.min(drawdown))

    @staticmethod
    def calculate_drawdown_duration(price_path: np.ndarray) -> Dict:
        """Calculate drawdown duration statistics"""
        if price_path.size == 0:
            return {'max_duration': 0, 'avg_duration': 0}
        
        peak = np.maximum.accumulate(price_path)
        drawdown = (price_path - peak) / (peak + 1e-8)
        
        in_drawdown = drawdown < 0
        durations = []
        current = 0
        
        for in_dd in in_drawdown:
            if in_dd:
                current += 1
            else:
                if current > 0:
                    durations.append(current)
                    current = 0
        
        if current > 0:
            durations.append(current)
        
        return {
            'max_duration': max(durations) if durations else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'num_periods': len(durations)
        }

    @staticmethod
    def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Omega ratio (probability weighted returns above threshold)"""
        if returns.size == 0:
            return 0.0
        
        threshold_daily = threshold / 252
        gains = returns[returns > threshold_daily] - threshold_daily
        losses = threshold_daily - returns[returns < threshold_daily]
        
        total_gain = np.sum(gains)
        total_loss = np.sum(losses)
        
        return float(total_gain / (total_loss + 1e-8))

    @staticmethod
    def calculate_tail_ratio(returns: np.ndarray, tail_percent: float = 0.05) -> float:
        """Tail ratio (right tail / left tail)"""
        if returns.size == 0:
            return 0.0
        
        left_tail = np.percentile(returns, tail_percent * 100)
        right_tail = np.percentile(returns, (1 - tail_percent) * 100)
        
        return float(abs(right_tail) / (abs(left_tail) + 1e-8))

    @staticmethod
    def calculate_gain_to_pain_ratio(returns: np.ndarray) -> float:
        """Gain to pain ratio (total gain / total loss)"""
        if returns.size == 0:
            return 0.0
        
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        total_gain = np.sum(gains)
        total_loss = abs(np.sum(losses))
        
        return float(total_gain / (total_loss + 1e-8))

    @staticmethod
    def calculate_skewness(returns: np.ndarray) -> float:
        """Skewness of returns"""
        if returns.size < 2:
            return 0.0
        return float(stats.skew(returns))

    @staticmethod
    def calculate_kurtosis(returns: np.ndarray) -> float:
        """Excess kurtosis of returns"""
        if returns.size < 2:
            return 0.0
        return float(stats.kurtosis(returns))

    @staticmethod
    def calculate_historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Historical VaR with confidence level"""
        if returns.size == 0:
            return 0.0
        return float(np.percentile(returns, (1 - confidence) * 100))

    @staticmethod
    def calculate_expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Expected Shortfall (CVaR)"""
        if returns.size == 0:
            return 0.0
        var = RiskCalculator.calculate_historical_var(returns, confidence)
        tail = returns[returns <= var]
        return float(np.mean(tail)) if len(tail) > 0 else var

    @staticmethod
    def calculate_upside_capture(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Upside capture ratio vs benchmark"""
        if returns.size == 0 or benchmark_returns.size == 0:
            return 0.0
        
        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        upside_mask = benchmark_returns > 0
        if np.sum(upside_mask) == 0:
            return 0.0
        
        asset_upside = returns[upside_mask].sum()
        bench_upside = benchmark_returns[upside_mask].sum()
        
        return float(asset_upside / (bench_upside + 1e-8))

    @staticmethod
    def calculate_downside_capture(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Downside capture ratio vs benchmark"""
        if returns.size == 0 or benchmark_returns.size == 0:
            return 0.0
        
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        downside_mask = benchmark_returns < 0
        if np.sum(downside_mask) == 0:
            return 0.0
        
        asset_downside = returns[downside_mask].sum()
        bench_downside = benchmark_returns[downside_mask].sum()
        
        return float(asset_downside / (bench_downside + 1e-8))

    @staticmethod
    def calculate_all_metrics(
        self,
        paths: np.ndarray,
        asset_names: List[str],
        input_type: str = "returns",
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calculate all risk metrics for multiple assets"""
        
        if paths is None or len(paths) == 0:
            return {}
        
        if paths.ndim != 3:
            raise ValueError("paths must be (n_paths, seq_len, n_assets)")
        
        n_paths, seq_len, n_assets = paths.shape
        metrics = {}
        
        for i, asset in enumerate(asset_names[:n_assets]):
            
            if input_type == "returns":
                asset_returns = paths[:, :, i].flatten()
                price_paths = np.exp(np.cumsum(paths[:, :, i], axis=1))
            else:
                prices = paths[:, :, i]
                asset_returns = np.diff(np.log(prices + 1e-8), axis=1).flatten()
                price_paths = prices
            
            # Clean data
            asset_returns = asset_returns[np.isfinite(asset_returns)]
            
            if asset_returns.size == 0:
                continue
            
            # Calculate all metrics
            var_95 = self.calculate_var(asset_returns, 0.95)
            cvar_95 = self.calculate_cvar(asset_returns, 0.95)
            var_99 = self.calculate_var(asset_returns, 0.99)
            cvar_99 = self.calculate_cvar(asset_returns, 0.99)
            
            sharpe = self.calculate_sharpe(asset_returns)
            sortino = self.calculate_sortino(asset_returns)
            
            # Drawdown metrics
            dd = self.calculate_max_drawdown(np.mean(price_paths, axis=0))
            dd_duration = self.calculate_drawdown_duration(np.mean(price_paths, axis=0))
            
            calmar = self.calculate_calmar(asset_returns, dd)
            omega = self.calculate_omega_ratio(asset_returns)
            tail_ratio = self.calculate_tail_ratio(asset_returns)
            gain_pain = self.calculate_gain_to_pain_ratio(asset_returns)
            
            skewness = self.calculate_skewness(asset_returns)
            kurtosis = self.calculate_kurtosis(asset_returns)
            
            expected_return = np.mean(asset_returns) * 252
            volatility = np.std(asset_returns) * np.sqrt(252)
            
            # Benchmark comparison if provided
            if benchmark_returns is not None:
                upside_capture = self.calculate_upside_capture(asset_returns, benchmark_returns)
                downside_capture = self.calculate_downside_capture(asset_returns, benchmark_returns)
            else:
                upside_capture = 0.0
                downside_capture = 0.0
            
            metrics[asset] = {
                'var_95': float(var_95),
                'cvar_95': float(cvar_95),
                'var_99': float(var_99),
                'cvar_99': float(cvar_99),
                'sharpe': float(sharpe),
                'sortino': float(sortino),
                'calmar': float(calmar),
                'max_drawdown': float(dd),
                'max_drawdown_duration': float(dd_duration['max_duration']),
                'avg_drawdown_duration': float(dd_duration['avg_duration']),
                'volatility': float(volatility),
                'expected_return': float(expected_return),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'omega_ratio': float(omega),
                'tail_ratio': float(tail_ratio),
                'gain_to_pain_ratio': float(gain_pain),
                'upside_capture': float(upside_capture),
                'downside_capture': float(downside_capture)
            }
        
        return metrics


class RollingRiskCalculator:
    """
    Rolling window risk metrics calculator
    """
    
    def __init__(self, window: int = 252):
        self.window = window
    
    def calculate_rolling_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> np.ndarray:
        """Calculate rolling VaR"""
        var = np.zeros(len(returns))
        for i in range(self.window, len(returns)):
            window_returns = returns[i-self.window:i]
            var[i] = np.percentile(window_returns, (1 - confidence) * 100)
        return var
    
    def calculate_rolling_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> np.ndarray:
        """Calculate rolling Sharpe ratio"""
        sharpe = np.zeros(len(returns))
        for i in range(self.window, len(returns)):
            window_returns = returns[i-self.window:i]
            mean_daily = np.mean(window_returns)
            vol_daily = np.std(window_returns)
            if vol_daily > 0:
                annual_return = mean_daily * 252
                annual_vol = vol_daily * np.sqrt(252)
                sharpe[i] = (annual_return - risk_free_rate) / annual_vol
        return sharpe
    
    def calculate_rolling_volatility(
        self,
        returns: np.ndarray
    ) -> np.ndarray:
        """Calculate rolling volatility"""
        vol = np.zeros(len(returns))
        for i in range(self.window, len(returns)):
            window_returns = returns[i-self.window:i]
            vol[i] = np.std(window_returns) * np.sqrt(252)
        return vol