"""
Advanced Performance Analytics Module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalytics:
    """
    Advanced performance analytics for portfolio analysis
    """
    
    @staticmethod
    def rolling_alpha_beta(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        window: int = 252
    ) -> Dict[str, np.ndarray]:
        """Calculate rolling alpha and beta"""
        n = len(returns)
        alpha = np.zeros(n)
        beta = np.zeros(n)
        r_squared = np.zeros(n)
        
        for i in range(window, n):
            y = returns[i-window:i]
            x = benchmark_returns[i-window:i]
            
            # Simple linear regression
            cov = np.cov(x, y)[0, 1]
            var_x = np.var(x)
            
            if var_x > 0:
                beta[i] = cov / var_x
                alpha[i] = np.mean(y) - beta[i] * np.mean(x)
                
                # R-squared
                y_pred = beta[i] * x + alpha[i]
                residuals = y - y_pred
                ss_total = np.sum((y - np.mean(y)) ** 2)
                ss_residual = np.sum(residuals ** 2)
                r_squared[i] = 1 - (ss_residual / (ss_total + 1e-8))
        
        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared
        }
    
    @staticmethod
    def information_ratio(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        annualize: bool = True
    ) -> float:
        """Information ratio (active return / tracking error)"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        active_returns = returns - benchmark_returns
        mean_active = np.mean(active_returns)
        tracking_error = np.std(active_returns)
        
        if annualize:
            mean_active = mean_active * 252
            tracking_error = tracking_error * np.sqrt(252)
        
        return float(mean_active / (tracking_error + 1e-8))
    
    @staticmethod
    def treynor_ratio(
        returns: np.ndarray,
        beta: float,
        risk_free_rate: float = 0.02
    ) -> float:
        """Treynor ratio (return / beta)"""
        if len(returns) == 0 or beta == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        return float((annual_return - risk_free_rate) / (beta + 1e-8))
    
    @staticmethod
    def jensens_alpha(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        beta: float,
        risk_free_rate: float = 0.02
    ) -> float:
        """Jensen's alpha"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        expected_return = risk_free_rate/252 + beta * (np.mean(benchmark_returns) - risk_free_rate/252)
        alpha = np.mean(returns) - expected_return
        
        return float(alpha * 252)  # Annualized
    
    @staticmethod
    def capture_ratios(
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict[str, float]:
        """Calculate up and down capture ratios"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return {'up_capture': 0.0, 'down_capture': 0.0}
        
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        up_mask = benchmark_returns > 0
        down_mask = benchmark_returns < 0
        
        if np.sum(up_mask) == 0:
            up_capture = 0.0
        else:
            up_capture = returns[up_mask].sum() / (benchmark_returns[up_mask].sum() + 1e-8)
        
        if np.sum(down_mask) == 0:
            down_capture = 0.0
        else:
            down_capture = returns[down_mask].sum() / (benchmark_returns[down_mask].sum() + 1e-8)
        
        return {
            'up_capture': float(up_capture),
            'down_capture': float(down_capture)
        }
    
    @staticmethod
    def ulcer_index(returns: np.ndarray) -> float:
        """Ulcer index (measure of downside risk)"""
        if len(returns) == 0:
            return 0.0
        
        # Convert to prices
        prices = np.exp(np.cumsum(returns))
        
        # Calculate drawdowns
        peak = np.maximum.accumulate(prices)
        drawdowns = (prices - peak) / (peak + 1e-8)
        
        # Ulcer index = sqrt(mean of squared drawdowns)
        ulcer = np.sqrt(np.mean(drawdowns ** 2))
        
        return float(ulcer)
    
    @staticmethod
    def martin_ratio(returns: np.ndarray) -> float:
        """Martin ratio (return / ulcer index)"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        ulcer = PerformanceAnalytics.ulcer_index(returns)
        
        return float(annual_return / (ulcer + 1e-8))
    
    @staticmethod
    def downside_deviation(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Downside deviation"""
        if len(returns) == 0:
            return 0.0
        
        threshold_daily = threshold / 252
        downside = returns[returns < threshold_daily]
        
        if len(downside) == 0:
            return 0.0
        
        deviation = np.sqrt(np.mean((downside - threshold_daily) ** 2))
        return float(deviation * np.sqrt(252))
    
    @staticmethod
    def upside_potential(returns: np.ndarray, threshold: float = 0.0) -> float:
        """Upside potential"""
        if len(returns) == 0:
            return 0.0
        
        threshold_daily = threshold / 252
        upside = returns[returns > threshold_daily]
        
        if len(upside) == 0:
            return 0.0
        
        potential = np.sqrt(np.mean((upside - threshold_daily) ** 2))
        return float(potential * np.sqrt(252))
    
    @staticmethod
    def kelly_criterion(returns: np.ndarray) -> float:
        """Kelly criterion optimal betting fraction"""
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        variance = np.var(returns)
        
        if variance == 0:
            return 0.0
        
        kelly = mean_return / variance
        return float(min(kelly, 1.0))  # Cap at 100%
    
    @staticmethod
    def pain_index(returns: np.ndarray) -> float:
        """Pain index (average drawdown)"""
        if len(returns) == 0:
            return 0.0
        
        prices = np.exp(np.cumsum(returns))
        peak = np.maximum.accumulate(prices)
        drawdowns = (prices - peak) / (peak + 1e-8)
        
        return float(np.mean(drawdowns[drawdowns < 0]))
    
    @staticmethod
    def pain_ratio(returns: np.ndarray) -> float:
        """Pain ratio (return / pain index)"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        pain = abs(PerformanceAnalytics.pain_index(returns))
        
        return float(annual_return / (pain + 1e-8))
    
    @staticmethod
    def calculate_all_performance_metrics(
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """Calculate all performance metrics"""
        
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        metrics = {
            'annual_return': float(np.mean(returns) * 252),
            'volatility': float(np.std(returns) * np.sqrt(252)),
            'sharpe': float((np.mean(returns) * 252 - risk_free_rate) / (np.std(returns) * np.sqrt(252) + 1e-8)),
            'skewness': float(stats.skew(returns)),
            'kurtosis': float(stats.kurtosis(returns)),
        }
        
        # Drawdown metrics
        prices = np.exp(np.cumsum(returns))
        peak = np.maximum.accumulate(prices)
        drawdowns = (prices - peak) / (peak + 1e-8)
        
        metrics['max_drawdown'] = float(np.min(drawdowns))
        metrics['avg_drawdown'] = float(np.mean(drawdowns[drawdowns < 0]))
        metrics['ulcer_index'] = PerformanceAnalytics.ulcer_index(returns)
        metrics['pain_index'] = PerformanceAnalytics.pain_index(returns)
        
        # Risk-adjusted metrics
        metrics['sortino'] = PerformanceAnalytics.calculate_sortino(returns, risk_free_rate)
        metrics['calmar'] = metrics['annual_return'] / (abs(metrics['max_drawdown']) + 1e-8)
        metrics['martin_ratio'] = metrics['annual_return'] / (metrics['ulcer_index'] + 1e-8)
        metrics['pain_ratio'] = metrics['annual_return'] / (abs(metrics['pain_index']) + 1e-8)
        
        # Kelly criterion
        metrics['kelly_fraction'] = PerformanceAnalytics.kelly_criterion(returns)
        
        # Benchmark relative metrics
        if benchmark_returns is not None:
            min_len = min(len(returns), len(benchmark_returns))
            r = returns[:min_len]
            b = benchmark_returns[:min_len]
            
            metrics['information_ratio'] = PerformanceAnalytics.information_ratio(r, b)
            
            # Beta and alpha
            cov = np.cov(r, b)[0, 1]
            var_b = np.var(b)
            if var_b > 0:
                beta = cov / var_b
                metrics['beta'] = float(beta)
                metrics['alpha'] = float((np.mean(r) - risk_free_rate/252) - beta * (np.mean(b) - risk_free_rate/252)) * 252
                metrics['treynor'] = (metrics['annual_return'] - risk_free_rate) / (beta + 1e-8)
            
            # Capture ratios
            capture = PerformanceAnalytics.capture_ratios(r, b)
            metrics['up_capture'] = capture['up_capture']
            metrics['down_capture'] = capture['down_capture']
        
        return metrics
    
    @staticmethod
    def calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Sortino ratio (standalone)"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        downside = PerformanceAnalytics.downside_deviation(returns)
        
        return float((annual_return - risk_free_rate) / (downside + 1e-8))