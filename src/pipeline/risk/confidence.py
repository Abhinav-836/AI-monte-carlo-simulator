"""
Advanced Confidence Interval Calculations with Bootstrap
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Optional, Dict
import pandas as pd
from sklearn.utils import resample


class ConfidenceIntervalCalculator:
    """Advanced confidence interval calculator with multiple methods"""
    
    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic: str = 'mean',
        confidence: float = 0.95,
        n_bootstrap: int = 5000,
        method: str = 'percentile'
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval with multiple methods
        
        Methods:
            - 'percentile': Simple percentile method
            - 'bca': Bias-corrected and accelerated
            - 'normal': Normal approximation
            - 'basic': Basic bootstrap
        """
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        
        if len(data) < 2:
            return np.nan, np.nan
        
        # Define statistic function
        if statistic == 'mean':
            stat_func = np.mean
        elif statistic == 'median':
            stat_func = np.median
        elif statistic == 'var':
            stat_func = np.var
        elif statistic == 'std':
            stat_func = np.std
        elif callable(statistic):
            stat_func = statistic
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        
        n = len(data)
        alpha = 1 - confidence
        
        if method == 'percentile':
            # Simple percentile bootstrap
            bootstrap_stats = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, n, replace=True)
                bootstrap_stats.append(stat_func(sample))
            
            bootstrap_stats = np.array(bootstrap_stats)
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            
        elif method == 'bca':
            # BCa bootstrap (more accurate)
            bootstrap_stats = []
            jackknife_stats = []
            
            # Bootstrap samples
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, n, replace=True)
                bootstrap_stats.append(stat_func(sample))
            
            bootstrap_stats = np.array(bootstrap_stats)
            theta_hat = stat_func(data)
            
            # Jackknife for bias correction
            for i in range(n):
                jack_sample = np.delete(data, i)
                jackknife_stats.append(stat_func(jack_sample))
            
            jackknife_stats = np.array(jackknife_stats)
            theta_jack = np.mean(jackknife_stats)
            
            # Bias correction
            z0 = stats.norm.ppf(np.mean(bootstrap_stats < theta_hat))
            
            # Acceleration
            numerator = np.sum((theta_jack - jackknife_stats) ** 3)
            denominator = 6 * (np.sum((theta_jack - jackknife_stats) ** 2)) ** 1.5
            a = numerator / (denominator + 1e-8)
            
            # BCa quantiles
            z_alpha = stats.norm.ppf(alpha / 2)
            z_1_alpha = stats.norm.ppf(1 - alpha / 2)
            
            lower_quantile = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
            upper_quantile = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))
            
            lower = np.percentile(bootstrap_stats, lower_quantile * 100)
            upper = np.percentile(bootstrap_stats, upper_quantile * 100)
            
        elif method == 'normal':
            # Normal approximation
            bootstrap_stats = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, n, replace=True)
                bootstrap_stats.append(stat_func(sample))
            
            bootstrap_stats = np.array(bootstrap_stats)
            se = np.std(bootstrap_stats)
            theta_hat = stat_func(data)
            
            z_score = stats.norm.ppf(1 - alpha / 2)
            lower = theta_hat - z_score * se
            upper = theta_hat + z_score * se
            
        elif method == 'basic':
            # Basic bootstrap
            bootstrap_stats = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, n, replace=True)
                bootstrap_stats.append(stat_func(sample))
            
            bootstrap_stats = np.array(bootstrap_stats)
            theta_hat = stat_func(data)
            
            lower = 2 * theta_hat - np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            upper = 2 * theta_hat - np.percentile(bootstrap_stats, 100 * (alpha / 2))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return float(lower), float(upper)

    @staticmethod
    def parametric_ci(
        data: np.ndarray,
        confidence: float = 0.95,
        distribution: str = 'normal'
    ) -> Tuple[float, float]:
        """Parametric confidence interval"""
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        
        n = len(data)
        if n < 2:
            return np.nan, np.nan
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(n)
        alpha = 1 - confidence
        
        if distribution == 'normal':
            z = stats.norm.ppf(1 - alpha / 2)
            return mean - z * se, mean + z * se
        elif distribution == 't':
            t = stats.t.ppf(1 - alpha / 2, df=n - 1)
            return mean - t * se, mean + t * se
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    @staticmethod
    def prediction_interval(
        data: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Prediction interval for future observations"""
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        
        n = len(data)
        if n < 2:
            return np.nan, np.nan
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        alpha = 1 - confidence
        
        t = stats.t.ppf(1 - alpha / 2, df=n - 1)
        se_pred = std * np.sqrt(1 + 1 / n)
        
        return mean - t * se_pred, mean + t * se_pred

    @staticmethod
    def tolerance_interval(
        data: np.ndarray,
        confidence: float = 0.95,
        proportion: float = 0.95
    ) -> Tuple[float, float]:
        """Tolerance interval (contains proportion of population)"""
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        
        n = len(data)
        if n < 2:
            return np.nan, np.nan
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # Normal tolerance interval
        from scipy.stats import chi2, norm
        z_p = norm.ppf((1 + proportion) / 2)
        chi2_alpha = chi2.ppf(1 - confidence, df=n - 1)
        
        k = z_p * np.sqrt((n + 1) / (n * chi2_alpha / (n - 1)))
        
        return mean - k * std, mean + k * std

    @staticmethod
    def simultaneous_ci(
        data: np.ndarray,
        confidence: float = 0.95
    ) -> List[Tuple[float, float]]:
        """Simultaneous confidence intervals for multiple points"""
        data = np.asarray(data)
        
        if data.ndim != 2:
            raise ValueError("Data must be 2D (n_points, n_samples)")
        
        n_points = data.shape[0]
        alpha = 1 - confidence
        point_conf = 1 - alpha / n_points
        
        cis = []
        for row in data:
            row_clean = row[~np.isnan(row)]
            if len(row_clean) < 2:
                cis.append((np.nan, np.nan))
            else:
                ci = ConfidenceIntervalCalculator.parametric_ci(row_clean, point_conf)
                cis.append(ci)
        
        return cis

    @staticmethod
    def fan_chart(
        paths: np.ndarray,
        percentiles: Optional[List[float]] = None,
        confidence_levels: Optional[List[float]] = None
    ) -> Dict:
        """Fan chart for visualizing uncertainty"""
        paths = np.asarray(paths)
        
        if paths.ndim != 2:
            raise ValueError("Paths must be 2D (n_paths, time_steps)")
        
        if percentiles is None and confidence_levels is None:
            percentiles = [5, 25, 50, 75, 95]
        
        result = {
            'median': np.median(paths, axis=0),
            'mean': np.mean(paths, axis=0),
            'std': np.std(paths, axis=0)
        }
        
        if percentiles:
            for p in percentiles:
                result[f'p{p}'] = np.percentile(paths, p, axis=0)
        
        if confidence_levels:
            for level in confidence_levels:
                alpha = 1 - level
                lower = np.percentile(paths, 100 * alpha / 2, axis=0)
                upper = np.percentile(paths, 100 * (1 - alpha / 2), axis=0)
                result[f'ci_{level}'] = {
                    'lower': lower,
                    'upper': upper
                }
        
        return result

    @staticmethod
    def probability_of_exceedance(
        paths: np.ndarray,
        threshold: float,
        final_step: bool = True
    ) -> float:
        """Probability of exceeding a threshold"""
        paths = np.asarray(paths)
        
        if paths.size == 0:
            return np.nan
        
        if final_step:
            values = paths[:, -1]
        else:
            values = paths.flatten()
        
        return float(np.mean(values > threshold))

    @staticmethod
    def time_to_exceedance(
        paths: np.ndarray,
        threshold: float,
        upper: bool = True
    ) -> Dict:
        """Distribution of time until threshold is hit"""
        paths = np.asarray(paths)
        
        if paths.ndim != 2:
            raise ValueError("Paths must be 2D (n_paths, time_steps)")
        
        hit_times = []
        
        for path in paths:
            if upper:
                hit_idx = np.where(path > threshold)[0]
            else:
                hit_idx = np.where(path < threshold)[0]
            
            if len(hit_idx) > 0:
                hit_times.append(hit_idx[0])
            else:
                hit_times.append(np.nan)
        
        hit_times = np.array(hit_times)
        hit_times = hit_times[~np.isnan(hit_times)]
        
        if len(hit_times) == 0:
            return {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'percentiles': {}
            }
        
        return {
            'mean': float(np.mean(hit_times)),
            'median': float(np.median(hit_times)),
            'std': float(np.std(hit_times)),
            'percentiles': {
                '25': float(np.percentile(hit_times, 25)),
                '50': float(np.percentile(hit_times, 50)),
                '75': float(np.percentile(hit_times, 75))
            }
        }

    @staticmethod
    def confidence_interval_string(
        lower: float,
        upper: float,
        decimals: int = 2,
        prefix: str = "[",
        suffix: str = "]"
    ) -> str:
        """Format confidence interval as string"""
        if np.isnan(lower) or np.isnan(upper):
            return "N/A"
        
        return f"{prefix}{lower:.{decimals}f}, {upper:.{decimals}f}{suffix}"