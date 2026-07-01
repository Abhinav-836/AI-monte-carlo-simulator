"""
Advanced Feature Engineering Pipeline with Technical Indicators
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging
from scipy import signal
from scipy.stats import zscore

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Advanced feature engineering configuration"""
    window_sizes: List[int] = None
    include_technical: bool = True
    include_microstructure: bool = True
    include_macro: bool = False
    include_momentum: bool = True
    include_volatility: bool = True
    include_correlation: bool = True
    include_cyclical: bool = True
    include_fractal: bool = False
    n_technical_indicators: int = 20
    use_frequency_domain: bool = False
    
    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [5, 10, 20, 50, 100]


class FeatureEngineer:
    """
    Advanced feature engineering with technical indicators and frequency domain
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.means_ = None
        self.stds_ = None
        self.scaler = None
        
    def create_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        if prices.empty:
            return pd.DataFrame()
        
        # Safe returns
        returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
        prices = prices.iloc[1:]
        returns = returns.iloc[1:]
        
        feature_list = []
        
        # Core features
        feature_list.append(self._price_features(prices))
        feature_list.append(self._return_features(returns))
        feature_list.append(self._volatility_features(returns))
        
        # Advanced features
        if self.config.include_momentum:
            feature_list.append(self._momentum_features(prices, returns))
        
        if self.config.include_technical:
            feature_list.append(self._technical_indicators(prices, returns))
        
        if self.config.include_correlation:
            feature_list.append(self._correlation_features(returns))
        
        if self.config.include_microstructure:
            feature_list.append(self._microstructure_features(returns))
        
        if self.config.include_cyclical:
            feature_list.append(self._cyclical_features(returns))
        
        if self.config.include_fractal:
            feature_list.append(self._fractal_features(prices))
        
        if self.config.use_frequency_domain:
            feature_list.append(self._frequency_features(returns))
        
        # Combine
        features = pd.concat(feature_list, axis=1)
        
        # Handle missing values
        features = features.ffill().bfill()
        
        # Clip outliers
        features = features.clip(-10, 10)
        
        # Add interaction features
        features = self._add_interaction_features(features)
        
        return features
    
    def _price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Price-based features"""
        df = pd.DataFrame(index=prices.index)
        
        for t in prices.columns:
            p = prices[t]
            
            # Log price
            df[f"{t}_log"] = np.log(p.clip(lower=0.01))
            
            # Normalized price (relative to recent range)
            for w in self.config.window_sizes:
                min_p = p.rolling(w, min_periods=w).min()
                max_p = p.rolling(w, min_periods=w).max()
                range_p = max_p - min_p
                df[f"{t}_norm_{w}"] = (p - min_p) / (range_p + 1e-8)
            
            # Price to moving average
            for w in self.config.window_sizes:
                ma = p.rolling(w, min_periods=w).mean()
                df[f"{t}_ma_ratio_{w}"] = p / (ma + 1e-8) - 1
            
            # Exponential moving averages
            for span in [12, 26, 50]:
                ema = p.ewm(span=span).mean()
                df[f"{t}_ema_{span}"] = p / (ema + 1e-8) - 1
            
            # Price acceleration
            df[f"{t}_accel"] = p.diff().diff()
        
        return df
    
    def _return_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Return-based features"""
        df = pd.DataFrame(index=returns.index)
        
        for t in returns.columns:
            r = returns[t]
            
            # Lags
            for lag in [1, 2, 3, 5, 10]:
                df[f"{t}_lag_{lag}"] = r.shift(lag)
            
            # Rolling sums
            for w in self.config.window_sizes:
                df[f"{t}_sum_{w}"] = r.rolling(w, min_periods=w).sum()
            
            # Rolling mean of absolute returns
            for w in self.config.window_sizes:
                df[f"{t}_abs_mean_{w}"] = r.abs().rolling(w, min_periods=w).mean()
            
            # Sign of returns
            df[f"{t}_sign"] = np.sign(r)
            
            # Returns rank
            df[f"{t}_rank"] = r.rolling(20).rank(pct=True)
        
        return df
    
    def _volatility_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Volatility-based features"""
        df = pd.DataFrame(index=returns.index)
        
        for t in returns.columns:
            r = returns[t]
            
            # Rolling volatility
            for w in self.config.window_sizes:
                df[f"{t}_vol_{w}"] = r.rolling(w, min_periods=w).std()
            
            # High-low volatility
            df[f"{t}_high_low"] = r.rolling(20).max() - r.rolling(20).min()
            
            # Volatility ratio (short/long)
            df[f"{t}_vol_ratio"] = (r.rolling(5).std() / (r.rolling(20).std() + 1e-8))
            
            # Parkinson volatility (requires high/low prices)
            # This is a proxy using returns range
            df[f"{t}_parkinson"] = r.rolling(20).apply(
                lambda x: np.sqrt(0.5 * np.mean((x - x.mean())**2))
            )
        
        return df
    
    def _momentum_features(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Momentum-based features"""
        df = pd.DataFrame(index=returns.index)
        
        for t in prices.columns:
            p = prices[t]
            r = returns[t]
            
            # ROC (Rate of Change)
            for period in [10, 20, 50]:
                df[f"{t}_roc_{period}"] = (p / p.shift(period) - 1) * 100
            
            # RSI (Relative Strength Index)
            df[f"{t}_rsi"] = self._calculate_rsi(r)
            
            # MACD
            macd, signal_line = self._calculate_macd(p)
            df[f"{t}_macd"] = macd
            df[f"{t}_macd_signal"] = signal_line
            df[f"{t}_macd_hist"] = macd - signal_line
            
            # Momentum indicator
            df[f"{t}_momentum"] = p - p.shift(10)
            
            # Williams %R
            df[f"{t}_williams_r"] = self._calculate_williams_r(p)
            
            # Aroon indicators
            df[f"{t}_aroon_up"], df[f"{t}_aroon_down"] = self._calculate_aroon(p)
        
        return df
    
    def _technical_indicators(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive technical indicators"""
        df = pd.DataFrame(index=returns.index)
        
        for t in prices.columns:
            p = prices[t]
            r = returns[t]
            
            # Bollinger Bands
            ma20 = p.rolling(20).mean()
            std20 = p.rolling(20).std()
            df[f"{t}_bb_upper"] = ma20 + 2 * std20
            df[f"{t}_bb_lower"] = ma20 - 2 * std20
            df[f"{t}_bb_width"] = (2 * std20) / (ma20 + 1e-8)
            df[f"{t}_bb_position"] = (p - ma20) / (2 * std20 + 1e-8)
            
            # Stochastic Oscillator
            df[f"{t}_stoch"] = self._calculate_stochastic(p)
            
            # ADX (Average Directional Index)
            df[f"{t}_adx"] = self._calculate_adx(p)
            
            # Ichimoku Cloud
            df[f"{t}_ichimoku_tenkan"] = self._calculate_ichimoku_tenkan(p)
            df[f"{t}_ichimoku_kijun"] = self._calculate_ichimoku_kijun(p)
            
            # CCI (Commodity Channel Index)
            df[f"{t}_cci"] = self._calculate_cci(p)
            
            # Money Flow Index
            df[f"{t}_mfi"] = self._calculate_mfi(p, returns)
            
            # Chaikin Money Flow
            df[f"{t}_cmf"] = self._calculate_cmf(p)
            
            # Ease of Movement
            df[f"{t}_eom"] = self._calculate_eom(p)
        
        return df
    
    def _correlation_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Correlation-based features"""
        if returns.shape[1] < 2:
            return pd.DataFrame(index=returns.index)
        
        df = pd.DataFrame(index=returns.index)
        
        # Rolling correlation to market average
        market_avg = returns.mean(axis=1)
        
        for t in returns.columns:
            df[f"{t}_corr_market"] = returns[t].rolling(20).corr(market_avg)
        
        # Pairwise correlations
        for i, t1 in enumerate(returns.columns):
            for t2 in returns.columns[i+1:]:
                df[f"{t1}_{t2}_corr"] = returns[t1].rolling(20).corr(returns[t2])
        
        return df
    
    def _microstructure_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features"""
        df = pd.DataFrame(index=returns.index)
        
        for t in returns.columns:
            r = returns[t]
            
            # Autocorrelation
            for lag in [1, 2, 5]:
                df[f"{t}_autocorr_{lag}"] = r.rolling(20).apply(
                    lambda x: pd.Series(x).autocorr(lag) if len(x) > lag else 0,
                    raw=True
                )
            
            # Serial correlation of absolute returns
            df[f"{t}_abs_autocorr"] = r.abs().rolling(20).apply(
                lambda x: pd.Series(x).autocorr(1) if len(x) > 1 else 0,
                raw=True
            )
            
            # Hurst exponent proxy
            df[f"{t}_hurst"] = self._calculate_hurst_exponent(r)
            
            # Variance ratio
            df[f"{t}_var_ratio"] = r.rolling(10).var() / (r.rolling(30).var() + 1e-8)
            
            # Tick-by-tick volatility
            df[f"{t}_tick_vol"] = r.abs().rolling(5).sum()
            
            # Signed volume (requires volume data - using returns magnitude as proxy)
            df[f"{t}_signed_volume"] = r * r.abs().rolling(10).mean()
        
        return df
    
    def _cyclical_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Cyclical/seasonal features"""
        df = pd.DataFrame(index=returns.index)
        
        # Day of week
        df['day_of_week'] = returns.index.dayofweek
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month
        df['month'] = returns.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Quarter
        df['quarter'] = returns.index.quarter
        
        # Year progress
        df['year_progress'] = returns.index.dayofyear / 365
        
        # Volatility seasonality
        for t in returns.columns:
            r = returns[t]
            df[f"{t}_vol_by_dow"] = r.groupby(r.index.dayofweek).transform('std')
            df[f"{t}_vol_by_month"] = r.groupby(r.index.month).transform('std')
        
        return df
    
    def _fractal_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Fractal dimension features"""
        df = pd.DataFrame(index=prices.index)
        
        for t in prices.columns:
            p = prices[t].values
            
            # Higuchi fractal dimension
            df[f"{t}_fractal_dim"] = self._calculate_fractal_dimension(p)
            
            # R/S analysis (Hurst exponent)
            df[f"{t}_rs_hurst"] = self._calculate_rs_hurst(p)
            
            # Box counting dimension
            df[f"{t}_box_dim"] = self._calculate_box_dimension(p)
        
        return df
    
    def _frequency_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Frequency domain features"""
        df = pd.DataFrame(index=returns.index)
        
        for t in returns.columns:
            r = returns[t].values
            
            # FFT spectral power
            fft_vals = np.fft.fft(r)
            spectral_power = np.abs(fft_vals) ** 2
            
            # Dominant frequency
            if len(spectral_power) > 2:
                dominant_idx = np.argmax(spectral_power[1:])
                df[f"{t}_dominant_freq"] = dominant_idx / len(spectral_power)
            
            # Spectral entropy
            power_norm = spectral_power / (np.sum(spectral_power) + 1e-8)
            df[f"{t}_spectral_entropy"] = -np.sum(power_norm * np.log(power_norm + 1e-8))
        
        return df
    
    def _add_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add interaction terms between features"""
        df = features.copy()
        
        # Select top features by variance
        variances = features.var()
        top_features = variances.nlargest(10).index.tolist()
        
        # Add interactions between top features
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                f1 = top_features[i]
                f2 = top_features[j]
                df[f"{f1}_{f2}_interact"] = features[f1] * features[f2]
        
        return df
    
    # ===== INDICATOR CALCULATIONS =====
    
    @staticmethod
    def _calculate_rsi(returns: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        gain = returns.clip(lower=0)
        loss = returns.clip(upper=0).abs()
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def _calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """MACD and signal line"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    @staticmethod
    def _calculate_williams_r(prices: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        high = prices.rolling(period).max()
        low = prices.rolling(period).min()
        return -100 * (high - prices) / (high - low + 1e-8)
    
    @staticmethod
    def _calculate_aroon(prices: pd.Series, period: int = 25) -> Tuple[pd.Series, pd.Series]:
        """Aroon indicators"""
        high_period = prices.rolling(period).max()
        low_period = prices.rolling(period).min()
        
        aroon_up = 100 * (period - (prices.rolling(period).apply(lambda x: x.argmax()))) / period
        aroon_down = 100 * (period - (prices.rolling(period).apply(lambda x: x.argmin()))) / period
        
        return aroon_up, aroon_down
    
    @staticmethod
    def _calculate_stochastic(prices: pd.Series, period: int = 14) -> pd.Series:
        """Stochastic Oscillator"""
        high = prices.rolling(period).max()
        low = prices.rolling(period).min()
        stoch = 100 * (prices - low) / (high - low + 1e-8)
        return stoch
    
    @staticmethod
    def _calculate_adx(prices: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index (simplified)"""
        # Simplified ADX using price movements
        movement = prices.diff()
        tr = prices.rolling(period).apply(lambda x: x.max() - x.min())
        
        positive_movement = movement.clip(lower=0)
        negative_movement = movement.clip(upper=0).abs()
        
        dm_plus = positive_movement.rolling(period).mean()
        dm_minus = negative_movement.rolling(period).mean()
        
        dx = 100 * (dm_plus - dm_minus).abs() / (dm_plus + dm_minus + 1e-8)
        adx = dx.rolling(period).mean()
        
        return adx
    
    @staticmethod
    def _calculate_ichimoku_tenkan(prices: pd.Series, period: int = 9) -> pd.Series:
        """Ichimoku Tenkan-sen"""
        high = prices.rolling(period).max()
        low = prices.rolling(period).min()
        return (high + low) / 2
    
    @staticmethod
    def _calculate_ichimoku_kijun(prices: pd.Series, period: int = 26) -> pd.Series:
        """Ichimoku Kijun-sen"""
        high = prices.rolling(period).max()
        low = prices.rolling(period).min()
        return (high + low) / 2
    
    @staticmethod
    def _calculate_cci(prices: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = prices  # Typical price (approximation)
        ma = tp.rolling(period).mean()
        mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (tp - ma) / (0.015 * mean_dev + 1e-8)
        return cci
    
    @staticmethod
    def _calculate_mfi(prices: pd.Series, returns: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = prices
        money_flow = typical_price * returns.abs()
        
        positive_flow = money_flow.where(returns > 0, 0)
        negative_flow = money_flow.where(returns < 0, 0)
        
        pos_sum = positive_flow.rolling(period).sum()
        neg_sum = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + pos_sum / (neg_sum + 1e-8)))
        return mfi
    
    @staticmethod
    def _calculate_cmf(prices: pd.Series, period: int = 20) -> pd.Series:
        """Chaikin Money Flow (simplified)"""
        # Simplified version using price changes
        money_flow = prices * prices.pct_change()
        return money_flow.rolling(period).sum()
    
    @staticmethod
    def _calculate_eom(prices: pd.Series, period: int = 14) -> pd.Series:
        """Ease of Movement (simplified)"""
        distance = prices.diff()
        eom = distance / prices.rolling(period).std()
        return eom
    
    @staticmethod
    def _calculate_hurst_exponent(returns: pd.Series, max_lag: int = 20) -> float:
        """Hurst exponent (simplified)"""
        if len(returns) < max_lag + 1:
            return 0.5
        
        try:
            # Calculate log returns
            log_returns = np.log(1 + returns)
            
            # R/S analysis
            rs_values = []
            lags = range(10, min(max_lag, len(log_returns) // 2))
            
            for lag in lags:
                if lag < 2:
                    continue
                
                # Split into windows
                n_windows = len(log_returns) // lag
                if n_windows < 2:
                    continue
                
                rs_window = []
                for i in range(n_windows):
                    window = log_returns[i*lag:(i+1)*lag]
                    if len(window) < 2:
                        continue
                    
                    mean = np.mean(window)
                    cumsum = np.cumsum(window - mean)
                    r = np.max(cumsum) - np.min(cumsum)
                    s = np.std(window, ddof=1)
                    
                    if s > 0:
                        rs_window.append(r / s)
                
                if rs_window:
                    rs_values.append(np.mean(rs_window))
            
            if len(rs_values) < 2:
                return 0.5
            
            # Fit power law
            lags = list(range(10, min(max_lag, len(log_returns) // 2)))
            lags = lags[:len(rs_values)]
            
            hurst = np.polyfit(np.log(lags), np.log(rs_values), 1)[0]
            return float(np.clip(hurst, 0, 1))
            
        except:
            return 0.5
    
    @staticmethod
    def _calculate_fractal_dimension(prices: np.ndarray) -> float:
        """Higuchi fractal dimension"""
        if len(prices) < 10:
            return 1.0
        
        try:
            max_k = min(10, len(prices) // 2)
            L = []
            k_values = []
            
            for k in range(1, max_k + 1):
                Lmk = []
                for m in range(k):
                    if m + k < len(prices):
                        sum_diff = np.sum(np.abs(
                            prices[m + k * i] - prices[m + k * (i - 1)]
                        ) for i in range(1, (len(prices) - m) // k + 1))
                        
                        if (len(prices) - m - 1) // k > 0:
                            Lmk.append(sum_diff / k * (len(prices) - 1) / ((len(prices) - m) // k * k))
                
                if Lmk:
                    L.append(np.mean(Lmk))
                    k_values.append(k)
            
            if len(L) < 2:
                return 1.0
            
            # Fit power law
            coef = np.polyfit(np.log(k_values), np.log(L), 1)
            dimension = -coef[0]
            return float(np.clip(dimension, 0, 2))
            
        except:
            return 1.0
    
    @staticmethod
    def _calculate_rs_hurst(prices: np.ndarray) -> float:
        """R/S Hurst exponent"""
        if len(prices) < 10:
            return 0.5
        
        try:
            # Log returns
            log_returns = np.diff(np.log(prices + 1e-8))
            
            n = len(log_returns)
            max_lag = min(20, n // 2)
            
            rs_values = []
            lags = []
            
            for lag in range(5, max_lag + 1):
                n_windows = n // lag
                if n_windows < 2:
                    continue
                
                rs_window = []
                for i in range(n_windows):
                    window = log_returns[i*lag:(i+1)*lag]
                    if len(window) < 2:
                        continue
                    
                    mean = np.mean(window)
                    cumsum = np.cumsum(window - mean)
                    r = np.max(cumsum) - np.min(cumsum)
                    s = np.std(window, ddof=1)
                    
                    if s > 0:
                        rs_window.append(r / s)
                
                if rs_window:
                    rs_values.append(np.mean(rs_window))
                    lags.append(lag)
            
            if len(rs_values) < 2:
                return 0.5
            
            # Fit power law
            hurst = np.polyfit(np.log(lags), np.log(rs_values), 1)[0]
            return float(np.clip(hurst, 0, 1))
            
        except:
            return 0.5
    
    @staticmethod
    def _calculate_box_dimension(prices: np.ndarray) -> float:
        """Box counting dimension"""
        if len(prices) < 10:
            return 1.0
        
        try:
            # Normalize
            prices_norm = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8)
            
            # Count boxes at different scales
            scales = []
            counts = []
            
            for scale in range(2, min(20, len(prices) // 2)):
                n_boxes = 0
                for i in range(0, len(prices_norm) - scale, scale):
                    window = prices_norm[i:i+scale]
                    if np.max(window) - np.min(window) > 1/scale:
                        n_boxes += 1
                
                if n_boxes > 0:
                    scales.append(np.log(1/scale))
                    counts.append(np.log(n_boxes))
            
            if len(scales) < 2:
                return 1.0
            
            # Fit linear regression
            coef = np.polyfit(scales, counts, 1)
            dimension = coef[0]
            return float(np.clip(dimension, 0, 2))
            
        except:
            return 1.0
    
    def create_sequences(self, features: pd.DataFrame, seq_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models"""
        if len(features) < seq_length + 1:
            return np.array([]), np.array([])
        
        values = features.values.astype(np.float32)
        
        X, y = [], []
        for i in range(len(values) - seq_length):
            X.append(values[i:i + seq_length])
            y.append(values[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def normalize_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features with robust scaling"""
        if fit:
            self.means_ = features.median()
            self.stds_ = features.quantile(0.75) - features.quantile(0.25)
        
        # Robust scaling
        normalized = (features - self.means_) / (self.stds_ + 1e-8)
        return normalized.clip(-5, 5)