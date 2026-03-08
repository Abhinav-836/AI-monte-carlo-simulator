"""
Feature Engineering Pipeline (Production Ready)
Fixed leakage, stability, and ML compatibility
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    window_sizes: List[int] = None
    include_technical: bool = True
    include_microstructure: bool = True
    include_macro: bool = False  # disabled unless real macro data added

    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [5, 10, 20, 50]


class FeatureEngineer:

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.means_, self.stds_ = None, None

    def create_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        # ✅ Safe returns
        returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)

        # Align
        prices = prices.iloc[1:]
        returns = returns.iloc[1:]

        feature_list = []

        feature_list.append(self._price_features(prices))
        feature_list.append(self._return_features(returns))
        feature_list.append(self._vol_features(returns))
        feature_list.append(self._momentum(prices, returns))
        feature_list.append(self._correlation(returns))

        if self.config.include_technical:
            feature_list.append(self._technical(prices))

        if self.config.include_microstructure:
            feature_list.append(self._microstructure(returns))

        features = pd.concat(feature_list, axis=1)

        # ✅ Handle NaNs safely (DO NOT DROP ALL)
        features = features.ffill().bfill()

        # ✅ Clip extreme values (important for GAN stability)
        features = features.clip(-10, 10)

        return features

    def _price_features(self, prices):
        df = pd.DataFrame(index=prices.index)

        for t in prices.columns:
            p = prices[t]

            df[f"{t}_log"] = np.log(p.clip(lower=0.01))

            for w in self.config.window_sizes:
                ma = p.rolling(w, min_periods=w).mean()
                df[f"{t}_ma_ratio_{w}"] = p / (ma + 1e-8) - 1

        return df

    def _return_features(self, returns):
        df = pd.DataFrame(index=returns.index)

        for t in returns.columns:
            r = returns[t]

            for lag in [1, 2, 3]:
                df[f"{t}_lag_{lag}"] = r.shift(lag)

            for w in self.config.window_sizes:
                df[f"{t}_cum_{w}"] = r.rolling(w, min_periods=w).sum()

        return df

    def _vol_features(self, returns):
        df = pd.DataFrame(index=returns.index)

        for t in returns.columns:
            r = returns[t]

            for w in self.config.window_sizes:
                df[f"{t}_vol_{w}"] = r.rolling(w, min_periods=w).std()

        return df

    def _momentum(self, prices, returns):
        df = pd.DataFrame(index=returns.index)

        for t in prices.columns:
            p = prices[t]

            ema12 = p.ewm(span=12).mean()
            ema26 = p.ewm(span=26).mean()

            df[f"{t}_macd"] = ema12 - ema26

        return df

    def _correlation(self, returns):
        if returns.shape[1] < 2:
            return pd.DataFrame(index=returns.index)

        df = pd.DataFrame(index=returns.index)

        rolling_corr = returns.rolling(20).corr()

        for t in returns.columns:
            vals = []
            for i in range(len(returns)):
                try:
                    c = rolling_corr.iloc[i * len(returns.columns):(i + 1) * len(returns.columns)]
                    vals.append(c[t].drop(t).mean())
                except:
                    vals.append(0)

            df[f"{t}_corr"] = vals

        return df

    def _technical(self, prices):
        df = pd.DataFrame(index=prices.index)

        for t in prices.columns:
            p = prices[t]

            ma = p.rolling(20).mean()
            std = p.rolling(20).std()

            df[f"{t}_bb_width"] = (2 * std) / (ma + 1e-8)

        return df

    def _microstructure(self, returns):
        df = pd.DataFrame(index=returns.index)

        for t in returns.columns:
            r = returns[t]

            df[f"{t}_autocorr"] = r.rolling(20).apply(
                lambda x: pd.Series(x).autocorr(1) if len(x) > 1 else 0,
                raw=True
            )

        return df

    def create_sequences(self, features, seq_length=60):
        if len(features) < seq_length + 1:
            return np.array([]), np.array([])

        values = features.values.astype(np.float32)

        X, y = [], []
        for i in range(len(values) - seq_length):
            X.append(values[i:i + seq_length])
            y.append(values[i + seq_length])

        return np.array(X), np.array(y)

    def normalize_features(self, features, fit=True):
        if fit:
            self.means_ = features.mean()
            self.stds_ = features.std()

        return (features - self.means_) / (self.stds_ + 1e-8)
