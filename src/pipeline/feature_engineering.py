"""
Feature Engineering Pipeline with Technical Indicators (leak-free, scale-safe)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
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
    max_pairwise_assets: int = 8  # pairwise-correlation columns grow O(n^2)

    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [5, 10, 20, 50, 100]


class FeatureEngineer:
    """
    NOTE: features are NOT clipped here any more (the old clip(-10, 10) flattened RSI, stochastic,
    Bollinger etc.). Call normalize_features(features, fit=True) on the training split.
    Every feature at time t only uses data up to t.
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.means_ = None
        self.stds_ = None
        self.scaler = None
        self.interaction_features_: Optional[List[str]] = None  # frozen after first fit

    # ------------------------------------------------------------
    def create_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        prices = prices.astype(float).sort_index()
        returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
        prices, returns = prices.iloc[1:], returns.iloc[1:]

        parts = [self._price_features(prices), self._return_features(returns),
                 self._volatility_features(returns)]
        if self.config.include_momentum:
            parts.append(self._momentum_features(prices, returns))
        if self.config.include_technical:
            parts.append(self._technical_indicators(prices, returns))
        if self.config.include_correlation:
            parts.append(self._correlation_features(returns))
        if self.config.include_microstructure:
            parts.append(self._microstructure_features(returns))
        if self.config.include_cyclical:
            parts.append(self._cyclical_features(returns))
        if self.config.include_fractal:
            parts.append(self._fractal_features(prices))
        if self.config.use_frequency_domain:
            parts.append(self._frequency_features(returns))

        features = pd.concat(parts, axis=1)
        # forward-fill only (bfill would copy FUTURE values into the warm-up rows)
        features = features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        return self._add_interaction_features(features)

    # ------------------------------------------------------------
    def _price_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for t in prices.columns:
            p = prices[t]
            cols[f"{t}_log"] = np.log(p.clip(lower=0.01))
            for w in self.config.window_sizes:
                mn, mx = p.rolling(w, min_periods=w).min(), p.rolling(w, min_periods=w).max()
                cols[f"{t}_norm_{w}"] = (p - mn) / (mx - mn + 1e-8)
                cols[f"{t}_ma_ratio_{w}"] = p / (p.rolling(w, min_periods=w).mean() + 1e-8) - 1
            for span in (12, 26, 50):
                cols[f"{t}_ema_{span}"] = p / (p.ewm(span=span).mean() + 1e-8) - 1
            cols[f"{t}_accel"] = p.diff().diff() / (p + 1e-8)
        return pd.DataFrame(cols, index=prices.index)

    def _return_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for t in returns.columns:
            r = returns[t]
            for lag in (1, 2, 3, 5, 10):
                cols[f"{t}_lag_{lag}"] = r.shift(lag)
            for w in self.config.window_sizes:
                cols[f"{t}_sum_{w}"] = r.rolling(w, min_periods=w).sum()
                cols[f"{t}_abs_mean_{w}"] = r.abs().rolling(w, min_periods=w).mean()
            cols[f"{t}_sign"] = np.sign(r)
            cols[f"{t}_rank"] = r.rolling(20).rank(pct=True)
        return pd.DataFrame(cols, index=returns.index)

    def _volatility_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for t in returns.columns:
            r = returns[t]
            for w in self.config.window_sizes:
                cols[f"{t}_vol_{w}"] = r.rolling(w, min_periods=w).std()
            cols[f"{t}_high_low"] = r.rolling(20).max() - r.rolling(20).min()
            cols[f"{t}_vol_ratio"] = r.rolling(5).std() / (r.rolling(20).std() + 1e-8)
            # (old "parkinson" column removed: it needs high/low prices and was just a std copy)
        return pd.DataFrame(cols, index=returns.index)

    def _momentum_features(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for t in prices.columns:
            p, r = prices[t], returns[t]
            for period in (10, 20, 50):
                cols[f"{t}_roc_{period}"] = (p / p.shift(period) - 1) * 100
            cols[f"{t}_rsi"] = self._calculate_rsi(r)
            macd, sig = self._calculate_macd(p)
            cols[f"{t}_macd"] = macd / (p + 1e-8)
            cols[f"{t}_macd_signal"] = sig / (p + 1e-8)
            cols[f"{t}_macd_hist"] = (macd - sig) / (p + 1e-8)
            cols[f"{t}_momentum"] = p / p.shift(10) - 1
            cols[f"{t}_williams_r"] = self._calculate_williams_r(p)
            up, down = self._calculate_aroon(p)
            cols[f"{t}_aroon_up"], cols[f"{t}_aroon_down"] = up, down
        return pd.DataFrame(cols, index=returns.index)

    def _technical_indicators(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for t in prices.columns:
            p, r = prices[t], returns[t]
            ma20, std20 = p.rolling(20).mean(), p.rolling(20).std()
            cols[f"{t}_bb_upper"] = (ma20 + 2 * std20) / (p + 1e-8) - 1   # relative, not price level
            cols[f"{t}_bb_lower"] = (ma20 - 2 * std20) / (p + 1e-8) - 1
            cols[f"{t}_bb_width"] = (2 * std20) / (ma20 + 1e-8)
            cols[f"{t}_bb_position"] = (p - ma20) / (2 * std20 + 1e-8)
            cols[f"{t}_stoch"] = self._calculate_stochastic(p)
            cols[f"{t}_adx"] = self._calculate_adx(p)
            cols[f"{t}_ichimoku_tenkan"] = self._calculate_ichimoku_tenkan(p) / (p + 1e-8) - 1
            cols[f"{t}_ichimoku_kijun"] = self._calculate_ichimoku_kijun(p) / (p + 1e-8) - 1
            cols[f"{t}_cci"] = self._calculate_cci(p)
            cols[f"{t}_mfi"] = self._calculate_mfi(p, r)        # was passed the whole DataFrame
            cols[f"{t}_cmf"] = self._calculate_cmf(p) / (p + 1e-8)
            cols[f"{t}_eom"] = self._calculate_eom(p)
        return pd.DataFrame(cols, index=returns.index)

    def _correlation_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        if returns.shape[1] < 2:
            return pd.DataFrame(index=returns.index)
        cols = {}
        market = returns.mean(axis=1)
        for t in returns.columns:
            cols[f"{t}_corr_market"] = returns[t].rolling(20).corr(market)
        if returns.shape[1] <= self.config.max_pairwise_assets:
            names = list(returns.columns)
            for i, t1 in enumerate(names):
                for t2 in names[i + 1:]:
                    cols[f"{t1}_{t2}_corr"] = returns[t1].rolling(20).corr(returns[t2])
        return pd.DataFrame(cols, index=returns.index)

    def _microstructure_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for t in returns.columns:
            r = returns[t]
            a = r.abs()
            for lag in (1, 2, 5):
                cols[f"{t}_autocorr_{lag}"] = r.rolling(20).corr(r.shift(lag))
            cols[f"{t}_abs_autocorr"] = a.rolling(20).corr(a.shift(1))
            cols[f"{t}_hurst"] = r.rolling(60).apply(self._hurst_rs, raw=True)  # rolling => no look-ahead
            cols[f"{t}_var_ratio"] = r.rolling(10).var() / (r.rolling(30).var() + 1e-8)
            cols[f"{t}_tick_vol"] = a.rolling(5).sum()
            cols[f"{t}_signed_volume"] = r * a.rolling(10).mean()
        return pd.DataFrame(cols, index=returns.index)

    def _cyclical_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        idx = returns.index
        cols = {
            "day_of_week": idx.dayofweek,
            "day_of_week_sin": np.sin(2 * np.pi * idx.dayofweek / 7),
            "day_of_week_cos": np.cos(2 * np.pi * idx.dayofweek / 7),
            "month": idx.month,
            "month_sin": np.sin(2 * np.pi * idx.month / 12),
            "month_cos": np.cos(2 * np.pi * idx.month / 12),
            "quarter": idx.quarter,
            "year_progress": idx.dayofyear / 365,
        }
        df = pd.DataFrame(cols, index=idx)
        extra = {}
        for t in returns.columns:
            r = returns[t]
            # expanding std per bucket: uses only past observations (old version used the full sample)
            extra[f"{t}_vol_by_dow"] = r.groupby(idx.dayofweek).transform(
                lambda s: s.expanding(min_periods=5).std())
            extra[f"{t}_vol_by_month"] = r.groupby(idx.month).transform(
                lambda s: s.expanding(min_periods=5).std())
        return pd.concat([df, pd.DataFrame(extra, index=idx)], axis=1)

    def _fractal_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for t in prices.columns:
            p = prices[t]
            cols[f"{t}_fractal_dim"] = p.rolling(100).apply(self._higuchi_fd, raw=True)
            cols[f"{t}_rs_hurst"] = p.rolling(100).apply(
                lambda x: self._hurst_rs(np.diff(np.log(x + 1e-8)), (5, 10, 20, 33)), raw=True)
            cols[f"{t}_box_dim"] = p.rolling(100).apply(self._box_dimension, raw=True)
        return pd.DataFrame(cols, index=prices.index)

    def _frequency_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for t in returns.columns:
            r = returns[t]
            cols[f"{t}_dominant_freq"] = r.rolling(64).apply(self._dominant_freq, raw=True)
            cols[f"{t}_spectral_entropy"] = r.rolling(64).apply(self._spectral_entropy, raw=True)
        return pd.DataFrame(cols, index=returns.index)

    def _add_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.interaction_features_ is None:
            self.interaction_features_ = features.var().nlargest(10).index.tolist()
        top = self.interaction_features_
        new = {}
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                f1, f2 = top[i], top[j]
                a = features[f1] if f1 in features else 0.0
                b = features[f2] if f2 in features else 0.0
                new[f"{f1}_{f2}_interact"] = a * b
        if not new:
            return features
        return pd.concat([features, pd.DataFrame(new, index=features.index)], axis=1)

    # ===== INDICATORS =====
    @staticmethod
    def _calculate_rsi(returns: pd.Series, period: int = 14) -> pd.Series:
        gain = returns.clip(lower=0).rolling(period).mean()
        loss = returns.clip(upper=0).abs().rolling(period).mean()
        return 100 - 100 / (1 + gain / (loss + 1e-8))

    @staticmethod
    def _calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        macd = prices.ewm(span=12).mean() - prices.ewm(span=26).mean()
        return macd, macd.ewm(span=9).mean()

    @staticmethod
    def _calculate_williams_r(prices: pd.Series, period: int = 14) -> pd.Series:
        hi, lo = prices.rolling(period).max(), prices.rolling(period).min()
        return -100 * (hi - prices) / (hi - lo + 1e-8)

    @staticmethod
    def _calculate_aroon(prices: pd.Series, period: int = 25) -> Tuple[pd.Series, pd.Series]:
        up = prices.rolling(period).apply(lambda x: 100.0 * (np.argmax(x) + 1) / period, raw=True)
        down = prices.rolling(period).apply(lambda x: 100.0 * (np.argmin(x) + 1) / period, raw=True)
        return up, down

    @staticmethod
    def _calculate_stochastic(prices: pd.Series, period: int = 14) -> pd.Series:
        hi, lo = prices.rolling(period).max(), prices.rolling(period).min()
        return 100 * (prices - lo) / (hi - lo + 1e-8)

    @staticmethod
    def _calculate_adx(prices: pd.Series, period: int = 14) -> pd.Series:
        move = prices.diff()
        dm_plus = move.clip(lower=0).rolling(period).mean()
        dm_minus = move.clip(upper=0).abs().rolling(period).mean()
        dx = 100 * (dm_plus - dm_minus).abs() / (dm_plus + dm_minus + 1e-8)
        return dx.rolling(period).mean()

    @staticmethod
    def _calculate_ichimoku_tenkan(prices: pd.Series, period: int = 9) -> pd.Series:
        return (prices.rolling(period).max() + prices.rolling(period).min()) / 2

    @staticmethod
    def _calculate_ichimoku_kijun(prices: pd.Series, period: int = 26) -> pd.Series:
        return (prices.rolling(period).max() + prices.rolling(period).min()) / 2

    @staticmethod
    def _calculate_cci(prices: pd.Series, period: int = 20) -> pd.Series:
        ma = prices.rolling(period).mean()
        mean_dev = prices.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        return (prices - ma) / (0.015 * mean_dev + 1e-8)

    @staticmethod
    def _calculate_mfi(prices: pd.Series, returns: pd.Series, period: int = 14) -> pd.Series:
        flow = prices * returns.abs()
        pos = flow.where(returns > 0, 0.0).rolling(period).sum()
        neg = flow.where(returns < 0, 0.0).rolling(period).sum()
        return 100 - 100 / (1 + pos / (neg + 1e-8))

    @staticmethod
    def _calculate_cmf(prices: pd.Series, period: int = 20) -> pd.Series:
        return (prices * prices.pct_change()).rolling(period).sum()

    @staticmethod
    def _calculate_eom(prices: pd.Series, period: int = 14) -> pd.Series:
        return prices.diff() / (prices.rolling(period).std() + 1e-8)

    # ===== WINDOW FUNCTIONS (operate on raw numpy windows) =====
    @staticmethod
    def _hurst_rs(x, lags=(6, 10, 15, 20, 30)) -> float:
        x = np.asarray(x, float)
        if len(x) < max(lags) or not np.isfinite(x).all():
            return 0.5
        log_lag, log_rs = [], []
        for lag in lags:
            n = len(x) // lag
            if n < 1:
                continue
            rs = []
            for i in range(n):
                w = x[i * lag:(i + 1) * lag]
                s = w.std(ddof=1)
                if s > 0:
                    c = np.cumsum(w - w.mean())
                    rs.append((c.max() - c.min()) / s)
            if rs:
                log_lag.append(np.log(lag))
                log_rs.append(np.log(np.mean(rs)))
        if len(log_lag) < 2:
            return 0.5
        return float(np.clip(np.polyfit(log_lag, log_rs, 1)[0], 0, 1))

    @staticmethod
    def _higuchi_fd(x, kmax: int = 10) -> float:
        x = np.asarray(x, float)
        N = len(x)
        if N < 20 or not np.isfinite(x).all():
            return 1.0
        ks, L = [], []
        for k in range(1, kmax + 1):
            lk = []
            for m in range(k):
                idx = np.arange(m, N, k)
                if len(idx) < 2:
                    continue
                d = np.abs(np.diff(x[idx])).sum()
                lk.append(d * (N - 1) / ((len(idx) - 1) * k) / k)
            if lk and np.mean(lk) > 0:
                ks.append(k)
                L.append(np.mean(lk))
        if len(ks) < 2:
            return 1.0
        return float(np.clip(-np.polyfit(np.log(ks), np.log(L), 1)[0], 0, 2))

    @staticmethod
    def _box_dimension(x) -> float:
        x = np.asarray(x, float)
        if len(x) < 10 or not np.isfinite(x).all():
            return 1.0
        xn = (x - x.min()) / (x.max() - x.min() + 1e-8)
        scales, counts = [], []
        for scale in range(2, min(20, len(xn) // 2)):
            n_boxes = 0
            for i in range(0, len(xn) - scale, scale):
                w = xn[i:i + scale]
                if w.max() - w.min() > 1 / scale:
                    n_boxes += 1
            if n_boxes > 0:
                scales.append(np.log(1 / scale))
                counts.append(np.log(n_boxes))
        if len(scales) < 2:
            return 1.0
        return float(np.clip(np.polyfit(scales, counts, 1)[0], 0, 2))

    @staticmethod
    def _dominant_freq(x) -> float:
        x = np.asarray(x, float)
        p = np.abs(np.fft.rfft(x - x.mean())) ** 2
        return float((np.argmax(p[1:]) + 1) / len(x)) if len(p) > 2 else 0.0

    @staticmethod
    def _spectral_entropy(x) -> float:
        x = np.asarray(x, float)
        p = (np.abs(np.fft.rfft(x - x.mean())) ** 2)[1:]
        if len(p) < 2:
            return 0.0
        p = p / (p.sum() + 1e-12)
        return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))

    # ------------------------------------------------------------
    def create_sequences(self, features: pd.DataFrame, seq_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        if len(features) < seq_length + 1:
            return np.array([]), np.array([])
        values = features.values.astype(np.float32)
        X = np.stack([values[i:i + seq_length] for i in range(len(values) - seq_length)])
        y = values[seq_length:]
        return X, y

    def normalize_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if fit or self.means_ is None:
            self.means_ = features.median()
            self.stds_ = features.quantile(0.75) - features.quantile(0.25)
        normalized = (features - self.means_) / (self.stds_ + 1e-8)
        return normalized.clip(-5, 5)