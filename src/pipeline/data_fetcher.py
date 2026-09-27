"""
Professional Data Fetcher - Working on Streamlit Cloud
Uses finnhub-python SDK (reliable) + Alpha Vantage + synthetic
"""

import pandas as pd
import numpy as np
import time
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
import streamlit as st

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Reliable multi-source fetcher:
        1. Finnhub (via official SDK — works on cloud)
        2. Alpha Vantage (25/day limit)
        3. Synthetic (never fails)
    """

    def __init__(self, use_live_simulation: bool = False):
        self.alpha_vantage_key = self._get_alpha_vantage_key()
        self.finnhub_key = self._get_finnhub_key()
        self.use_live_simulation = use_live_simulation

        self.current_source = "No Data"

        self.live_simulator = None
        self._price_subscribers = []
        self.is_running = False
        self.live_prices = {}

        self.price_cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 300  # 5 min — preserve AV quota

        self.historical_cache = {}
        self.historical_timestamp = {}
        self.historical_cache_duration = 3600

        # ✅ Initialize Finnhub SDK
        self.finnhub_client = None
        if self.finnhub_key:
            try:
                import finnhub
                self.finnhub_client = finnhub.Client(api_key=self.finnhub_key)
                print(f"   🔑 Finnhub SDK: ✅ initialized")
            except ImportError:
                print(f"   ⚠️ finnhub-python not installed")
            except Exception as e:
                print(f"   ⚠️ Finnhub SDK init failed: {e}")

        print("✅ DataFetcher initialized")
        print(f"   🔑 Finnhub: {'✅' if self.finnhub_client else '❌'}")
        print(f"   🔑 Alpha Vantage: {'✅' if self.alpha_vantage_key else '❌'}")

    def _get_alpha_vantage_key(self) -> str:
        try:
            if hasattr(st, 'secrets'):
                try:
                    return st.secrets.get('ALPHA_VANTAGE_API_KEY', '') or ''
                except Exception:
                    pass
        except Exception:
            pass
        return os.environ.get("ALPHA_VANTAGE_API_KEY", "") or ""

    def _get_finnhub_key(self) -> str:
        try:
            if hasattr(st, 'secrets'):
                try:
                    return st.secrets.get('FINNHUB_API_KEY', '') or ''
                except Exception:
                    pass
        except Exception:
            pass
        return os.environ.get("FINNHUB_API_KEY", "") or ""

    # ============================================================
    # TIER 1: FINNHUB via official SDK
    # ============================================================
    def _call_finnhub(self, symbol: str) -> Optional[float]:
        if not self.finnhub_client:
            return None
        try:
            # ✅ Use the SDK — handles headers, sessions, retries
            quote = self.finnhub_client.quote(symbol)
            if quote and isinstance(quote, dict):
                price = quote.get('c', 0)
                if price and price > 0:
                    print(f"✅ Finnhub: {symbol} = ${price}")
                    return float(price)
                else:
                    print(f"⚠️ Finnhub returned 0 for {symbol} (delisted or unsupported)")
        except Exception as e:
            print(f"⚠️ Finnhub error for {symbol}: {e}")
        return None

    # ============================================================
    # TIER 2: ALPHA VANTAGE
    # ============================================================
    def _call_alpha_vantage(self, symbol: str) -> Optional[float]:
        if not self.alpha_vantage_key:
            return None
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.alpha_vantage_key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "Global Quote" in data and data["Global Quote"]:
                    price = data["Global Quote"].get("05. price")
                    if price:
                        print(f"✅ Alpha Vantage: {symbol} = ${price}")
                        return float(price)
                elif "Note" in data:
                    print(f"⏳ Alpha Vantage rate limit hit")
                elif "Information" in data:
                    print(f"⏳ Alpha Vantage: {data['Information'][:100]}")
        except Exception as e:
            print(f"⚠️ Alpha Vantage error for {symbol}: {e}")
        return None

    # ============================================================
    # CURRENT PRICES — 3-tier fallback (finnhub → AV → synthetic)
    # ============================================================
    def get_current_prices(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, float]:
        now = time.time()
        prices = {}
        stale_tickers = []

        for ticker in tickers:
            if not force_refresh and ticker in self.cache_timestamp:
                if now - self.cache_timestamp[ticker] < self.cache_duration:
                    prices[ticker] = self.price_cache[ticker]
                    continue
            stale_tickers.append(ticker)

        if not stale_tickers:
            return prices

        print(f"📡 Fetching fresh prices for: {stale_tickers}")

        missing = []
        for ticker in stale_tickers:
            price = None
            source = None

            # Tier 1: Finnhub
            price = self._call_finnhub(ticker)
            if price: source = "Finnhub"

            # Tier 2: Alpha Vantage
            if price is None:
                price = self._call_alpha_vantage(ticker)
                if price: source = "Alpha Vantage"

            if price is not None and price > 0:
                prices[ticker] = price
                self.price_cache[ticker] = price
                self.cache_timestamp[ticker] = now
                self.current_source = source
            else:
                missing.append(ticker)

            time.sleep(0.2)

        if missing:
            print(f"⚠️ All APIs failed for: {missing} — using synthetic fallback")
            for ticker in missing:
                fb = 100.0
                for key, df in self.historical_cache.items():
                    if ticker in df.columns and len(df) > 0:
                        try:
                            fb = float(df[ticker].iloc[-1])
                        except Exception:
                            pass
                        break

                prices[ticker] = fb
                self.price_cache[ticker] = fb
                self.cache_timestamp[ticker] = now
            self.current_source = "Synthetic Fallback"

        return prices

    # ============================================================
    # HISTORICAL DATA
    # ============================================================
    def get_historical_data(self, tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
        """Historical data via Finnhub stock_candles → AV → synthetic"""

        cache_key = f"{tickers_tuple}_{period}"
        now = time.time()

        if cache_key in self.historical_timestamp:
            if now - self.historical_timestamp[cache_key] < self.historical_cache_duration:
                print("📦 Using cached historical data")
                return self.historical_cache[cache_key]

        tickers = list(tickers_tuple)
        print(f"📥 Fetching historical data for {len(tickers)} stocks")

        all_data = []
        days = self._period_to_days(period)

        for ticker in tickers:
            data = None

            # ---- Tier 1: Finnhub candles ----
            if data is None and self.finnhub_client:
                try:
                    end_ts = int(datetime.now().timestamp())
                    start_ts = int((datetime.now() - timedelta(days=int(days * 1.6))).timestamp())

                    print(f"📡 Trying Finnhub candles for {ticker}...")
                    candles = self.finnhub_client.stock_candles(
                        ticker, 'D', start_ts, end_ts
                    )

                    if candles and isinstance(candles, dict) and candles.get('s') == 'ok':
                        closes = candles.get('c', [])
                        timestamps = candles.get('t', [])
                        if closes and timestamps:
                            dates = pd.to_datetime(timestamps, unit='s')
                            df = pd.DataFrame({ticker: closes}, index=dates)
                            df = df.sort_index()
                            if len(df) > days:
                                df = df.iloc[-days:]
                            data = df
                            print(f"✅ Finnhub historical: {ticker} ({len(data)} days)")
                    else:
                        print(f"⚠️ Finnhub candles failed for {ticker} (may require paid tier)")
                except Exception as e:
                    print(f"⚠️ Finnhub candles error for {ticker}: {e}")

            # ---- Tier 2: Alpha Vantage ----
            if data is None and self.alpha_vantage_key:
                try:
                    url = "https://www.alphavantage.co/query"
                    params = {
                        "function": "TIME_SERIES_DAILY",
                        "symbol": ticker,
                        "outputsize": "compact",
                        "apikey": self.alpha_vantage_key
                    }
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        if "Time Series (Daily)" in result:
                            ts = result["Time Series (Daily)"]
                            df = pd.DataFrame.from_dict(ts, orient="index")
                            df.index = pd.to_datetime(df.index)
                            df = df.astype(float).sort_index()
                            if len(df) > days:
                                df = df.last(days)
                            if not df.empty:
                                data = df[['4. close']].rename(columns={'4. close': ticker})
                                print(f"✅ Alpha Vantage historical: {ticker} ({len(data)} days)")
                except Exception as e:
                    print(f"⚠️ Alpha Vantage historical error for {ticker}: {e}")

            # ---- Tier 3: Synthetic ----
            if data is None:
                print(f"⚠️ No historical data for {ticker}, using synthetic")
                base_price = 100.0
                current_price = (
                    self._call_finnhub(ticker) or
                    self._call_alpha_vantage(ticker)
                )
                if current_price:
                    base_price = current_price

                returns = np.random.randn(days) * 0.02 + 0.0003
                prices = base_price * np.exp(np.cumsum(returns))
                prices = np.maximum(prices, base_price * 0.3)
                dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
                data = pd.DataFrame({ticker: prices}, index=dates)
                print(f"📊 Generated synthetic data for {ticker} (base: ${base_price:.2f})")

            if data is not None:
                all_data.append(data)

            time.sleep(0.3)

        if not all_data:
            print("❌ No historical data available")
            return None

        try:
            combined = pd.concat(all_data, axis=1)
            combined = combined.ffill().bfill()

            self.historical_cache[cache_key] = combined
            self.historical_timestamp[cache_key] = now

            print(f"✅ Successfully generated data for {len(combined.columns)} stocks")
            return combined

        except Exception as e:
            print(f"❌ Data generation failed: {e}")
            return None

    def _period_to_days(self, period: str) -> int:
        if period.endswith('d'):
            return int(period[:-1])
        elif period.endswith('mo'):
            return int(period[:-2]) * 21
        elif period.endswith('y'):
            return int(period[:-1]) * 252
        return 252

    # ============================================================
    # OPTION CHAIN — stub (no reliable free source)
    # ============================================================
    def get_option_chain(self, ticker: str) -> Dict:
        """Options data unavailable on cloud without a paid provider."""
        return {}

    def get_data_source_info(self) -> str:
        return self.current_source

    # ============================================================
    # LIVE SIMULATION
    # ============================================================
    def start_live_simulation(self, tickers: List[str]):
        if self.use_live_simulation:
            print(f"🔄 Starting live simulation for {tickers}")
            self.live_prices = {}
            for ticker in tickers:
                price = (
                    self._call_finnhub(ticker) or
                    self._call_alpha_vantage(ticker)
                )
                self.live_prices[ticker] = price if price else 100
            self.is_running = True
            self.current_source = "Live Simulation"

    def stop_live_simulation(self):
        self.is_running = False
        print("🔄 Live simulation stopped")

    def subscribe_prices(self, callback):
        self._price_subscribers.append(callback)

    def get_live_prices(self) -> Dict[str, float]:
        if hasattr(self, 'live_prices') and self.is_running:
            for ticker in self.live_prices:
                change = np.random.uniform(-0.005, 0.005) * self.live_prices[ticker]
                self.live_prices[ticker] += change
                self.live_prices[ticker] = max(self.live_prices[ticker], 0.01)
            return self.live_prices
        return {}

    def get_price_history(self, ticker: str) -> List[Dict]:
        if hasattr(self, 'live_prices') and ticker in self.live_prices:
            return [{
                'time': datetime.now() - timedelta(minutes=i),
                'price': self.live_prices[ticker] * (1 + np.random.uniform(-0.01, 0.01))
            } for i in range(20)]
        return []