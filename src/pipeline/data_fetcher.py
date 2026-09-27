"""
Professional Data Fetcher - Finnhub → Alpha Vantage → yfinance → Synthetic
Never returns empty prices. Layered fallback strategy.
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
    Data Fetcher with 4-tier fallback:
        1. Finnhub        (fast, primary)
        2. Alpha Vantage  (secondary)
        3. yfinance       (broad coverage, no key needed)  ← NEW
        4. Synthetic      (last resort)
    """

    def __init__(self, use_live_simulation: bool = False):
        self.alpha_vantage_key = self._get_alpha_vantage_key()
        self.finnhub_key = self._get_finnhub_key()
        self.use_live_simulation = use_live_simulation

        self.current_source = "No Data"

        # Live simulation support
        self.live_simulator = None
        self._price_subscribers = []
        self.is_running = False
        self.live_prices = {}

        # Cache
        self.price_cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 60

        self.historical_cache = {}
        self.historical_timestamp = {}
        self.historical_cache_duration = 3600

        print("✅ DataFetcher initialized")
        print(f"   🔑 Finnhub: {'✅' if self.finnhub_key else '❌'}")
        print(f"   🔑 Alpha Vantage: {'✅' if self.alpha_vantage_key else '❌'}")
        print("   📊 yfinance: ✅ (always available)")

    # ============================================================
    # SAFE SECRETS ACCESS
    # ============================================================
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
    # TIER 1: FINNHUB
    # ============================================================
    def _call_finnhub(self, symbol: str) -> Optional[float]:
        if not self.finnhub_key:
            return None
        try:
            url = "https://finnhub.io/api/v1/quote"
            params = {"symbol": symbol, "token": self.finnhub_key}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                price = data.get('c', 0)
                if price and price > 0:
                    print(f"✅ Finnhub: {symbol} = ${price}")
                    return float(price)
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
                    print(f"⏳ Alpha Vantage rate limit: {symbol}")
        except Exception as e:
            print(f"⚠️ Alpha Vantage error for {symbol}: {e}")
        return None

    # ============================================================
    # TIER 3: YFINANCE  ← NEW
    # ============================================================
    def _call_yfinance(self, symbol: str) -> Optional[float]:
        """yfinance fallback — works for global + crypto, no API key needed"""
        try:
            import yfinance as yf

            # fast_info is faster than .info and less likely to be empty
            ticker = yf.Ticker(symbol)
            try:
                price = ticker.fast_info.get('last_price')
                if price and price > 0:
                    print(f"✅ yfinance: {symbol} = ${price:.2f}")
                    return float(price)
            except Exception:
                pass

            # Fallback: use recent history (last 5 days) and take last close
            try:
                hist = ticker.history(period="5d", auto_adjust=False)
                if not hist.empty and 'Close' in hist.columns:
                    price = float(hist['Close'].iloc[-1])
                    if price > 0:
                        print(f"✅ yfinance (history): {symbol} = ${price:.2f}")
                        return price
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ yfinance error for {symbol}: {e}")
        return None

    # ============================================================
    # CURRENT PRICES — 4-tier fallback
    # ============================================================
    def get_current_prices(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """Get current prices: Finnhub → AV → yfinance → synthetic (never empty)"""
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

            # Tier 1
            price = self._call_finnhub(ticker)
            if price: source = "Finnhub"

            # Tier 2
            if price is None:
                price = self._call_alpha_vantage(ticker)
                if price: source = "Alpha Vantage"

            # Tier 3
            if price is None:
                price = self._call_yfinance(ticker)
                if price: source = "yfinance"

            # Store or mark missing
            if price is not None and price > 0:
                prices[ticker] = price
                self.price_cache[ticker] = price
                self.cache_timestamp[ticker] = now
                self.current_source = source
            else:
                missing.append(ticker)

            time.sleep(0.2)

        # Tier 4: synthetic fallback (never return empty)
        if missing:
            print(f"⚠️ All APIs failed for: {missing} — using synthetic fallback")
            for ticker in missing:
                fb = 100.0
                # Try historical cache first
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
    # HISTORICAL DATA — AV → yfinance → synthetic
    # ============================================================
    def get_historical_data(self, tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
        """Historical data: Alpha Vantage → yfinance → synthetic"""

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

            # ---- Tier 1: Alpha Vantage ----
            if self.alpha_vantage_key:
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

            # ---- Tier 2: yfinance (NEW) ----
            if data is None:
                try:
                    import yfinance as yf
                    print(f"📡 Trying yfinance historical for {ticker}...")
                    yf_period = self._days_to_yf_period(days)
                    hist = yf.Ticker(ticker).history(period=yf_period, auto_adjust=False)

                    if not hist.empty and 'Close' in hist.columns:
                        df = hist[['Close']].rename(columns={'Close': ticker})
                        df.index = pd.to_datetime(df.index).tz_localize(None)
                        if len(df) > days:
                            df = df.last(days)
                        data = df
                        print(f"✅ yfinance historical: {ticker} ({len(data)} days)")
                except Exception as e:
                    print(f"⚠️ yfinance historical error for {ticker}: {e}")

            # ---- Tier 3: Synthetic ----
            if data is None:
                print(f"⚠️ No historical data for {ticker}, using synthetic")
                base_price = 100.0
                # Try to get a real base price first
                current_price = (
                    self._call_finnhub(ticker) or
                    self._call_alpha_vantage(ticker) or
                    self._call_yfinance(ticker)
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

    # ============================================================
    # HELPERS
    # ============================================================
    def _period_to_days(self, period: str) -> int:
        if period.endswith('d'):
            return int(period[:-1])
        elif period.endswith('mo'):
            return int(period[:-2]) * 21
        elif period.endswith('y'):
            return int(period[:-1]) * 252
        return 252

    def _days_to_yf_period(self, days: int) -> str:
        """Convert day count → yfinance period string"""
        if days <= 5:    return "5d"
        if days <= 30:   return "1mo"
        if days <= 90:   return "3mo"
        if days <= 180:  return "6mo"
        if days <= 365:  return "1y"
        if days <= 730:  return "2y"
        if days <= 1825: return "5y"
        return "10y"

    # ============================================================
    # OPTION CHAIN — yfinance (with graceful failure)
    # ============================================================
    def get_option_chain(self, ticker: str) -> Dict:
        """Option chain via yfinance (already the only source)"""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            expirations = stock.options
            if not expirations:
                return {}

            chain = stock.option_chain(expirations[0])

            calls = []
            for _, row in chain.calls.head(10).iterrows():
                calls.append({
                    'strike': float(row.get('strike', 0)),
                    'lastPrice': float(row.get('lastPrice', 0)),
                    'bid': float(row.get('bid', 0)),
                    'ask': float(row.get('ask', 0)),
                    'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                })

            puts = []
            for _, row in chain.puts.head(10).iterrows():
                puts.append({
                    'strike': float(row.get('strike', 0)),
                    'lastPrice': float(row.get('lastPrice', 0)),
                    'bid': float(row.get('bid', 0)),
                    'ask': float(row.get('ask', 0)),
                    'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                })

            return {
                'calls': calls,
                'puts': puts,
                'expiration': expirations[0],
                'underlying_price': float(chain.underlying['price']) if 'price' in chain.underlying else 0
            }
        except Exception as e:
            print(f"⚠️ Option chain error for {ticker}: {e}")
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
                    self._call_alpha_vantage(ticker) or
                    self._call_yfinance(ticker)
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