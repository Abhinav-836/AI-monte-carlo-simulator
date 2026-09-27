"""
Professional Data Fetcher - 4-tier fallback
Finnhub → Alpha Vantage → yfinance (native curl_cffi) → Synthetic

yfinance >= 0.2.54 has native curl_cffi support. When curl_cffi is
installed, yfinance automatically impersonates Chrome — no session
parameter needed.
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
    4-tier fallback:
        1. Finnhub
        2. Alpha Vantage
        3. yfinance (with native curl_cffi impersonation)
        4. Synthetic
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
        self.cache_duration = 60

        self.historical_cache = {}
        self.historical_timestamp = {}
        self.historical_cache_duration = 3600

        # ✅ Verify availability
        self._yf_ok = False
        self._curl_cffi_ok = False
        try:
            import yfinance as yf
            self._yf_ok = True
            print(f"   📊 yfinance version: {yf.__version__}")
        except ImportError:
            pass

        try:
            from curl_cffi import requests as _cr  # noqa: F401
            self._curl_cffi_ok = True
        except ImportError:
            pass

        print("✅ DataFetcher initialized")
        print(f"   🔑 Finnhub: {'✅' if self.finnhub_key else '❌'}")
        print(f"   🔑 Alpha Vantage: {'✅' if self.alpha_vantage_key else '❌'}")
        print(f"   📊 yfinance: {'✅' if self._yf_ok else '❌'} (curl_cffi: {'✅' if self._curl_cffi_ok else '❌'})")

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
    # ✅ YFINANCE HELPER — no session= passing!
    #
    # yfinance >= 0.2.54 uses curl_cffi automatically when installed.
    # We must NOT pass session= — that API expects a specific type
    # and breaks with the wrong curl_cffi version.
    # ============================================================
    def _make_yf_ticker(self, symbol: str):
        """Create a plain yfinance Ticker — curl_cffi is used automatically."""
        import yfinance as yf
        return yf.Ticker(symbol)

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
                else:
                    print(f"⚠️ Finnhub no data for {symbol}: {data}")
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
    # TIER 3: YFINANCE
    # ============================================================
    def _call_yfinance(self, symbol: str) -> Optional[float]:
        """yfinance fallback — curl_cffi handled internally by yfinance"""
        try:
            ticker = self._make_yf_ticker(symbol)

            # Try fast_info
            try:
                price = ticker.fast_info.get('last_price')
                if price and price > 0:
                    print(f"✅ yfinance: {symbol} = ${price:.2f}")
                    return float(price)
            except Exception as e:
                print(f"⚠️ yfinance fast_info failed for {symbol}: {e}")

            # Try history
            try:
                hist = ticker.history(period="5d", auto_adjust=False)
                if not hist.empty and 'Close' in hist.columns:
                    price = float(hist['Close'].iloc[-1])
                    if price > 0:
                        print(f"✅ yfinance: {symbol} = ${price:.2f}")
                        return price
            except Exception as e:
                print(f"⚠️ yfinance history failed for {symbol}: {e}")

        except Exception as e:
            print(f"⚠️ yfinance error for {symbol}: {e}")
        return None

    # ============================================================
    # CURRENT PRICES — 4-tier fallback
    # ============================================================
    def get_current_prices(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """Get current prices: Finnhub → AV → yfinance → synthetic"""
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

            price = self._call_finnhub(ticker)
            if price: source = "Finnhub"

            if price is None:
                price = self._call_alpha_vantage(ticker)
                if price: source = "Alpha Vantage"

            if price is None:
                price = self._call_yfinance(ticker)
                if price: source = "yfinance"

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
        """Historical data: yfinance → AV → synthetic"""

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

            # ---- Tier 1: yfinance (preferred — full history) ----
            if data is None:
                try:
                    from datetime import datetime as _dt, timedelta as _td

                    print(f"📡 Trying yfinance historical for {ticker}...")
                    ticker_obj = self._make_yf_ticker(ticker)

                    end_dt = _dt.now()
                    start_dt = end_dt - _td(days=int(days * 1.6))

                    hist = ticker_obj.history(
                        start=start_dt.strftime("%Y-%m-%d"),
                        end=end_dt.strftime("%Y-%m-%d"),
                        auto_adjust=False,
                        interval="1d"
                    )

                    if not hist.empty and 'Close' in hist.columns:
                        df = hist[['Close']].rename(columns={'Close': ticker})
                        df.index = pd.to_datetime(df.index).tz_localize(None)
                        if len(df) > days:
                            df = df.iloc[-days:]
                        data = df
                        print(f"✅ yfinance historical: {ticker} ({len(data)} days)")
                except Exception as e:
                    print(f"⚠️ yfinance historical error for {ticker}: {e}")

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

    def _period_to_days(self, period: str) -> int:
        if period.endswith('d'):
            return int(period[:-1])
        elif period.endswith('mo'):
            return int(period[:-2]) * 21
        elif period.endswith('y'):
            return int(period[:-1]) * 252
        return 252

    # ============================================================
    # OPTION CHAIN
    # ============================================================
    def get_option_chain(self, ticker: str) -> Dict:
        try:
            stock = self._make_yf_ticker(ticker)
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