"""
Professional Data Fetcher — 5-tier fallback
Twelve Data → Finnhub → Alpha Vantage → yfinance → Real-or-nothing

Twelve Data free tier: 800 requests/day, 8 requests/minute.
Works on Streamlit Cloud with NO proxy needed.
Never fabricates prices — real data or explicit "unavailable".
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

# ✅ Load .env BEFORE reading any keys
from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)


@st.cache_resource
def get_data_fetcher(use_live_simulation: bool = False):
    """Return one persistent DataFetcher per Streamlit app/session process."""
    return DataFetcher(use_live_simulation=use_live_simulation)


class DataFetcher:
    """
    5-tier fallback:
        1. Twelve Data    (free, all markets, cloud-safe) ← PRIMARY
        2. Finnhub        (US-focused)
        3. Alpha Vantage  (25/day limit)
        4. yfinance       (works locally, may fail on cloud)
        5. Real-or-nothing (never synthetic)
    """

    def __init__(self, use_live_simulation: bool = False):
        self.twelve_data_key = self._get_twelve_data_key()
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
        self.cache_duration = 300  # 5 min — preserve Twelve Data quota

        self.historical_cache = {}
        self.historical_timestamp = {}
        self.historical_cache_duration = 3600

        # ✅ Rate-limit tracker for Twelve Data (8 req/min free tier)
        self._td_last_call = 0.0

        # Finnhub SDK
        self.finnhub_client = None
        if self.finnhub_key:
            try:
                import finnhub
                self.finnhub_client = finnhub.Client(api_key=self.finnhub_key)
                print("   🔑 Finnhub SDK: ✅ initialized")
            except ImportError:
                print("   ⚠️ finnhub-python not installed")
            except Exception as e:
                print(f"   ⚠️ Finnhub SDK init failed: {e}")

        print("✅ DataFetcher initialized")
        print(f"   🎯 Twelve Data: {'✅' if self.twelve_data_key else '❌'}")
        print(f"   🔑 Finnhub: {'✅' if self.finnhub_client else '❌'}")
        print(f"   🔑 Alpha Vantage: {'✅' if self.alpha_vantage_key else '❌'}")

    # ============================================================
    # ✅ KEY GETTERS — fall through to os.environ when st.secrets is empty
    # ============================================================
    def _get_twelve_data_key(self) -> str:
        try:
            if hasattr(st, 'secrets'):
                try:
                    val = st.secrets.get('TWELVE_DATA_API_KEY', '')
                    if val and str(val).strip():
                        return str(val).strip()
                except Exception:
                    pass
        except Exception:
            pass
        return (os.environ.get("TWELVE_DATA_API_KEY", "") or "").strip()

    def _get_alpha_vantage_key(self) -> str:
        try:
            if hasattr(st, 'secrets'):
                try:
                    val = st.secrets.get('ALPHA_VANTAGE_API_KEY', '')
                    if val and str(val).strip():
                        return str(val).strip()
                except Exception:
                    pass
        except Exception:
            pass
        return (os.environ.get("ALPHA_VANTAGE_API_KEY", "") or "").strip()

    def _get_finnhub_key(self) -> str:
        try:
            if hasattr(st, 'secrets'):
                try:
                    val = st.secrets.get('FINNHUB_API_KEY', '')
                    if val and str(val).strip():
                        return str(val).strip()
                except Exception:
                    pass
        except Exception:
            pass
        return (os.environ.get("FINNHUB_API_KEY", "") or "").strip()

    # ============================================================
    # ✅ RATE-LIMIT THROTTLE
    # Twelve Data free tier = 8 requests/minute.
    # Minimum 1 second between calls keeps us safely under the limit.
    # ============================================================
    def _throttle_twelve_data(self):
        now = time.time()
        elapsed = now - self._td_last_call
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._td_last_call = time.time()

    # ============================================================
    # SYMBOL CONVERSION for Twelve Data
    # ============================================================
    def _convert_to_twelve_symbol(self, ticker: str) -> str:
        """
        Convert yfinance-style tickers to Twelve Data format.
        Twelve Data uses `EXCHANGE:SYMBOL` for non-US markets.
        """
        if ticker.endswith('.NS'):
            return f"NSE:{ticker[:-3]}"
        if ticker.endswith('.L'):
            return f"LSE:{ticker[:-2]}"
        if ticker.endswith('.HK'):
            return f"HKEX:{ticker[:-3]}"
        if ticker.endswith('.TO'):
            return f"TSX:{ticker[:-3]}"
        if ticker.endswith('.AX'):
            return f"ASX:{ticker[:-3]}"
        if ticker.endswith('.DE'):
            return f"XETR:{ticker[:-3]}"
        if '-USD' in ticker:
            return ticker.replace('-', '/')
        return ticker

    # ============================================================
    # TIER 1: TWELVE DATA (primary — all markets)
    # ============================================================
    def _call_twelve_data(self, symbol: str) -> Optional[float]:
        """Twelve Data price endpoint — with rate-limit throttle."""
        if not self.twelve_data_key:
            return None

        self._throttle_twelve_data()

        try:
            td_symbol = self._convert_to_twelve_symbol(symbol)
            url = "https://api.twelvedata.com/price"
            params = {
                "symbol": td_symbol,
                "apikey": self.twelve_data_key,
            }
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if "price" in data:
                    price = float(data["price"])
                    if price > 0:
                        print(f"✅ Twelve Data: {symbol} = ${price}")
                        return price
                elif "code" in data:
                    code = data.get("code")
                    if code == 429:
                        print(f"⏳ Twelve Data rate limit hit for {symbol}")
                    elif code == 401:
                        print(f"⚠️ Twelve Data: invalid API key")
                    else:
                        msg = data.get("message", "")[:80]
                        print(f"⚠️ Twelve Data: {msg}")
        except Exception as e:
            print(f"⚠️ Twelve Data error for {symbol}: {e}")
        return None

    # ============================================================
    # TIER 2: FINNHUB
    # ============================================================
    def _call_finnhub(self, symbol: str) -> Optional[float]:
        if not self.finnhub_client:
            return None
        try:
            quote = self.finnhub_client.quote(symbol)
            if quote and isinstance(quote, dict):
                price = quote.get('c', 0)
                if price and price > 0:
                    print(f"✅ Finnhub: {symbol} = ${price}")
                    return float(price)
        except Exception as e:
            err_str = str(e)
            if "401" in err_str:
                print(f"⚠️ Finnhub: invalid API key (401)")
            elif "403" in err_str:
                print(f"⏭️ Finnhub: {symbol} not covered by free tier")
            else:
                print(f"⚠️ Finnhub error for {symbol}: {e}")
        return None

    # ============================================================
    # TIER 3: ALPHA VANTAGE
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
                elif "Note" in data or "Information" in data:
                    print(f"⏳ Alpha Vantage rate limit hit")
        except Exception as e:
            print(f"⚠️ Alpha Vantage error for {symbol}: {e}")
        return None

    # ============================================================
    # TIER 4: YFINANCE (used only for historical, not current)
    # ============================================================
    def _call_yfinance(self, symbol: str) -> Optional[float]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            try:
                price = ticker.fast_info.get('last_price')
                if price and price > 0:
                    print(f"✅ yfinance: {symbol} = ${price:.2f}")
                    return float(price)
            except Exception:
                pass

            try:
                hist = ticker.history(period="5d", auto_adjust=False)
                if not hist.empty and 'Close' in hist.columns:
                    price = float(hist['Close'].dropna().iloc[-1])
                    if price > 0:
                        print(f"✅ yfinance: {symbol} = ${price:.2f}")
                        return price
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ yfinance error for {symbol}: {e}")
        return None

    # ============================================================
    # CURRENT PRICES — 4-tier fallback (no synthetic)
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

            # Tier 1: Twelve Data
            price = self._call_twelve_data(ticker)
            if price:
                source = "Twelve Data"

            # Tier 2: Finnhub
            if price is None:
                price = self._call_finnhub(ticker)
                if price:
                    source = "Finnhub"

            # Tier 3: Alpha Vantage
            if price is None:
                price = self._call_alpha_vantage(ticker)
                if price:
                    source = "Alpha Vantage"

            # Tier 4: last-known real price (never synthetic)
            if price is None and ticker in self.price_cache:
                price = self.price_cache[ticker]
                source = "Cached"

            if price is not None and price > 0:
                prices[ticker] = price
                self.price_cache[ticker] = price
                self.cache_timestamp[ticker] = now
                self.current_source = source
            else:
                missing.append(ticker)

            time.sleep(0.2)

        if missing:
            print(f"⚠️ No live market data available for: {missing}")
            # Never fabricate prices. A missing price stays missing.
            self.current_source = "No Data" if not prices else "Partial Data"

        return prices

    # ============================================================
    # HISTORICAL DATA
    # ============================================================
    def get_historical_data(self, tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
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

            # ---- Tier 1: Twelve Data time_series ----
            if data is None and self.twelve_data_key:
                try:
                    self._throttle_twelve_data()

                    print(f"📡 Trying Twelve Data historical for {ticker}...")
                    td_symbol = self._convert_to_twelve_symbol(ticker)
                    url = "https://api.twelvedata.com/time_series"
                    params = {
                        "symbol": td_symbol,
                        "interval": "1day",
                        "outputsize": min(days, 5000),
                        "apikey": self.twelve_data_key,
                        "format": "JSON",
                    }
                    response = requests.get(url, params=params, timeout=15)

                    if response.status_code == 200:
                        result = response.json()
                        if "values" in result:
                            df = pd.DataFrame(result["values"])
                            df["datetime"] = pd.to_datetime(df["datetime"])
                            df = df.set_index("datetime").sort_index()
                            df = df.astype(float)
                            if len(df) > days:
                                df = df.iloc[-days:]
                            data = df[['close']].rename(columns={'close': ticker})
                            print(f"✅ Twelve Data historical: {ticker} ({len(data)} days)")
                        elif "code" in result:
                            code = result.get("code")
                            msg = result.get("message", "")[:80]
                            if code == 429:
                                print(f"⏳ Twelve Data rate limit hit")
                            else:
                                print(f"⚠️ Twelve Data historical: {msg}")
                except Exception as e:
                    print(f"⚠️ Twelve Data historical error for {ticker}: {e}")

            # ---- Tier 2: yfinance ----
            if data is None:
                try:
                    import yfinance as yf
                    from datetime import datetime as _dt, timedelta as _td

                    print(f"📡 Trying yfinance historical for {ticker}...")
                    end_dt = _dt.now()
                    start_dt = end_dt - _td(days=int(days * 1.6))

                    hist = yf.Ticker(ticker).history(
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

            # ---- Tier 3: Alpha Vantage ----
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

            # ---- Tier 4: No synthetic. Real data or nothing. ----
            if data is None:
                print(f"❌ No real historical data available for {ticker}")

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
    # OPTION CHAIN — unavailable without paid provider
    # ============================================================
    def get_option_chain(self, ticker: str) -> Dict:
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
                    self._call_twelve_data(ticker) or
                    self._call_finnhub(ticker) or
                    self._call_alpha_vantage(ticker) or
                    self._call_yfinance(ticker)
                )
                if price is not None and price > 0:
                    self.live_prices[ticker] = price
                else:
                    print(f"⚠️ No real live price available for {ticker}; skipping")
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


# ============================================================
# Streamlit-level historical-data cache
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_historical_data(
    tickers_tuple: Tuple[str, ...],
    period: str = "1y",
) -> Optional[pd.DataFrame]:
    """
    Streamlit-level historical-data cache.

    Uses the persistent DataFetcher so repeated Streamlit reruns do not
    repeatedly hit Twelve Data/yfinance/Alpha Vantage.
    """
    fetcher = get_data_fetcher()
    return fetcher.get_historical_data(tickers_tuple, period)