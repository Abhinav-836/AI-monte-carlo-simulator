"""
Professional Data Fetcher — multi-tier fallback, cloud-safe.

Current price : Twelve Data -> Finnhub (US only) -> yfinance -> Yahoo chart API -> Alpha Vantage -> last real cached
Historical    : Twelve Data -> yfinance -> Yahoo chart API -> Alpha Vantage
Never fabricates prices. Missing tickers are reported in `failed_tickers` / `last_errors`.
"""

import os
import time
import threading
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=False)  # never let a stale local .env override cloud secrets

logger = logging.getLogger(__name__)

_YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# yfinance suffix -> Twelve Data exchange
_TD_EXCHANGES = {
    ".NS": "NSE", ".BO": "BSE", ".L": "LSE", ".HK": "HKEX",
    ".TO": "TSX", ".AX": "ASX", ".DE": "XETR",
}
# yfinance suffix -> Alpha Vantage suffix
_AV_SUFFIX = {".NS": ".BSE", ".BO": ".BSE", ".L": ".LON", ".TO": ".TRT", ".DE": ".DEX"}


def _get_key(name: str) -> str:
    try:
        val = st.secrets.get(name, "")
        if val and str(val).strip():
            return str(val).strip()
    except Exception:
        pass
    return (os.environ.get(name, "") or "").strip()


@st.cache_resource
def get_data_fetcher(use_live_simulation: bool = False):
    """One persistent DataFetcher per Streamlit server process (shared caches + rate limiter)."""
    return DataFetcher(use_live_simulation=use_live_simulation)


class DataFetcher:
    TD_PER_MINUTE = 7  # free tier is 8/min; keep one in reserve

    def __init__(self, use_live_simulation: bool = False):
        self.twelve_data_key = _get_key("TWELVE_DATA_API_KEY")
        self.alpha_vantage_key = _get_key("ALPHA_VANTAGE_API_KEY")
        self.finnhub_key = _get_key("FINNHUB_API_KEY")
        self.use_live_simulation = use_live_simulation

        self.current_source = "No Data"
        self.price_sources: Dict[str, str] = {}
        self.failed_tickers: List[str] = []
        self.last_errors: Dict[str, str] = {}

        self._price_subscribers = []
        self.is_running = False
        self.live_prices: Dict[str, float] = {}

        self.price_cache: Dict[str, float] = {}
        self.cache_timestamp: Dict[str, float] = {}
        self.cache_duration = 300

        self.historical_cache: Dict[str, pd.DataFrame] = {}
        self.historical_timestamp: Dict[str, float] = {}
        self.historical_ttl: Dict[str, float] = {}
        self.historical_cache_duration = 3600
        self.partial_cache_duration = 300  # partial results expire fast so failures get retried

        # Twelve Data: sliding-window limiter + symbols the plan doesn't cover
        self._td_lock = threading.Lock()
        self._td_calls: deque = deque()
        self._td_unsupported = set()

        self.finnhub_client = None
        if self.finnhub_key:
            try:
                import finnhub
                self.finnhub_client = finnhub.Client(api_key=self.finnhub_key)
            except Exception as e:
                print(f"⚠️ Finnhub init failed: {e}")

        print("✅ DataFetcher initialized | "
              f"TwelveData={'✅' if self.twelve_data_key else '❌'} "
              f"Finnhub={'✅' if self.finnhub_client else '❌'} "
              f"AlphaVantage={'✅' if self.alpha_vantage_key else '❌'}")

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    @staticmethod
    def _is_us_like(ticker: str) -> bool:
        return "." not in ticker and "-" not in ticker and "^" not in ticker

    @staticmethod
    def _twelve_params(ticker: str) -> Dict[str, str]:
        for suffix, exchange in _TD_EXCHANGES.items():
            if ticker.endswith(suffix):
                return {"symbol": ticker[: -len(suffix)], "exchange": exchange}
        if "-USD" in ticker:
            return {"symbol": ticker.replace("-", "/")}
        return {"symbol": ticker}

    @staticmethod
    def _alpha_symbol(ticker: str) -> str:
        for suffix, av in _AV_SUFFIX.items():
            if ticker.endswith(suffix):
                return ticker[: -len(suffix)] + av
        return ticker

    @staticmethod
    def _days_to_yahoo_range(days: int) -> str:
        for limit, rng in [(5, "5d"), (21, "1mo"), (63, "3mo"), (126, "6mo"),
                           (252, "1y"), (504, "2y"), (1260, "5y")]:
            if days <= limit:
                return rng
        return "10y"

    @staticmethod
    def _clean_series(s: pd.Series, name: str) -> pd.Series:
        s = pd.Series(s).astype(float).dropna()
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        s.index = idx.normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.name = name
        return s[s > 0]

    def _acquire_twelve_slot(self, max_wait: float = 15.0) -> bool:
        """Sliding-window limiter. Returns False if no slot within max_wait (caller falls through)."""
        deadline = time.time() + max_wait
        while True:
            with self._td_lock:
                now = time.time()
                while self._td_calls and now - self._td_calls[0] >= 60:
                    self._td_calls.popleft()
                if len(self._td_calls) < self.TD_PER_MINUTE:
                    self._td_calls.append(now)
                    return True
                wait = 60 - (now - self._td_calls[0]) + 0.1
            if time.time() + wait > deadline:
                return False
            time.sleep(wait)

    def _handle_td_error(self, ticker: str, data: dict):
        code = data.get("code")
        msg = str(data.get("message", ""))[:120]
        if code == 429:
            print("⏳ Twelve Data rate limit hit")
        elif code == 401:
            print("⚠️ Twelve Data: invalid API key")
        elif code in (400, 403, 404):
            self._td_unsupported.add(ticker)  # don't waste quota on it again
            print(f"⏭️ Twelve Data cannot serve {ticker} on this plan: {msg}")
        else:
            print(f"⚠️ Twelve Data {code}: {msg}")
        self.last_errors[ticker] = f"TwelveData {code}: {msg}"

    # ------------------------------------------------------------
    # CURRENT PRICE PROVIDERS
    # ------------------------------------------------------------
    def _call_twelve_data(self, symbol: str) -> Optional[float]:
        if not self.twelve_data_key or symbol in self._td_unsupported:
            return None
        if not self._acquire_twelve_slot():
            return None
        try:
            params = {**self._twelve_params(symbol), "apikey": self.twelve_data_key}
            r = requests.get("https://api.twelvedata.com/price", params=params, timeout=10)
            data = r.json()
            if "price" in data:
                p = float(data["price"])
                if p > 0:
                    return p
            elif "code" in data:
                self._handle_td_error(symbol, data)
        except Exception as e:
            self.last_errors[symbol] = f"TwelveData error: {e}"
        return None

    def _call_finnhub(self, symbol: str) -> Optional[float]:
        if not self.finnhub_client or not self._is_us_like(symbol):
            return None  # free tier is US-only
        try:
            q = self.finnhub_client.quote(symbol)
            p = q.get("c", 0) if isinstance(q, dict) else 0
            if p and p > 0:
                return float(p)
        except Exception as e:
            self.last_errors[symbol] = f"Finnhub error: {e}"
        return None

    def _call_yfinance(self, symbol: str) -> Optional[float]:
        try:
            import yfinance as yf
            t = yf.Ticker(symbol)
            try:
                p = t.fast_info.get("last_price")
                if p and p > 0:
                    return float(p)
            except Exception:
                pass
            hist = t.history(period="5d", auto_adjust=False)
            if not hist.empty:
                p = float(hist["Close"].dropna().iloc[-1])
                if p > 0:
                    return p
        except Exception as e:
            self.last_errors[symbol] = f"yfinance error: {e}"
        return None

    def _yahoo_chart(self, symbol: str, rng: str):
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": rng, "interval": "1d"},
            headers=_YAHOO_HEADERS, timeout=15,
        )
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        res = r.json()["chart"]["result"][0]
        close = res["indicators"]["quote"][0]["close"]
        s = pd.Series(close, index=pd.to_datetime(res["timestamp"], unit="s"), dtype=float)
        return s.dropna(), res.get("meta", {})

    def _call_yahoo_chart(self, symbol: str) -> Optional[float]:
        try:
            s, meta = self._yahoo_chart(symbol, "5d")
            p = meta.get("regularMarketPrice") or (float(s.iloc[-1]) if len(s) else None)
            if p and p > 0:
                return float(p)
        except Exception as e:
            self.last_errors[symbol] = f"Yahoo chart error: {e}"
        return None

    def _call_alpha_vantage(self, symbol: str) -> Optional[float]:
        if not self.alpha_vantage_key:
            return None
        try:
            r = requests.get(
                "https://www.alphavantage.co/query",
                params={"function": "GLOBAL_QUOTE", "symbol": self._alpha_symbol(symbol),
                        "apikey": self.alpha_vantage_key},
                timeout=10,
            )
            data = r.json()
            q = data.get("Global Quote") or {}
            if q.get("05. price"):
                return float(q["05. price"])
            if "Note" in data or "Information" in data:
                print("⏳ Alpha Vantage rate limit hit")
        except Exception as e:
            self.last_errors[symbol] = f"AlphaVantage error: {e}"
        return None

    def get_current_prices(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, float]:
        now = time.time()
        prices: Dict[str, float] = {}
        stale = []
        for t in tickers:
            if (not force_refresh and t in self.cache_timestamp
                    and now - self.cache_timestamp[t] < self.cache_duration):
                prices[t] = self.price_cache[t]
            else:
                stale.append(t)
        if not stale:
            return prices

        missing = []
        for t in stale:
            providers: List[Tuple[str, Callable]] = [
                ("Twelve Data", self._call_twelve_data),
                ("Finnhub", self._call_finnhub),
                ("yfinance", self._call_yfinance),
                ("Yahoo chart", self._call_yahoo_chart),
                ("Alpha Vantage", self._call_alpha_vantage),
            ]
            price, source = None, None
            for name, fn in providers:
                price = fn(t)
                if price:
                    source = name
                    break

            if price:
                prices[t] = price
                self.price_cache[t] = price
                self.cache_timestamp[t] = now
                self.price_sources[t] = source
                self.current_source = source
            elif t in self.price_cache:  # last real price; timestamp untouched so we retry next call
                prices[t] = self.price_cache[t]
                self.price_sources[t] = "Cached (stale)"
            else:
                missing.append(t)

        if missing:
            print(f"⚠️ No live market data for: {missing}")
            self.current_source = "No Data" if not prices else "Partial Data"
        return prices

    # ------------------------------------------------------------
    # HISTORICAL PROVIDERS (each returns a clean Series or None)
    # ------------------------------------------------------------
    def _hist_twelve(self, ticker: str, days: int) -> Optional[pd.Series]:
        if not self.twelve_data_key or ticker in self._td_unsupported:
            return None
        if not self._acquire_twelve_slot():
            return None
        params = {**self._twelve_params(ticker), "interval": "1day",
                  "outputsize": min(days, 5000), "apikey": self.twelve_data_key}
        data = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=15).json()
        if "values" not in data:
            if "code" in data:
                self._handle_td_error(ticker, data)
            return None
        df = pd.DataFrame(data["values"])
        s = pd.Series(df["close"].astype(float).values, index=pd.to_datetime(df["datetime"]))
        return self._clean_series(s, ticker).iloc[-days:]

    def _hist_yfinance(self, ticker: str, days: int) -> Optional[pd.Series]:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=self._days_to_yahoo_range(days),
                                         interval="1d", auto_adjust=False)
        if hist.empty or "Close" not in hist.columns:
            return None
        return self._clean_series(hist["Close"], ticker).iloc[-days:]

    def _hist_yahoo_chart(self, ticker: str, days: int) -> Optional[pd.Series]:
        s, _ = self._yahoo_chart(ticker, self._days_to_yahoo_range(days))
        return self._clean_series(s, ticker).iloc[-days:] if len(s) else None

    def _hist_alpha(self, ticker: str, days: int) -> Optional[pd.Series]:
        if not self.alpha_vantage_key:
            return None
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "TIME_SERIES_DAILY", "symbol": self._alpha_symbol(ticker),
                    "outputsize": "compact", "apikey": self.alpha_vantage_key},
            timeout=15,
        ).json()
        ts = r.get("Time Series (Daily)")
        if not ts:
            return None
        df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
        return self._clean_series(df["4. close"], ticker).iloc[-days:]

    def _fetch_history_one(self, ticker: str, days: int) -> Optional[pd.Series]:
        tiers = [("Twelve Data", self._hist_twelve), ("yfinance", self._hist_yfinance),
                 ("Yahoo chart", self._hist_yahoo_chart), ("Alpha Vantage", self._hist_alpha)]
        for name, fn in tiers:
            try:
                s = fn(ticker, days)
            except Exception as e:
                self.last_errors[ticker] = f"{name}: {e}"
                s = None
            if s is not None and len(s) >= 20:
                print(f"✅ {name} historical: {ticker} ({len(s)} days)")
                return s
        print(f"❌ No real historical data for {ticker}: {self.last_errors.get(ticker, 'all tiers failed')}")
        return None

    def get_historical_data(self, tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
        key = f"{tickers_tuple}_{period}"
        now = time.time()
        if key in self.historical_timestamp:
            if now - self.historical_timestamp[key] < self.historical_ttl.get(key, self.historical_cache_duration):
                return self.historical_cache[key].copy()

        days = self._period_to_days(period)
        series, failed = [], []
        for ticker in tickers_tuple:
            s = self._fetch_history_one(ticker, days)
            (series if s is not None else failed).append(s if s is not None else ticker)

        self.failed_tickers = failed
        if not series:
            print("❌ No historical data available")
            return None

        combined = pd.concat(series, axis=1).sort_index().ffill()
        clean = combined.dropna()  # no bfill: it would copy future prices into the past
        combined = clean if len(clean) >= 30 else combined.bfill()

        self.historical_cache[key] = combined
        self.historical_timestamp[key] = now
        self.historical_ttl[key] = (self.partial_cache_duration if failed
                                    else self.historical_cache_duration)
        if failed:
            print(f"⚠️ Partial data. Missing: {failed}")
        return combined.copy()

    @staticmethod
    def _period_to_days(period: str) -> int:
        if period.endswith("mo"):
            return int(period[:-2]) * 21
        if period.endswith("d"):
            return int(period[:-1])
        if period.endswith("y"):
            return int(period[:-1]) * 252
        return 252

    # ------------------------------------------------------------
    def get_option_chain(self, ticker: str) -> Dict:
        return {}

    def get_data_source_info(self) -> str:
        return self.current_source

    # ------------------------------------------------------------
    # LIVE SIMULATION (random walk around the last REAL price; uses cache, no extra API calls)
    # ------------------------------------------------------------
    def start_live_simulation(self, tickers: List[str]):
        if not self.use_live_simulation:
            return
        self.live_prices = {t: p for t, p in self.get_current_prices(tickers).items() if p > 0}
        self.is_running = True
        self.current_source = "Live Simulation"

    def stop_live_simulation(self):
        self.is_running = False

    def subscribe_prices(self, callback):
        self._price_subscribers.append(callback)

    def get_live_prices(self) -> Dict[str, float]:
        if not self.is_running:
            return {}
        for t in self.live_prices:
            self.live_prices[t] = max(
                self.live_prices[t] * (1 + np.random.uniform(-0.005, 0.005)), 0.01)
        snapshot = dict(self.live_prices)
        for cb in list(self._price_subscribers):
            try:
                cb(snapshot)
            except Exception as e:
                logger.warning(f"price subscriber failed: {e}")
        return snapshot

    def get_price_history(self, ticker: str) -> List[Dict]:
        if ticker in self.live_prices:
            return [{"time": datetime.now() - timedelta(minutes=i),
                     "price": self.live_prices[ticker] * (1 + np.random.uniform(-0.01, 0.01))}
                    for i in range(20)]
        return []


def get_cached_historical_data(tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Uses the fetcher's own cache, which only stores successful results.
    (st.cache_data would cache a failed `None` for an hour — that was hiding markets.)
    """
    return get_data_fetcher().get_historical_data(tickers_tuple, period)