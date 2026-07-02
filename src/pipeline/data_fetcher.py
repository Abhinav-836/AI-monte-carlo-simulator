"""
Professional Data Fetcher with Alpha Vantage + Yahoo Finance Fallback
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import threading
import json
from collections import deque
import os
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """Alpha Vantage API Client for real-time and historical data"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.base_url = "https://www.alphavantage.co/query"
        self.last_call = 0
        self.min_interval = 12  # Alpha Vantage free tier: 5 calls per minute = 12 seconds between calls
        
    def _rate_limit(self):
        """Apply rate limiting for Alpha Vantage API"""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for a symbol"""
        if not self.api_key:
            return None
            
        try:
            self._rate_limit()
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "Global Quote" in data and data["Global Quote"]:
                    quote = data["Global Quote"]
                    return {
                        "symbol": quote.get("01. symbol", symbol),
                        "price": float(quote.get("05. price", 0)),
                        "change": float(quote.get("09. change", 0)),
                        "change_percent": float(quote.get("10. change percent", "0%").strip("%")),
                        "volume": int(quote.get("06. volume", 0)),
                        "timestamp": quote.get("07. latest trading day", datetime.now().strftime("%Y-%m-%d"))
                    }
        except Exception as e:
            logger.warning(f"Alpha Vantage quote error for {symbol}: {e}")
        
        return None
    
    def get_historical(self, symbol: str, output_size: str = "full") -> Optional[pd.DataFrame]:
        """Get historical data from Alpha Vantage"""
        if not self.api_key:
            return None
            
        try:
            self._rate_limit()
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": output_size,
                "apikey": self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if "Time Series (Daily)" in data:
                    time_series = data["Time Series (Daily)"]
                    df = pd.DataFrame.from_dict(time_series, orient="index")
                    df.index = pd.to_datetime(df.index)
                    df = df.astype(float)
                    df.columns = ["Open", "High", "Low", "Close", "Volume"]
                    df = df.sort_index()
                    return df
        except Exception as e:
            logger.warning(f"Alpha Vantage historical error for {symbol}: {e}")
        
        return None
    
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """Get quotes for multiple symbols"""
        prices = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote and quote.get("price", 0) > 0:
                prices[symbol] = quote["price"]
        return prices


class YahooFinanceClient:
    """Yahoo Finance Client as Fallback"""
    
    def __init__(self):
        self.last_call = 0
        self.min_interval = 1  # 1 second between calls
    
    def _rate_limit(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def get_quote(self, symbol: str) -> Optional[float]:
        """Get real-time quote from Yahoo Finance"""
        try:
            self._rate_limit()
            stock = yf.Ticker(symbol)
            info = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('ask') or info.get('previousClose')
            if price and price > 0:
                return float(price)
        except Exception as e:
            logger.warning(f"Yahoo Finance quote error for {symbol}: {e}")
        return None
    
    def get_historical(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data from Yahoo Finance"""
        try:
            self._rate_limit()
            data = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
            if not data.empty:
                return data
        except Exception as e:
            logger.warning(f"Yahoo Finance historical error for {symbol}: {e}")
        return None
    
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """Get quotes for multiple symbols"""
        prices = {}
        for symbol in symbols:
            price = self.get_quote(symbol)
            if price:
                prices[symbol] = price
        return prices


class DataFetcher:
    """
    Advanced Data Fetcher with Alpha Vantage Primary + Yahoo Finance Fallback
    """
    
    def __init__(self, use_live_simulation: bool = False):
        # Initialize clients
        self.alpha_vantage = AlphaVantageClient()
        self.yahoo_finance = YahooFinanceClient()
        
        # Track which source is being used
        self.primary_source = "Alpha Vantage" if self.alpha_vantage.api_key else "Yahoo Finance"
        self.current_source = self.primary_source
        self.fallback_used = False
        
        # Enhanced caching
        self.price_cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 30  # 30 seconds for live data
        
        self.historical_cache = {}
        self.historical_timestamp = {}
        self.historical_cache_duration = 3600  # 1 hour
        
        # Live simulation
        self.use_live_simulation = use_live_simulation
        self.live_simulator = None
        
        # Default prices
        self.default_prices = self._load_default_prices()
        
        # Failed requests tracking
        self.failed_requests = {}
        self.failed_cooldown = 300
        
        logger.info(f"✅ DataFetcher initialized (Primary: {self.primary_source})")

    def _load_default_prices(self) -> Dict[str, float]:
        """Load default prices as ultimate fallback"""
        return {
            # US Stocks
            'AAPL': 175.32, 'MSFT': 380.45, 'GOOGL': 142.18,
            'AMZN': 178.50, 'META': 472.30, 'TSLA': 198.75,
            'NVDA': 820.15, 'JPM': 152.80, 'V': 255.60,
            'WMT': 60.25, 'JNJ': 155.30, 'PG': 160.45,
            'UNH': 490.20, 'HD': 360.15, 'DIS': 110.30,
            'MA': 440.25, 'BAC': 35.80, 'NFLX': 620.45,
            'ADBE': 525.30, 'CRM': 290.15, 'AMD': 180.25,
            'INTC': 42.50, 'CSCO': 52.75, 'PEP': 170.30,
            'COST': 700.25, 'CVX': 155.40, 'WFC': 55.20,
            'QCOM': 165.30, 'TMO': 550.15, 'ABT': 110.25,
            
            # Indian Stocks
            'RELIANCE.NS': 2500.00, 'TCS.NS': 3500.00, 'HDFCBANK.NS': 1600.00,
            'INFY.NS': 1450.00, 'ICICIBANK.NS': 1050.00, 'SBIN.NS': 600.00,
            'BHARTIARTL.NS': 850.00, 'ITC.NS': 400.00, 'WIPRO.NS': 500.00,
            'HINDUNILVR.NS': 2500.00, 'TITAN.NS': 2800.00, 'BAJFINANCE.NS': 7000.00,
            'MARUTI.NS': 9500.00, 'SUNPHARMA.NS': 1200.00, 'ONGC.NS': 150.00,
            
            # Crypto
            'BTC-USD': 65000, 'ETH-USD': 3500, 'BNB-USD': 450,
            'XRP-USD': 0.75, 'ADA-USD': 0.45, 'DOGE-USD': 0.12,
            'SOL-USD': 150.00, 'DOT-USD': 8.00, 'MATIC-USD': 0.80,
        }

    def get_current_prices(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """Get current prices with Alpha Vantage primary + Yahoo fallback"""
        
        now = time.time()
        prices = {}
        
        # Check cache first
        stale_tickers = []
        for ticker in tickers:
            if not force_refresh and ticker in self.cache_timestamp:
                if now - self.cache_timestamp[ticker] < self.cache_duration:
                    prices[ticker] = self.price_cache[ticker]
                    continue
            stale_tickers.append(ticker)
        
        if not stale_tickers:
            return prices
        
        # Try Alpha Vantage first
        self.current_source = "Alpha Vantage"
        self.fallback_used = False
        
        # Try batch fetch from Alpha Vantage
        av_prices = self.alpha_vantage.get_batch_quotes(stale_tickers)
        
        for ticker in stale_tickers:
            if ticker in av_prices and av_prices[ticker] > 0:
                prices[ticker] = av_prices[ticker]
                self.price_cache[ticker] = av_prices[ticker]
                self.cache_timestamp[ticker] = now
            else:
                # Try Yahoo Finance as fallback
                yf_price = self.yahoo_finance.get_quote(ticker)
                if yf_price and yf_price > 0:
                    prices[ticker] = yf_price
                    self.price_cache[ticker] = yf_price
                    self.cache_timestamp[ticker] = now
                    self.fallback_used = True
                    self.current_source = "Yahoo Finance (Fallback)"
                else:
                    # Use default as ultimate fallback
                    prices[ticker] = self.default_prices.get(ticker, 100)
                    self.price_cache[ticker] = prices[ticker]
                    self.cache_timestamp[ticker] = now
        
        # Update source tracking
        if self.fallback_used:
            self.current_source = "Yahoo Finance (Fallback)"
        
        return prices

    def get_historical_data(self, tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data with Alpha Vantage primary + Yahoo fallback"""
        
        cache_key = f"{tickers_tuple}_{period}"
        now = time.time()
        
        # Check cache
        if cache_key in self.historical_timestamp:
            if now - self.historical_timestamp[cache_key] < self.historical_cache_duration:
                logger.info(f"📦 Using cached historical data")
                return self.historical_cache[cache_key]
        
        tickers = list(tickers_tuple)
        logger.info(f"📥 Fetching fresh historical data for {len(tickers)} stocks")
        
        all_data = []
        self.fallback_used = False
        
        for ticker in tickers:
            # Try Alpha Vantage first
            av_data = self.alpha_vantage.get_historical(ticker, "full")
            
            if av_data is not None and not av_data.empty:
                # Alpha Vantage returns daily data, we need to filter by period
                if period == "6mo":
                    av_data = av_data.last(126)  # ~6 months
                elif period == "1y":
                    av_data = av_data.last(252)  # ~1 year
                elif period == "2y":
                    av_data = av_data.last(504)  # ~2 years
                elif period == "5y":
                    av_data = av_data.last(1260)  # ~5 years
                
                all_data.append(av_data[['Close']].rename(columns={'Close': ticker}))
                logger.info(f"✅ Alpha Vantage: {ticker} ({len(av_data)} days)")
            else:
                # Fallback to Yahoo Finance
                yf_data = self.yahoo_finance.get_historical(ticker, period)
                if yf_data is not None and not yf_data.empty:
                    if 'Close' in yf_data.columns:
                        all_data.append(yf_data[['Close']].rename(columns={'Close': ticker}))
                        self.fallback_used = True
                        logger.info(f"✅ Yahoo Fallback: {ticker} ({len(yf_data)} days)")
                else:
                    # Generate synthetic data as last resort
                    logger.warning(f"⚠️ Generating synthetic data for {ticker}")
                    synthetic = self._generate_synthetic_data([ticker], period)
                    all_data.append(synthetic)
        
        if not all_data:
            logger.warning(f"⚠️ No data available, using synthetic")
            return self._generate_synthetic_data(tickers, period)
        
        # Combine all data
        try:
            combined = pd.concat(all_data, axis=1)
            combined = combined.ffill().bfill()
            
            # Cache
            self.historical_cache[cache_key] = combined
            self.historical_timestamp[cache_key] = now
            
            # Update source info
            if self.fallback_used:
                self.current_source = "Yahoo Finance (Alpha Vantage Fallback)"
            else:
                self.current_source = "Alpha Vantage"
            
            logger.info(f"✅ Successfully fetched data for {len(combined.columns)} stocks")
            return combined
            
        except Exception as e:
            logger.error(f"❌ Data combination failed: {e}")
            return self._generate_synthetic_data(tickers, period)

    def _generate_synthetic_data(self, tickers: List[str], period: str) -> pd.DataFrame:
        """Generate synthetic data as ultimate fallback"""
        logger.info(f"🔄 Generating synthetic data for {tickers}")
        
        days = self._period_to_days(period)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
        
        data = {}
        for ticker in tickers:
            base_price = self.default_prices.get(ticker, 100)
            returns = np.random.randn(days) * 0.02 + 0.0003
            prices = base_price * np.exp(np.cumsum(returns))
            prices = np.maximum(prices, base_price * 0.3)
            data[ticker] = prices
        
        self.current_source = "Synthetic (Fallback)"
        return pd.DataFrame(data, index=dates)

    def _period_to_days(self, period: str) -> int:
        """Convert period to days"""
        if period.endswith('d'):
            return int(period[:-1])
        elif period.endswith('mo'):
            return int(period[:-2]) * 21
        elif period.endswith('y'):
            return int(period[:-1]) * 252
        return 252

    def get_option_chain(self, ticker: str) -> Dict:
        """Get option chain (Yahoo Finance only - Alpha Vantage doesn't provide this)"""
        return self.yahoo_finance.get_option_chain(ticker) if hasattr(self.yahoo_finance, 'get_option_chain') else {}

    def get_data_source_info(self) -> str:
        """Get info about current data source"""
        return self.current_source

    def get_market_summary(self) -> Dict:
        """Get market summary with key indices (Yahoo Finance)"""
        return self.yahoo_finance.get_market_summary() if hasattr(self.yahoo_finance, 'get_market_summary') else {}