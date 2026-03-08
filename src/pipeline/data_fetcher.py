"""
Professional Data Fetcher with Aggressive Rate Limiting
Handles 429 errors gracefully with multiple fallback strategies
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

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Strict rate limiter to prevent 429 errors
    """
    def __init__(self, max_calls_per_minute=20):  # Reduced to 20/min to be safe
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we've exceeded rate limit"""
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if now - t < 60]
            
            if len(self.calls) >= self.max_calls_per_minute:
                # Wait until we can make another call
                oldest = min(self.calls)
                wait_time = 60 - (now - oldest) + 2
                logger.info(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                # Clear old calls after waiting
                self.calls = []
            
            self.calls.append(now)


class DataFetcher:
    """
    Ultra-reliable financial data fetcher with aggressive rate limiting
    """

    def __init__(self):
        # Strict rate limiting - 20 calls per minute max
        self.rate_limiter = RateLimiter(max_calls_per_minute=20)
        
        # Data source tracking
        self.source_used = "Yahoo Finance"
        
        # Cache for current prices - 30 minute cache to reduce calls
        self.price_cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 1800  # 30 minutes
        
        # Cache for historical data - 6 hour cache
        self.historical_cache = {}
        self.historical_timestamp = {}
        self.historical_cache_duration = 21600  # 6 hours
        
        # Extended default prices for fallback
        self.default_prices = self._load_default_prices()
        
        # Track failed requests to avoid hammering
        self.failed_requests = {}
        self.failed_cooldown = 3600  # 1 hour cooldown for failed tickers
        
        logger.info("✅ DataFetcher initialized with aggressive rate limiting")

    def _load_default_prices(self) -> Dict[str, float]:
        """Load extended default prices for global stocks"""
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
            
            # Indian Stocks (NSE)
            'RELIANCE.NS': 2500.00, 'TCS.NS': 3500.00, 'HDFCBANK.NS': 1600.00,
            'INFY.NS': 1450.00, 'ICICIBANK.NS': 1050.00, 'SBIN.NS': 600.00,
            'BHARTIARTL.NS': 850.00, 'ITC.NS': 400.00, 'WIPRO.NS': 500.00,
            'HINDUNILVR.NS': 2500.00, 'TITAN.NS': 2800.00, 'BAJFINANCE.NS': 7000.00,
            'MARUTI.NS': 9500.00, 'SUNPHARMA.NS': 1200.00, 'ONGC.NS': 150.00,
            'NTPC.NS': 200.00, 'POWERGRID.NS': 220.00, 'ULTRACEMCO.NS': 8000.00,
            'HCLTECH.NS': 1200.00, 'TECHM.NS': 1100.00, 'ASIANPAINT.NS': 3200.00,
            'AXISBANK.NS': 1000.00, 'KOTAKBANK.NS': 1800.00, 'INDUSINDBK.NS': 1400.00,
            
            # Indian Stocks (BSE)
            'RELIANCE.BO': 2495.00, 'TCS.BO': 3495.00, 'HDFCBANK.BO': 1595.00,
            
            # UK Stocks
            'BP.L': 450.00, 'HSBA.L': 650.00, 'GSK.L': 1400.00,
            'AZN.L': 10500.00, 'DGE.L': 2800.00, 'LLOY.L': 45.00,
            'BARC.L': 150.00, 'VOD.L': 130.00, 'RIO.L': 5000.00,
            'GLEN.L': 400.00, 'AAL.L': 2200.00, 'PRU.L': 800.00,
            
            # Canadian Stocks
            'SHOP.TO': 90.00, 'RY.TO': 135.00, 'TD.TO': 82.00,
            'ENB.TO': 50.00, 'CNR.TO': 160.00, 'CP.TO': 90.00,
            'BNS.TO': 65.00, 'BMO.TO': 120.00, 'CM.TO': 60.00,
            'SU.TO': 45.00, 'TRP.TO': 55.00, 'CVE.TO': 20.00,
            
            # European Stocks
            'SAP.DE': 150.00, 'AIR.PA': 140.00, 'NESN.SW': 95.00,
            'MC.PA': 700.00, 'OR.PA': 45.00, 'TTE.PA': 60.00,
            'SAN.PA': 85.00, 'BNP.PA': 60.00, 'DBK.DE': 12.00,
            'ALV.DE': 220.00, 'SIEGY': 150.00, 'VOW3.DE': 120.00,
            'ASML.AS': 800.00, 'INGA.AS': 14.00, 'PHG.AS': 20.00,
            
            # Asian Stocks
            '0700.HK': 350.00, '9988.HK': 80.00, '7203.T': 2500.00,
            '005930.KS': 70000.00, 'BABA': 75.00, 'JD': 25.00,
            'BIDU': 100.00, 'NTES': 90.00, 'TCEHY': 50.00,
            'SFTBY': 25.00, 'NIO': 5.00, 'LI': 25.00,
            'XPEV': 10.00, 'BILI': 12.00, 'IQ': 3.00,
            
            # ETFs
            'SPY': 480.25, 'QQQ': 420.75, 'IWM': 210.50,
            'VOO': 450.00, 'IVV': 470.00, 'VTI': 240.00,
            'EFA': 75.00, 'EEM': 40.00, 'AGG': 95.00,
            'GLD': 185.40, 'SLV': 22.00, 'TLT': 95.30,
            
            # Crypto
            'BTC-USD': 65000, 'ETH-USD': 3500, 'BNB-USD': 450,
            'XRP-USD': 0.75, 'ADA-USD': 0.45, 'DOGE-USD': 0.12,
            'SOL-USD': 150.00, 'DOT-USD': 8.00, 'MATIC-USD': 0.80,
        }

    def _is_rate_limited(self, ticker: str) -> bool:
        """Check if a ticker is in cooldown due to rate limiting"""
        if ticker in self.failed_requests:
            cooldown_until = self.failed_requests[ticker]
            if time.time() < cooldown_until:
                return True
            else:
                # Clear expired cooldown
                del self.failed_requests[ticker]
        return False

    def _mark_rate_limited(self, ticker: str):
        """Mark a ticker as rate limited for cooldown"""
        self.failed_requests[ticker] = time.time() + self.failed_cooldown

    def _strict_rate_limit(self):
        """Strict rate limiting - ensures we never hit Yahoo's limits"""
        self.rate_limiter.wait_if_needed()

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter"""
        base_delay = 5  # Start with 5 seconds
        delay = base_delay * (2 ** attempt) + random.uniform(1, 5)
        return min(delay, 60)  # Cap at 60 seconds

    def _fetch_with_retry(self, func, *args, ticker: str = None, **kwargs):
        """
        Generic fetch function with exponential backoff
        """
        max_retries = 3
        
        # Check if ticker is rate limited
        if ticker and self._is_rate_limited(ticker):
            logger.info(f"⏸️ {ticker} in cooldown, skipping")
            return None
        
        for attempt in range(max_retries):
            try:
                self._strict_rate_limit()
                result = func(*args, **kwargs)
                
                # If successful and ticker was rate limited, clear it
                if ticker and ticker in self.failed_requests:
                    del self.failed_requests[ticker]
                
                return result
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"⚠️ Attempt {attempt + 1} for {ticker or 'request'} failed: {error_str[:100]}")
                
                if "429" in error_str or "Too Many Requests" in error_str:
                    # Mark ticker as rate limited
                    if ticker:
                        self._mark_rate_limited(ticker)
                    
                    # Exponential backoff
                    delay = self._exponential_backoff(attempt)
                    logger.info(f"⏳ Rate limited. Waiting {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
        
        return None

    def get_historical_data(self, tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Get historical data with aggressive caching
        """
        cache_key = f"{tickers_tuple}_{period}"
        now = time.time()
        
        # Check cache first
        if cache_key in self.historical_timestamp:
            if now - self.historical_timestamp[cache_key] < self.historical_cache_duration:
                logger.info(f"📦 Using cached historical data for {len(tickers_tuple)} stocks")
                return self.historical_cache[cache_key]
        
        tickers = list(tickers_tuple)
        logger.info(f"📥 Fetching fresh historical data for {len(tickers)} stocks")
        
        # Split into chunks to avoid rate limits
        chunk_size = 3  # Fetch 3 tickers at a time
        all_data = []
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            
            def fetch_func():
                return yf.download(
                    chunk,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    timeout=30
                )
            
            chunk_data = self._fetch_with_retry(fetch_func)
            
            if chunk_data is not None and not chunk_data.empty:
                all_data.append(chunk_data)
            
            # Extra delay between chunks
            if i + chunk_size < len(tickers):
                time.sleep(3)
        
        if not all_data:
            logger.warning(f"⚠️ No data for {tickers}, using synthetic")
            return self._generate_synthetic_data(tickers, period)
        
        # Combine all chunks
        try:
            # Handle different data formats
            combined_data = pd.DataFrame()
            
            for data in all_data:
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Close' in data.columns.levels[0]:
                        close_data = data['Close']
                    else:
                        close_data = data.xs('Close', axis=1, level=0, drop_level=False)
                else:
                    if 'Close' in data.columns:
                        close_data = data[['Close']]
                        close_data.columns = [col for col in data.columns if col != 'Close']
                    else:
                        close_data = data
                
                for col in close_data.columns:
                    combined_data[col] = close_data[col]
            
            combined_data = combined_data.ffill().bfill()
            
            # Cache the result
            self.historical_cache[cache_key] = combined_data
            self.historical_timestamp[cache_key] = now
            self.source_used = "Yahoo Finance"
            
            logger.info(f"✅ Successfully fetched data for {len(combined_data.columns)} stocks")
            return combined_data
            
        except Exception as e:
            logger.error(f"❌ Data cleaning failed: {e}")
            return self._generate_synthetic_data(tickers, period)

    def _generate_synthetic_data(self, tickers: List[str], period: str) -> pd.DataFrame:
        """
        Generate synthetic data when Yahoo fails
        """
        logger.info(f"🔄 Generating synthetic data for {tickers}")
        self.source_used = "Synthetic (Fallback)"
        
        days = self._period_to_days(period)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
        
        data = {}
        for ticker in tickers:
            base_price = self.default_prices.get(ticker, 100)
            # Generate realistic random walk with mean reversion
            returns = np.random.randn(days) * 0.02
            # Add slight positive drift
            returns = returns + 0.0003
            # Add mean reversion
            for i in range(1, len(returns)):
                returns[i] = returns[i] - 0.01 * (np.sum(returns[:i]) / (i + 1))
            
            prices = base_price * np.exp(np.cumsum(returns))
            # Ensure prices are positive
            prices = np.maximum(prices, base_price * 0.5)
            data[ticker] = prices
        
        return pd.DataFrame(data, index=dates)

    def _period_to_days(self, period: str) -> int:
        """Convert Yahoo period to days"""
        if period.endswith('d'):
            return int(period[:-1])
        elif period.endswith('mo'):
            return int(period[:-2]) * 21
        elif period.endswith('y'):
            return int(period[:-1]) * 252
        return 252

    def get_current_prices(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """
        Get current prices with 30-minute cache
        """
        now = time.time()
        prices = {}
        
        # Check cache first
        fresh_tickers = []
        stale_tickers = []
        
        for ticker in tickers:
            if not force_refresh and ticker in self.cache_timestamp:
                if now - self.cache_timestamp[ticker] < self.cache_duration:
                    fresh_tickers.append(ticker)
                    prices[ticker] = self.price_cache[ticker]
                    continue
            stale_tickers.append(ticker)
        
        if fresh_tickers:
            logger.info(f"📦 Using cached prices for {len(fresh_tickers)} stocks")
        
        # Fetch stale tickers
        if stale_tickers:
            logger.info(f"📡 Fetching fresh prices for {len(stale_tickers)} stocks")
            
            for ticker in stale_tickers:
                price = self._fetch_single_price(ticker)
                if price is not None:
                    prices[ticker] = price
                    self.price_cache[ticker] = price
                    self.cache_timestamp[ticker] = now
                else:
                    # Fallback to default
                    prices[ticker] = self.default_prices.get(ticker, 100)
                    logger.info(f"📋 Using default price for {ticker}")
        
        return prices

    def _fetch_single_price(self, ticker: str) -> Optional[float]:
        """
        Fetch price for a single ticker with aggressive retry logic
        """
        def fetch_func():
            stock = yf.Ticker(ticker)
            
            # Try multiple price sources
            try:
                # Try fast info first (fewer API calls)
                fast_info = stock.fast_info
                price = fast_info.get("last_price")
                if price and price > 0:
                    return float(price)
            except:
                pass
            
            # Wait a bit before trying full info
            time.sleep(1)
            
            # Fallback to regular info
            try:
                info = stock.info
                price = (
                    info.get("currentPrice") or
                    info.get("regularMarketPrice") or
                    info.get("ask") or
                    info.get("bid") or
                    info.get("previousClose")
                )
                
                if price and price > 0:
                    return float(price)
            except:
                pass
            
            return None
        
        result = self._fetch_with_retry(fetch_func, ticker=ticker)
        
        if result:
            self.source_used = "Yahoo Finance (Live)"
            logger.info(f"✅ Fetched {ticker}: {result:.2f}")
        else:
            logger.warning(f"❌ Could not fetch {ticker}")
        
        return result

    def get_option_chain(self, ticker: str) -> Dict:
        """
        Fetch option chain for US stocks only
        """
        # Skip non-US stocks and rate-limited tickers
        if '.' in ticker or self._is_rate_limited(ticker):
            return {}
        
        def fetch_func():
            stock = yf.Ticker(ticker)
            
            try:
                expirations = stock.options
                if not expirations:
                    return {}
                
                # Get nearest expiration
                chain = stock.option_chain(expirations[0])
                
                # Format the data
                calls = []
                for _, row in chain.calls.head(10).iterrows():
                    calls.append({
                        'strike': float(row.get('strike', 0)),
                        'lastPrice': float(row.get('lastPrice', 0)),
                        'bid': float(row.get('bid', 0)),
                        'ask': float(row.get('ask', 0)),
                        'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                        'impliedVolatility': float(row.get('impliedVolatility', 0))
                    })
                
                puts = []
                for _, row in chain.puts.head(10).iterrows():
                    puts.append({
                        'strike': float(row.get('strike', 0)),
                        'lastPrice': float(row.get('lastPrice', 0)),
                        'bid': float(row.get('bid', 0)),
                        'ask': float(row.get('ask', 0)),
                        'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                        'impliedVolatility': float(row.get('impliedVolatility', 0))
                    })
                
                return {
                    'calls': calls,
                    'puts': puts,
                    'expiration': expirations[0],
                    'underlying_price': float(chain.underlying['price']) if 'price' in chain.underlying else 0
                }
                
            except Exception as e:
                logger.warning(f"Option chain error for {ticker}: {e}")
                return {}
        
        result = self._fetch_with_retry(fetch_func, ticker=ticker)
        return result or {}

    def get_data_source_info(self) -> str:
        """Get info about data source used"""
        return self.source_used