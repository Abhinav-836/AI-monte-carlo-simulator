"""
Professional Data Fetcher with Alpha Vantage Primary + Yahoo Finance Fallback
FIXED for Streamlit Cloud deployment
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
        self.min_interval = 12  # Free tier: 5 calls per minute
        self.has_api_key = bool(self.api_key and self.api_key != "your_alpha_vantage_api_key_here")
        
        if self.has_api_key:
            logger.info("✅ Alpha Vantage API key found")
        else:
            logger.warning("⚠️ No Alpha Vantage API key found - using fallback only")
        
    def _rate_limit(self):
        """Apply rate limiting for Alpha Vantage API"""
        if not self.has_api_key:
            return
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for a symbol"""
        if not self.has_api_key:
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
                    price = quote.get("05. price")
                    if price and float(price) > 0:
                        return {
                            "symbol": quote.get("01. symbol", symbol),
                            "price": float(price),
                            "change": float(quote.get("09. change", 0)),
                            "change_percent": float(quote.get("10. change percent", "0%").strip("%")),
                            "volume": int(quote.get("06. volume", 0)),
                            "timestamp": quote.get("07. latest trading day", datetime.now().strftime("%Y-%m-%d"))
                        }
            else:
                logger.warning(f"Alpha Vantage API returned status {response.status_code} for {symbol}")
        except Exception as e:
            logger.warning(f"Alpha Vantage quote error for {symbol}: {e}")
        
        return None
    
    def get_historical(self, symbol: str, output_size: str = "compact") -> Optional[pd.DataFrame]:
        """Get historical data from Alpha Vantage"""
        if not self.has_api_key:
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
                elif "Note" in data:
                    logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                elif "Error Message" in data:
                    logger.warning(f"Alpha Vantage error: {data['Error Message']}")
        except Exception as e:
            logger.warning(f"Alpha Vantage historical error for {symbol}: {e}")
        
        return None
    
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """Get quotes for multiple symbols"""
        if not self.has_api_key:
            return {}
            
        prices = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote and quote.get("price", 0) > 0:
                prices[symbol] = quote["price"]
            # Add delay between symbols for rate limiting
            time.sleep(1)
        return prices


class YahooFinanceClient:
    """Yahoo Finance Client as Fallback - Reduced calls to avoid rate limiting"""
    
    def __init__(self):
        self.last_call = 0
        self.min_interval = 2  # Increased to 2 seconds to avoid rate limiting
    
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
            # Try fast_info first (faster, fewer API calls)
            try:
                fast_info = stock.fast_info
                price = fast_info.get("last_price")
                if price and price > 0:
                    return float(price)
            except:
                pass
            
            # Fallback to regular info
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
            data = yf.download(
                symbol, 
                period=period, 
                interval="1d", 
                progress=False, 
                auto_adjust=True,
                threads=False,
                timeout=10
            )
            if not data.empty:
                return data
        except Exception as e:
            logger.warning(f"Yahoo Finance historical error for {symbol}: {e}")
        return None
    
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, float]:
        """Get quotes for multiple symbols - rate limited"""
        prices = {}
        for symbol in symbols:
            price = self.get_quote(symbol)
            if price:
                prices[symbol] = price
            time.sleep(1)  # Delay between symbols
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
        self.primary_source = "Alpha Vantage" if self.alpha_vantage.has_api_key else "Yahoo Finance"
        self.current_source = self.primary_source
        self.fallback_used = False
        
        # Enhanced caching
        self.price_cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 60  # 60 seconds for live data
        
        self.historical_cache = {}
        self.historical_timestamp = {}
        self.historical_cache_duration = 1800  # 30 minutes
        
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
            'RELIANCE.NS': 2500.00, 'TCS.NS': 3500.00, 'HDFCBANK.NS': 1600.00,
            'INFY.NS': 1450.00, 'ICICIBANK.NS': 1050.00, 'SBIN.NS': 600.00,
            'BTC-USD': 65000, 'ETH-USD': 3500, 'SOL-USD': 150.00,
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
        
        # Try Alpha Vantage first if available
        av_prices = {}
        if self.alpha_vantage.has_api_key:
            self.current_source = "Alpha Vantage"
            self.fallback_used = False
            av_prices = self.alpha_vantage.get_batch_quotes(stale_tickers)
        
        # Track which tickers need fallback
        fallback_tickers = []
        
        for ticker in stale_tickers:
            if ticker in av_prices and av_prices[ticker] > 0:
                prices[ticker] = av_prices[ticker]
                self.price_cache[ticker] = av_prices[ticker]
                self.cache_timestamp[ticker] = now
            else:
                fallback_tickers.append(ticker)
        
        # Try Yahoo Finance for remaining tickers
        if fallback_tickers:
            yf_prices = self.yahoo_finance.get_batch_quotes(fallback_tickers)
            
            for ticker in fallback_tickers:
                if ticker in yf_prices and yf_prices[ticker] > 0:
                    prices[ticker] = yf_prices[ticker]
                    self.price_cache[ticker] = yf_prices[ticker]
                    self.cache_timestamp[ticker] = now
                    self.fallback_used = True
                    self.current_source = "Yahoo Finance (Fallback)"
                else:
                    # Use default as ultimate fallback
                    prices[ticker] = self.default_prices.get(ticker, 100)
                    self.price_cache[ticker] = prices[ticker]
                    self.cache_timestamp[ticker] = now
        
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
            data = None
            
            # Try Alpha Vantage first
            if self.alpha_vantage.has_api_key:
                av_data = self.alpha_vantage.get_historical(ticker, "compact")
                if av_data is not None and not av_data.empty and len(av_data) > 30:
                    # Filter by period
                    if period == "6mo":
                        av_data = av_data.last(126)
                    elif period == "1y":
                        av_data = av_data.last(252)
                    elif period == "2y":
                        av_data = av_data.last(504)
                    elif period == "5y":
                        av_data = av_data.last(1260)
                    
                    if not av_data.empty:
                        data = av_data[['Close']].rename(columns={'Close': ticker})
                        logger.info(f"✅ Alpha Vantage: {ticker} ({len(data)} days)")
            
            # Fallback to Yahoo if Alpha Vantage failed
            if data is None:
                yf_data = self.yahoo_finance.get_historical(ticker, period)
                if yf_data is not None and not yf_data.empty:
                    if 'Close' in yf_data.columns:
                        data = yf_data[['Close']].rename(columns={'Close': ticker})
                        self.fallback_used = True
                        logger.info(f"✅ Yahoo Fallback: {ticker} ({len(data)} days)")
            
            # Generate synthetic data as last resort
            if data is None:
                logger.warning(f"⚠️ Generating synthetic data for {ticker}")
                data = self._generate_single_synthetic(ticker, period)
            
            if data is not None:
                all_data.append(data)
            
            # Add delay between tickers
            time.sleep(1)
        
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

    def _generate_single_synthetic(self, ticker: str, period: str) -> pd.DataFrame:
        """Generate synthetic data for a single ticker"""
        days = self._period_to_days(period)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
        base_price = self.default_prices.get(ticker, 100)
        returns = np.random.randn(days) * 0.02 + 0.0003
        prices = base_price * np.exp(np.cumsum(returns))
        prices = np.maximum(prices, base_price * 0.3)
        return pd.DataFrame({ticker: prices}, index=dates)

    def _generate_synthetic_data(self, tickers: List[str], period: str) -> pd.DataFrame:
        """Generate synthetic data as ultimate fallback"""
        logger.info(f"🔄 Generating synthetic data for {tickers}")
        self.current_source = "Synthetic (Fallback)"
        
        all_data = []
        for ticker in tickers:
            data = self._generate_single_synthetic(ticker, period)
            all_data.append(data)
        
        return pd.concat(all_data, axis=1)

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
        """Get option chain (Yahoo Finance only)"""
        try:
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

    def get_data_source_info(self) -> str:
        """Get info about current data source"""
        return self.current_source