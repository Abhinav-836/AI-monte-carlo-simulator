"""
Professional Data Fetcher with Alpha Vantage Primary + Yahoo Finance Fallback
Optimized for Streamlit Cloud deployment
"""

import yfinance as yf
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
    Data Fetcher with multiple data sources and fallbacks
    """
    
    def __init__(self, use_live_simulation: bool = False):
        # Get API keys from Streamlit secrets or environment
        self.alpha_vantage_key = self._get_api_key()
        self.use_live_simulation = use_live_simulation
        
        # Track source
        self.current_source = "Synthetic (Initial)"
        self.fallback_used = False
        
        # Live simulation support
        self.live_simulator = None
        self._price_subscribers = []
        self.is_running = False
        
        # Cache
        self.price_cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 60
        
        self.historical_cache = {}
        self.historical_timestamp = {}
        self.historical_cache_duration = 1800
        
        # Default prices
        self.default_prices = self._load_default_prices()
        
        logger.info(f"✅ DataFetcher initialized (Alpha Vantage: {'✓' if self.alpha_vantage_key else '✗'})")
    
    def _get_api_key(self):
        """Get Alpha Vantage API key from various sources"""
        # Try Streamlit secrets first
        try:
            if hasattr(st, 'secrets') and 'ALPHA_VANTAGE_API_KEY' in st.secrets:
                key = st.secrets['ALPHA_VANTAGE_API_KEY']
                if key and key != "your_alpha_vantage_api_key_here":
                    return key
        except:
            pass
        
        # Try environment variables
        key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if key and key != "your_alpha_vantage_api_key_here":
            return key
        
        # Try .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            if key and key != "your_alpha_vantage_api_key_here":
                return key
        except:
            pass
        
        return None
    
    def _load_default_prices(self) -> Dict[str, float]:
        """Load default prices"""
        return {
            'AAPL': 175.32, 'MSFT': 380.45, 'GOOGL': 142.18,
            'AMZN': 178.50, 'META': 472.30, 'TSLA': 198.75,
            'NVDA': 820.15, 'JPM': 152.80, 'V': 255.60,
            'WMT': 60.25, 'JNJ': 155.30, 'PG': 160.45,
            'RELIANCE.NS': 2500.00, 'TCS.NS': 3500.00, 
            'HDFCBANK.NS': 1600.00, 'INFY.NS': 1450.00,
            'BTC-USD': 65000, 'ETH-USD': 3500, 'SOL-USD': 150.00,
        }
    
    def _call_alpha_vantage(self, symbol: str, function: str = "GLOBAL_QUOTE") -> Optional[Dict]:
        """Call Alpha Vantage API"""
        if not self.alpha_vantage_key:
            return None
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.alpha_vantage_key
            }
            
            if function == "TIME_SERIES_DAILY":
                params["outputsize"] = "compact"
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for API errors
                if "Note" in data:
                    logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                    return None
                if "Error Message" in data:
                    logger.warning(f"Alpha Vantage error: {data['Error Message']}")
                    return None
                
                return data
            
        except Exception as e:
            logger.warning(f"Alpha Vantage API error: {e}")
        
        return None
    
    def get_current_prices(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """Get current prices with multiple fallbacks"""
        
        now = time.time()
        prices = {}
        
        # Check cache
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
        if self.alpha_vantage_key:
            for ticker in stale_tickers:
                data = self._call_alpha_vantage(ticker, "GLOBAL_QUOTE")
                if data and "Global Quote" in data:
                    quote = data["Global Quote"]
                    price = quote.get("05. price")
                    if price and float(price) > 0:
                        prices[ticker] = float(price)
                        self.price_cache[ticker] = float(price)
                        self.cache_timestamp[ticker] = now
                        self.current_source = "Alpha Vantage"
                        continue
                
                # If Alpha Vantage fails, try Yahoo
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
                    if price and price > 0:
                        prices[ticker] = float(price)
                        self.price_cache[ticker] = float(price)
                        self.cache_timestamp[ticker] = now
                        self.current_source = "Yahoo Finance"
                        continue
                except:
                    pass
                
                # Ultimate fallback: default price
                prices[ticker] = self.default_prices.get(ticker, 100)
                self.price_cache[ticker] = prices[ticker]
                self.cache_timestamp[ticker] = now
                self.current_source = "Default (Fallback)"
        
        return prices
    
    def get_historical_data(self, tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data with multiple fallbacks"""
        
        cache_key = f"{tickers_tuple}_{period}"
        now = time.time()
        
        # Check cache
        if cache_key in self.historical_timestamp:
            if now - self.historical_timestamp[cache_key] < self.historical_cache_duration:
                logger.info(f"📦 Using cached historical data")
                return self.historical_cache[cache_key]
        
        tickers = list(tickers_tuple)
        logger.info(f"📥 Fetching historical data for {len(tickers)} stocks")
        
        all_data = []
        days = self._period_to_days(period)
        
        for ticker in tickers:
            data = None
            
            # Try Alpha Vantage
            if self.alpha_vantage_key:
                result = self._call_alpha_vantage(ticker, "TIME_SERIES_DAILY")
                if result and "Time Series (Daily)" in result:
                    ts = result["Time Series (Daily)"]
                    df = pd.DataFrame.from_dict(ts, orient="index")
                    df.index = pd.to_datetime(df.index)
                    df = df.astype(float)
                    df = df.sort_index()
                    
                    # Filter by period
                    if len(df) > days:
                        df = df.last(days)
                    
                    if not df.empty:
                        data = df[['4. close']].rename(columns={'4. close': ticker})
                        logger.info(f"✅ Alpha Vantage: {ticker} ({len(data)} days)")
            
            # Try Yahoo if Alpha Vantage failed
            if data is None:
                try:
                    yf_data = yf.download(
                        ticker,
                        period=period,
                        interval="1d",
                        progress=False,
                        auto_adjust=True,
                        threads=False,
                        timeout=10
                    )
                    if not yf_data.empty and 'Close' in yf_data.columns:
                        data = yf_data[['Close']].rename(columns={'Close': ticker})
                        logger.info(f"✅ Yahoo Finance: {ticker} ({len(data)} days)")
                except Exception as e:
                    logger.warning(f"Yahoo failed for {ticker}: {e}")
            
            # Generate synthetic as last resort
            if data is None:
                logger.warning(f"⚠️ Generating synthetic data for {ticker}")
                base_price = self.default_prices.get(ticker, 100)
                returns = np.random.randn(days) * 0.02 + 0.0003
                prices = base_price * np.exp(np.cumsum(returns))
                prices = np.maximum(prices, base_price * 0.3)
                dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
                data = pd.DataFrame({ticker: prices}, index=dates)
                self.current_source = "Synthetic (Fallback)"
            
            if data is not None:
                all_data.append(data)
            
            # Small delay between requests
            time.sleep(0.5)
        
        if not all_data:
            logger.warning("⚠️ No data available")
            return None
        
        # Combine data
        try:
            combined = pd.concat(all_data, axis=1)
            combined = combined.ffill().bfill()
            
            # Cache
            self.historical_cache[cache_key] = combined
            self.historical_timestamp[cache_key] = now
            
            logger.info(f"✅ Successfully fetched {len(combined.columns)} stocks")
            return combined
            
        except Exception as e:
            logger.error(f"Data combination failed: {e}")
            return None
    
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
        """Get option chain (Yahoo Finance)"""
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
            logger.warning(f"Option chain error for {ticker}: {e}")
            return {}
    
    def get_data_source_info(self) -> str:
        """Get current data source"""
        return self.current_source
    
    # ===================== LIVE SIMULATION METHODS =====================
    
    def start_live_simulation(self, tickers: List[str]):
        """Start live price simulation"""
        if self.use_live_simulation:
            logger.info(f"🔄 Starting live simulation for {tickers}")
            # Initialize with default prices
            self.live_prices = {}
            for ticker in tickers:
                self.live_prices[ticker] = self.default_prices.get(ticker, 100)
            self.is_running = True
            self.current_source = "Live Simulation"
    
    def stop_live_simulation(self):
        """Stop live price simulation"""
        self.is_running = False
        logger.info("🔄 Live simulation stopped")
    
    def subscribe_prices(self, callback):
        """Subscribe to price updates"""
        self._price_subscribers.append(callback)
    
    def get_live_prices(self) -> Dict[str, float]:
        """Get current live prices"""
        if hasattr(self, 'live_prices'):
            # Simulate small price changes
            for ticker in self.live_prices:
                change = np.random.uniform(-0.005, 0.005) * self.live_prices[ticker]
                self.live_prices[ticker] += change
                self.live_prices[ticker] = max(self.live_prices[ticker], 0.01)
            return self.live_prices
        return {}
    
    def get_price_history(self, ticker: str) -> List[Dict]:
        """Get price history for a ticker"""
        # Return a simple history for now
        if hasattr(self, 'live_prices') and ticker in self.live_prices:
            return [{
                'time': datetime.now() - timedelta(minutes=i),
                'price': self.live_prices[ticker] * (1 + np.random.uniform(-0.01, 0.01))
            } for i in range(20)]
        return []