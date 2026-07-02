"""
Professional Data Fetcher with Multiple Data Sources
Works reliably on Streamlit Cloud
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
        # Get API keys
        self.alpha_vantage_key = self._get_alpha_vantage_key()
        self.finnhub_key = self._get_finnhub_key()
        self.twelve_data_key = self._get_twelve_data_key()
        self.use_live_simulation = use_live_simulation
        
        # Track source
        self.current_source = "Synthetic (Initial)"
        
        # Live simulation support
        self.live_simulator = None
        self._price_subscribers = []
        self.is_running = False
        self.live_prices = {}
        
        # Cache
        self.price_cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 60  # 1 minute
        
        self.historical_cache = {}
        self.historical_timestamp = {}
        self.historical_cache_duration = 3600
        
        # Default prices
        self.default_prices = self._load_default_prices()
        
        # Rate limiting
        self.last_call = {}
        self.call_interval = 1
        
        # Log status
        print("✅ DataFetcher initialized")
        print(f"   Alpha Vantage: {'✅' if self.alpha_vantage_key else '❌'}")
        print(f"   Finnhub: {'✅' if self.finnhub_key else '❌'}")
        print(f"   Twelve Data: {'✅' if self.twelve_data_key else '❌'}")
    
    def _get_alpha_vantage_key(self):
        """Get Alpha Vantage API key"""
        try:
            if hasattr(st, 'secrets') and 'ALPHA_VANTAGE_API_KEY' in st.secrets:
                return st.secrets['ALPHA_VANTAGE_API_KEY']
        except:
            pass
        return os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    
    def _get_finnhub_key(self):
        """Get Finnhub API key"""
        try:
            if hasattr(st, 'secrets') and 'FINNHUB_API_KEY' in st.secrets:
                return st.secrets['FINNHUB_API_KEY']
        except:
            pass
        return os.environ.get("FINNHUB_API_KEY", "")
    
    def _get_twelve_data_key(self):
        """Get Twelve Data API key"""
        try:
            if hasattr(st, 'secrets') and 'TWELVEDATA_API_KEY' in st.secrets:
                return st.secrets['TWELVEDATA_API_KEY']
        except:
            pass
        return os.environ.get("TWELVEDATA_API_KEY", "")
    
    def _load_default_prices(self) -> Dict[str, float]:
        """Load default prices"""
        return {
            'AAPL': 294.38, 'MSFT': 384.28, 'GOOGL': 361.21,
            'NVDA': 197.58, 'META': 612.91, 'AMZN': 178.50,
            'TSLA': 198.75, 'JPM': 152.80, 'V': 255.60,
            'RELIANCE.NS': 2500.00, 'TCS.NS': 3500.00,
            'BTC-USD': 65000, 'ETH-USD': 3500,
        }
    
    def _call_finnhub(self, symbol: str) -> Optional[float]:
        """Call Finnhub API for price"""
        if not self.finnhub_key:
            return None
        
        try:
            url = "https://finnhub.io/api/v1/quote"
            params = {"symbol": symbol, "token": self.finnhub_key}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                price = data.get('c', 0)  # Current price
                if price and price > 0:
                    print(f"✅ Finnhub: {symbol} = ${price}")
                    return float(price)
        except Exception as e:
            print(f"⚠️ Finnhub error for {symbol}: {e}")
        
        return None
    
    def _call_twelve_data(self, symbol: str) -> Optional[float]:
        """Call Twelve Data API for price"""
        if not self.twelve_data_key:
            return None
        
        try:
            url = "https://api.twelvedata.com/price"
            params = {"symbol": symbol, "apikey": self.twelve_data_key}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                price = data.get('price')
                if price:
                    print(f"✅ Twelve Data: {symbol} = ${price}")
                    return float(price)
        except Exception as e:
            print(f"⚠️ Twelve Data error for {symbol}: {e}")
        
        return None
    
    def _call_alpha_vantage(self, symbol: str) -> Optional[float]:
        """Call Alpha Vantage API for price"""
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
                if "Global Quote" in data:
                    quote = data["Global Quote"]
                    price = quote.get("05. price")
                    if price:
                        print(f"✅ Alpha Vantage: {symbol} = ${price}")
                        return float(price)
        except Exception as e:
            print(f"⚠️ Alpha Vantage error for {symbol}: {e}")
        
        return None
    
    def _call_yahoo_finance(self, symbol: str) -> Optional[float]:
        """Call Yahoo Finance for price"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if price and price > 0:
                print(f"✅ Yahoo Finance: {symbol} = ${price}")
                return float(price)
        except Exception as e:
            print(f"⚠️ Yahoo Finance error for {symbol}: {e}")
        
        return None
    
    def get_current_prices(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """Get current prices with multiple data sources"""
        
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
        
        print(f"📡 Fetching prices for: {stale_tickers}")
        
        for ticker in stale_tickers:
            price = None
            
            # Try data sources in order (each with fallback)
            sources = [
                ("Finnhub", self._call_finnhub),
                ("Twelve Data", self._call_twelve_data),
                ("Alpha Vantage", self._call_alpha_vantage),
                ("Yahoo Finance", self._call_yahoo_finance)
            ]
            
            for source_name, source_func in sources:
                if price is None:
                    try:
                        price = source_func(ticker)
                        if price and price > 0:
                            self.current_source = source_name
                            break
                    except:
                        continue
            
            # Use default as ultimate fallback
            if price is None or price <= 0:
                price = self.default_prices.get(ticker, 100)
                print(f"📋 Default price for {ticker}: ${price}")
                self.current_source = "Default Prices"
            
            # Store in cache
            prices[ticker] = price
            self.price_cache[ticker] = price
            self.cache_timestamp[ticker] = now
            
            # Small delay between requests
            time.sleep(0.5)
        
        return prices
    
    def get_historical_data(self, tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data with caching and fallbacks"""
        
        cache_key = f"{tickers_tuple}_{period}"
        now = time.time()
        
        # Check cache
        if cache_key in self.historical_timestamp:
            if now - self.historical_timestamp[cache_key] < self.historical_cache_duration:
                print(f"📦 Using cached historical data")
                return self.historical_cache[cache_key]
        
        tickers = list(tickers_tuple)
        print(f"📥 Fetching historical data for {len(tickers)} stocks")
        
        all_data = []
        days = self._period_to_days(period)
        
        for ticker in tickers:
            data = None
            
            # Try Yahoo Finance first
            try:
                yf_data = yf.download(
                    ticker,
                    period=period,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                    timeout=15
                )
                if not yf_data.empty and 'Close' in yf_data.columns:
                    data = yf_data[['Close']].rename(columns={'Close': ticker})
                    print(f"✅ Yahoo Finance: {ticker} ({len(data)} days)")
            except Exception as e:
                print(f"⚠️ Yahoo failed for {ticker}: {e}")
            
            # Generate synthetic as last resort
            if data is None:
                print(f"⚠️ Generating synthetic data for {ticker}")
                base_price = self.default_prices.get(ticker, 100)
                returns = np.random.randn(days) * 0.02 + 0.0003
                prices = base_price * np.exp(np.cumsum(returns))
                prices = np.maximum(prices, base_price * 0.3)
                dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
                data = pd.DataFrame({ticker: prices}, index=dates)
            
            if data is not None:
                all_data.append(data)
            
            time.sleep(0.5)
        
        if not all_data:
            return None
        
        try:
            combined = pd.concat(all_data, axis=1)
            combined = combined.ffill().bfill()
            
            self.historical_cache[cache_key] = combined
            self.historical_timestamp[cache_key] = now
            
            print(f"✅ Successfully fetched {len(combined.columns)} stocks")
            return combined
            
        except Exception as e:
            print(f"❌ Data combination failed: {e}")
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
            print(f"⚠️ Option chain error for {ticker}: {e}")
            return {}
    
    def get_data_source_info(self) -> str:
        """Get current data source"""
        return self.current_source
    
    # ===================== LIVE SIMULATION METHODS =====================
    
    def start_live_simulation(self, tickers: List[str]):
        """Start live price simulation"""
        if self.use_live_simulation:
            print(f"🔄 Starting live simulation for {tickers}")
            self.live_prices = {}
            for ticker in tickers:
                self.live_prices[ticker] = self.default_prices.get(ticker, 100)
            self.is_running = True
            self.current_source = "Live Simulation"
    
    def stop_live_simulation(self):
        """Stop live price simulation"""
        self.is_running = False
        print("🔄 Live simulation stopped")
    
    def subscribe_prices(self, callback):
        """Subscribe to price updates"""
        self._price_subscribers.append(callback)
    
    def get_live_prices(self) -> Dict[str, float]:
        """Get current live prices"""
        if hasattr(self, 'live_prices') and self.is_running:
            for ticker in self.live_prices:
                change = np.random.uniform(-0.005, 0.005) * self.live_prices[ticker]
                self.live_prices[ticker] += change
                self.live_prices[ticker] = max(self.live_prices[ticker], 0.01)
            return self.live_prices
        return {}
    
    def get_price_history(self, ticker: str) -> List[Dict]:
        """Get price history for a ticker"""
        if hasattr(self, 'live_prices') and ticker in self.live_prices:
            return [{
                'time': datetime.now() - timedelta(minutes=i),
                'price': self.live_prices[ticker] * (1 + np.random.uniform(-0.01, 0.01))
            } for i in range(20)]
        return []