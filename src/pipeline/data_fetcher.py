"""
Professional Data Fetcher - Fixed for Streamlit Cloud
Uses Yahoo Finance with smart caching + Alpha Vantage fallback
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
    Data Fetcher with Yahoo Finance primary + Alpha Vantage fallback
    Optimized for Streamlit Cloud with smart caching
    """
    
    def __init__(self, use_live_simulation: bool = False):
        # Get API keys
        self.alpha_vantage_key = self._get_api_key()
        self.use_live_simulation = use_live_simulation
        
        # Track source
        self.current_source = "Synthetic (Initial)"
        self.fallback_used = False
        
        # Live simulation support
        self.live_simulator = None
        self._price_subscribers = []
        self.is_running = False
        self.live_prices = {}
        
        # Cache - increased duration to reduce API calls
        self.price_cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 300  # 5 minutes
        
        self.historical_cache = {}
        self.historical_timestamp = {}
        self.historical_cache_duration = 3600  # 1 hour
        
        # Default prices
        self.default_prices = self._load_default_prices()
        
        # Rate limiting
        self.last_yahoo_call = 0
        self.yahoo_call_interval = 2  # 2 seconds between calls
        
        # Log API key status
        if self.alpha_vantage_key:
            print(f"✅ Alpha Vantage API key loaded: {self.alpha_vantage_key[:8]}...")
            logger.info(f"✅ Alpha Vantage API key loaded: {self.alpha_vantage_key[:8]}...")
        else:
            print("ℹ️ No Alpha Vantage API key - using Yahoo Finance only")
            logger.info("ℹ️ No Alpha Vantage API key - using Yahoo Finance only")
    
    def _get_api_key(self):
        """Get Alpha Vantage API key from various sources"""
        api_key = None
        
        # 1. Try Streamlit secrets (for cloud deployment)
        try:
            if hasattr(st, 'secrets'):
                if 'ALPHA_VANTAGE_API_KEY' in st.secrets:
                    key = st.secrets['ALPHA_VANTAGE_API_KEY']
                    if isinstance(key, str):
                        key = key.strip().strip('"').strip("'")
                        if key and len(key) > 10:
                            api_key = key
                            print(f"✅ API key found in Streamlit secrets: {key[:8]}...")
                            return api_key
        except Exception as e:
            print(f"⚠️ Error reading Streamlit secrets: {e}")
        
        # 2. Try environment variables
        key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if key:
            key = key.strip().strip('"').strip("'")
            if key and len(key) > 10:
                api_key = key
                print(f"✅ API key found in environment: {key[:8]}...")
                return api_key
        
        # 3. Try .env file (local development)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            if key:
                key = key.strip().strip('"').strip("'")
                if key and len(key) > 10:
                    api_key = key
                    print(f"✅ API key found in .env: {key[:8]}...")
                    return api_key
        except Exception as e:
            pass
        
        print("ℹ️ No Alpha Vantage API key found")
        return None
    
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
            
            # Indian Stocks (NSE)
            'RELIANCE.NS': 2500.00, 'TCS.NS': 3500.00, 'HDFCBANK.NS': 1600.00,
            'INFY.NS': 1450.00, 'ICICIBANK.NS': 1050.00, 'SBIN.NS': 600.00,
            'BHARTIARTL.NS': 850.00, 'ITC.NS': 400.00, 'WIPRO.NS': 500.00,
            'HINDUNILVR.NS': 2500.00, 'TITAN.NS': 2800.00, 'BAJFINANCE.NS': 7000.00,
            'MARUTI.NS': 9500.00, 'SUNPHARMA.NS': 1200.00, 'ONGC.NS': 150.00,
            'NTPC.NS': 200.00, 'POWERGRID.NS': 220.00, 'ULTRACEMCO.NS': 8000.00,
            'HCLTECH.NS': 1200.00, 'TECHM.NS': 1100.00, 'ASIANPAINT.NS': 3200.00,
            'AXISBANK.NS': 1000.00, 'KOTAKBANK.NS': 1800.00, 'INDUSINDBK.NS': 1400.00,
            
            # UK Stocks
            'BP.L': 450.00, 'HSBA.L': 650.00, 'GSK.L': 1400.00,
            'AZN.L': 10500.00, 'DGE.L': 2800.00, 'LLOY.L': 45.00,
            'BARC.L': 150.00, 'VOD.L': 130.00, 'RIO.L': 5000.00,
            'GLEN.L': 400.00, 'AAL.L': 2200.00, 'PRU.L': 800.00,
            
            # Crypto
            'BTC-USD': 65000, 'ETH-USD': 3500, 'BNB-USD': 450,
            'XRP-USD': 0.75, 'ADA-USD': 0.45, 'DOGE-USD': 0.12,
            'SOL-USD': 150.00, 'DOT-USD': 8.00, 'MATIC-USD': 0.80,
        }
    
    def _call_alpha_vantage(self, symbol: str) -> Optional[Dict]:
        """Call Alpha Vantage API with multiple function attempts"""
        if not self.alpha_vantage_key:
            return None
        
        try:
            url = "https://www.alphavantage.co/query"
            
            # Try multiple functions
            functions = [
                ("GLOBAL_QUOTE", {"function": "GLOBAL_QUOTE", "symbol": symbol}),
                ("SYMBOL_SEARCH", {"function": "SYMBOL_SEARCH", "keywords": symbol})
            ]
            
            for func_name, params in functions:
                params["apikey"] = self.alpha_vantage_key
                
                try:
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Check for API errors
                        if "Note" in data:
                            print(f"⏳ Alpha Vantage rate limit: {data['Note'][:50]}...")
                            continue
                        if "Error Message" in data:
                            print(f"⚠️ Alpha Vantage error: {data['Error Message']}")
                            continue
                        
                        # Check for valid data
                        if "Global Quote" in data and data["Global Quote"]:
                            return data
                        
                        if "bestMatches" in data and data["bestMatches"]:
                            match = data["bestMatches"][0]
                            matched_symbol = match.get("1. symbol", "")
                            if matched_symbol:
                                # Retry with matched symbol
                                return self._call_alpha_vantage(matched_symbol)
                        
                        if "Time Series (Daily)" in data:
                            return data
                except Exception as e:
                    print(f"⚠️ Alpha Vantage {func_name} error: {e}")
                    continue
                
                # Small delay between attempts
                time.sleep(0.5)
            
            return None
            
        except Exception as e:
            print(f"❌ Alpha Vantage API error: {e}")
            return None
    
    def _call_yahoo_finance(self, ticker: str) -> Optional[float]:
        """Call Yahoo Finance with rate limiting"""
        try:
            # Rate limiting
            now = time.time()
            elapsed = now - self.last_yahoo_call
            if elapsed < self.yahoo_call_interval:
                time.sleep(self.yahoo_call_interval - elapsed)
            self.last_yahoo_call = time.time()
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try multiple price sources
            price = (
                info.get('currentPrice') or
                info.get('regularMarketPrice') or
                info.get('ask') or
                info.get('bid') or
                info.get('previousClose')
            )
            
            if price and price > 0:
                return float(price)
            
        except Exception as e:
            print(f"⚠️ Yahoo Finance error for {ticker}: {e}")
        
        return None
    
    def get_current_prices(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """Get current prices with smart caching and multiple fallbacks"""
        
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
        
        print(f"📡 Fetching fresh prices for: {stale_tickers}")
        
        for ticker in stale_tickers:
            price = None
            
            # Try Alpha Vantage first (if available)
            if self.alpha_vantage_key:
                print(f"📡 Trying Alpha Vantage for {ticker}...")
                data = self._call_alpha_vantage(ticker)
                if data and "Global Quote" in data:
                    quote = data["Global Quote"]
                    price_str = quote.get("05. price")
                    if price_str:
                        try:
                            price = float(price_str)
                            if price > 0:
                                print(f"✅ Alpha Vantage: {ticker} = ${price}")
                                self.current_source = "Alpha Vantage"
                        except:
                            pass
            
            # Try Yahoo Finance if Alpha Vantage failed
            if price is None:
                print(f"📡 Trying Yahoo Finance for {ticker}...")
                price = self._call_yahoo_finance(ticker)
                if price and price > 0:
                    print(f"✅ Yahoo Finance: {ticker} = ${price}")
                    self.current_source = "Yahoo Finance"
            
            # Use default as ultimate fallback
            if price is None or price <= 0:
                price = self.default_prices.get(ticker, 100)
                print(f"📋 Default price for {ticker}: ${price}")
                self.current_source = "Default (Fallback)"
            
            # Store in cache
            prices[ticker] = price
            self.price_cache[ticker] = price
            self.cache_timestamp[ticker] = now
        
        return prices
    
    def get_historical_data(self, tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data with caching"""
        
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
        
        for ticker in tickers:
            data = None
            
            # Try Yahoo Finance first
            try:
                print(f"📡 Fetching {ticker} from Yahoo Finance...")
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
                    self.current_source = "Yahoo Finance"
            except Exception as e:
                print(f"⚠️ Yahoo failed for {ticker}: {e}")
            
            # Generate synthetic as last resort
            if data is None:
                print(f"⚠️ Generating synthetic data for {ticker}")
                days = self._period_to_days(period)
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
            time.sleep(1)
        
        if not all_data:
            print("⚠️ No data available")
            return None
        
        # Combine data
        try:
            combined = pd.concat(all_data, axis=1)
            combined = combined.ffill().bfill()
            
            # Cache
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