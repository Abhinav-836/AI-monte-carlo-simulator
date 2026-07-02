"""
Professional Data Fetcher - Finnhub + Alpha Vantage Only
No Yahoo Finance - removes 429 errors completely
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
    Data Fetcher - Finnhub FIRST, Alpha Vantage SECOND, Default LAST
    No Yahoo Finance - avoids 429 rate limiting errors
    """
    
    def __init__(self, use_live_simulation: bool = False):
        # Get API keys
        self.alpha_vantage_key = self._get_alpha_vantage_key()
        self.finnhub_key = self._get_finnhub_key()
        self.use_live_simulation = use_live_simulation
        
        # Track source
        self.current_source = "Default Prices"
        
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
        
        # Default prices (updated regularly)
        self.default_prices = self._load_default_prices()
        
        # Log status
        print("✅ DataFetcher initialized")
        print(f"   🔑 Finnhub: {'✅' if self.finnhub_key else '❌'}")
        print(f"   🔑 Alpha Vantage: {'✅' if self.alpha_vantage_key else '❌'}")
        print(f"   📊 Default prices loaded: {len(self.default_prices)}")
        print("   ℹ️  Yahoo Finance DISABLED (to avoid 429 errors)")
    
    def _get_alpha_vantage_key(self):
        try:
            if hasattr(st, 'secrets') and 'ALPHA_VANTAGE_API_KEY' in st.secrets:
                return st.secrets['ALPHA_VANTAGE_API_KEY']
        except:
            pass
        return os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    
    def _get_finnhub_key(self):
        try:
            if hasattr(st, 'secrets') and 'FINNHUB_API_KEY' in st.secrets:
                return st.secrets['FINNHUB_API_KEY']
        except:
            pass
        return os.environ.get("FINNHUB_API_KEY", "")
    
    def _load_default_prices(self) -> Dict[str, float]:
        """Load default prices"""
        return {
            # US Stocks
            'AAPL': 294.38, 'MSFT': 384.28, 'GOOGL': 361.21,
            'NVDA': 197.58, 'META': 612.91, 'AMZN': 178.50,
            'TSLA': 198.75, 'JPM': 152.80, 'V': 255.60,
            'WMT': 60.25, 'JNJ': 155.30, 'PG': 160.45,
            'UNH': 490.20, 'HD': 360.15, 'DIS': 110.30,
            'MA': 440.25, 'BAC': 35.80, 'NFLX': 620.45,
            'ADBE': 525.30, 'CRM': 290.15, 'AMD': 180.25,
            'INTC': 42.50, 'CSCO': 52.75, 'PEP': 170.30,
            'COST': 700.25, 'CVX': 155.40, 'WFC': 55.20,
            'QCOM': 165.30, 'TMO': 550.15, 'ABT': 110.25,
            'NKE': 95.30, 'SBUX': 95.25, 'MCD': 280.15,
            
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
    
    def _call_finnhub(self, symbol: str) -> Optional[float]:
        """Call Finnhub API - PRIMARY SOURCE"""
        if not self.finnhub_key:
            print(f"⚠️ No Finnhub API key for {symbol}")
            return None
        
        try:
            url = "https://finnhub.io/api/v1/quote"
            params = {"symbol": symbol, "token": self.finnhub_key}
            response = requests.get(url, params=params, timeout=10)
            
            print(f"📡 Finnhub API call for {symbol} - Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                price = data.get('c', 0)  # Current price
                if price and price > 0:
                    print(f"✅ Finnhub: {symbol} = ${price}")
                    self.current_source = "Finnhub"
                    return float(price)
                else:
                    print(f"⚠️ Finnhub: No price for {symbol} (response: {data})")
            else:
                print(f"⚠️ Finnhub: HTTP {response.status_code} for {symbol}")
                
        except Exception as e:
            print(f"⚠️ Finnhub error for {symbol}: {e}")
        
        return None
    
    def _call_alpha_vantage(self, symbol: str) -> Optional[float]:
        """Call Alpha Vantage API - SECONDARY SOURCE"""
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
                    quote = data["Global Quote"]
                    price = quote.get("05. price")
                    if price:
                        print(f"✅ Alpha Vantage: {symbol} = ${price}")
                        self.current_source = "Alpha Vantage"
                        return float(price)
                elif "Note" in data:
                    print(f"⏳ Alpha Vantage rate limit: {symbol}")
                else:
                    print(f"⚠️ Alpha Vantage: No data for {symbol}")
        except Exception as e:
            print(f"⚠️ Alpha Vantage error for {symbol}: {e}")
        
        return None
    
    def get_current_prices(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, float]:
        """Get current prices - Finnhub FIRST, then Alpha Vantage, then Default"""
        
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
        
        print(f"📡 Fetching fresh prices for: {stale_tickers}")
        
        for ticker in stale_tickers:
            price = None
            
            # TRY FINNHUB FIRST (Primary)
            print(f"📡 Trying Finnhub for {ticker}...")
            price = self._call_finnhub(ticker)
            
            # If Finnhub fails, try Alpha Vantage
            if price is None:
                print(f"📡 Trying Alpha Vantage for {ticker}...")
                price = self._call_alpha_vantage(ticker)
            
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
            time.sleep(0.3)
        
        return prices
    
    def get_historical_data(self, tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data - uses synthetic data with realistic patterns"""
        
        cache_key = f"{tickers_tuple}_{period}"
        now = time.time()
        
        # Check cache
        if cache_key in self.historical_timestamp:
            if now - self.historical_timestamp[cache_key] < self.historical_cache_duration:
                print(f"📦 Using cached historical data")
                return self.historical_cache[cache_key]
        
        tickers = list(tickers_tuple)
        print(f"📥 Generating historical data for {len(tickers)} stocks")
        
        all_data = []
        days = self._period_to_days(period)
        
        for ticker in tickers:
            base_price = self.default_prices.get(ticker, 100)
            
            # Generate realistic price path with drift and volatility
            mu = 0.0003  # daily drift (about 7.5% annual)
            sigma = 0.02  # daily volatility
            
            # Generate returns
            returns = np.random.randn(days) * sigma + mu
            
            # Add some mean reversion
            for i in range(1, len(returns)):
                returns[i] = returns[i] - 0.001 * (np.sum(returns[:i]) / i)
            
            # Generate prices
            prices = base_price * np.exp(np.cumsum(returns))
            prices = np.maximum(prices, base_price * 0.3)
            dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
            data = pd.DataFrame({ticker: prices}, index=dates)
            all_data.append(data)
            
            print(f"📊 Generated data for {ticker} ({len(data)} days, base: ${base_price})")
        
        # Combine all data
        try:
            combined = pd.concat(all_data, axis=1)
            combined = combined.ffill().bfill()
            
            # Cache
            self.historical_cache[cache_key] = combined
            self.historical_timestamp[cache_key] = now
            
            print(f"✅ Successfully generated data for {len(combined.columns)} stocks")
            return combined
            
        except Exception as e:
            print(f"❌ Data generation failed: {e}")
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
        """Get option chain - returns empty (no Yahoo Finance)"""
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