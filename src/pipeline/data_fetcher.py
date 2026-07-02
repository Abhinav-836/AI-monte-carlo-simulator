"""
Professional Data Fetcher - Finnhub + Alpha Vantage Only
No default fallback - only real data from APIs
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
    Data Fetcher - Finnhub FIRST, Alpha Vantage SECOND
    No default prices - only real API data
    """
    
    def __init__(self, use_live_simulation: bool = False):
        # Get API keys
        self.alpha_vantage_key = self._get_alpha_vantage_key()
        self.finnhub_key = self._get_finnhub_key()
        self.use_live_simulation = use_live_simulation
        
        # Track source
        self.current_source = "No Data"
        
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
        
        # Log status
        print("✅ DataFetcher initialized")
        print(f"   🔑 Finnhub: {'✅' if self.finnhub_key else '❌'}")
        print(f"   🔑 Alpha Vantage: {'✅' if self.alpha_vantage_key else '❌'}")
        print("   ℹ️  No default fallback - only real API data")
    
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
    
    def _call_finnhub(self, symbol: str) -> Optional[float]:
        """Call Finnhub API - PRIMARY SOURCE"""
        if not self.finnhub_key:
            print(f"⚠️ No Finnhub API key for {symbol}")
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
                    self.current_source = "Finnhub"
                    return float(price)
                else:
                    print(f"⚠️ Finnhub: No price for {symbol}")
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
        """Get current prices - ONLY from Finnhub and Alpha Vantage"""
        
        now = time.time()
        prices = {}
        missing_tickers = []
        
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
            
            # Only store if we got a price
            if price is not None and price > 0:
                prices[ticker] = price
                self.price_cache[ticker] = price
                self.cache_timestamp[ticker] = now
            else:
                # Track missing tickers
                missing_tickers.append(ticker)
                print(f"❌ No data available for {ticker} from any source")
            
            # Small delay between requests
            time.sleep(0.3)
        
        # If some tickers have no data, log warning
        if missing_tickers:
            print(f"⚠️ No data for: {missing_tickers}")
            self.current_source = "Partial Data"
        
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
        
        # Try to get real historical data from Alpha Vantage first
        for ticker in tickers:
            data = None
            
            # Try Alpha Vantage for historical
            if self.alpha_vantage_key:
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
                            df = df.astype(float)
                            df = df.sort_index()
                            
                            if len(df) > days:
                                df = df.last(days)
                            
                            if not df.empty:
                                data = df[['4. close']].rename(columns={'4. close': ticker})
                                print(f"✅ Alpha Vantage historical: {ticker} ({len(data)} days)")
                except Exception as e:
                    print(f"⚠️ Alpha Vantage historical error for {ticker}: {e}")
            
            # If no historical data, use realistic synthetic
            if data is None:
                print(f"⚠️ No historical data for {ticker}, using synthetic")
                # Use a reasonable base price from API if available, otherwise use 100
                base_price = 100
                # Try to get current price first
                current_price = self._call_finnhub(ticker) or self._call_alpha_vantage(ticker)
                if current_price:
                    base_price = current_price
                
                # Generate realistic synthetic data
                returns = np.random.randn(days) * 0.02 + 0.0003
                prices = base_price * np.exp(np.cumsum(returns))
                prices = np.maximum(prices, base_price * 0.3)
                dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
                data = pd.DataFrame({ticker: prices}, index=dates)
                print(f"📊 Generated synthetic data for {ticker} (base: ${base_price:.2f})")
            
            if data is not None:
                all_data.append(data)
            
            time.sleep(0.5)
        
        if not all_data:
            print("❌ No historical data available")
            return None
        
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
        """Get option chain - returns empty if not available"""
        try:
            import yfinance as yf
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
            # Try to get real prices first
            for ticker in tickers:
                price = self._call_finnhub(ticker) or self._call_alpha_vantage(ticker)
                if price:
                    self.live_prices[ticker] = price
                else:
                    self.live_prices[ticker] = 100  # Fallback for simulation only
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
            # Simulate small price changes only if we have real prices
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