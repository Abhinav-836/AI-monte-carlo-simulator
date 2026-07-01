"""
Professional Data Fetcher with WebSocket Simulation & Advanced Caching
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
import asyncio
from collections import deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """Advanced rate limiter with adaptive throttling"""
    
    def __init__(self, max_calls_per_minute=30):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = deque()
        self.lock = threading.Lock()
        self.adaptive_factor = 1.0
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
            
            if len(self.calls) >= self.max_calls_per_minute * self.adaptive_factor:
                oldest = self.calls[0]
                wait_time = 60 - (now - oldest) + random.uniform(0.5, 2)
                logger.info(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.calls.clear()
                # Increase adaptive factor if we hit limit often
                self.adaptive_factor = min(1.5, self.adaptive_factor * 1.1)
            
            self.calls.append(now)
            # Gradually reduce adaptive factor
            self.adaptive_factor = max(1.0, self.adaptive_factor * 0.99)


class LivePriceSimulator:
    """Simulates live price updates with realistic market behavior"""
    
    def __init__(self):
        self.prices = {}
        self.history = {}
        self.volatility = {}
        self.drift = {}
        self.last_update = {}
        self.is_running = False
        self.thread = None
        self.callbacks = []
    
    def start(self, tickers: List[str], initial_prices: Dict[str, float] = None):
        """Start the live price simulator"""
        self.is_running = True
        
        for ticker in tickers:
            base_price = initial_prices.get(ticker, 100) if initial_prices else 100
            self.prices[ticker] = base_price
            self.volatility[ticker] = random.uniform(0.01, 0.03)
            self.drift[ticker] = random.uniform(-0.0002, 0.0004)
            self.history[ticker] = deque(maxlen=100)
            self.last_update[ticker] = datetime.now()
        
        # Start background thread
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the simulator"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _run(self):
        """Background thread for price updates"""
        while self.is_running:
            for ticker in self.prices:
                # Mean reversion
                mean_price = self.prices[ticker]
                noise = np.random.normal(0, self.volatility[ticker] * mean_price)
                drift_term = self.drift[ticker] * mean_price
                reversion = -0.001 * (self.prices[ticker] - mean_price)
                
                # Update price
                new_price = self.prices[ticker] + drift_term + noise + reversion
                new_price = max(new_price, 0.01)  # Ensure positive
                
                self.prices[ticker] = new_price
                self.history[ticker].append({
                    'time': datetime.now(),
                    'price': new_price
                })
                self.last_update[ticker] = datetime.now()
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(self.prices)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            time.sleep(0.5)  # Update every 500ms
    
    def register_callback(self, callback):
        """Register a callback for price updates"""
        self.callbacks.append(callback)
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current simulated prices"""
        return self.prices.copy()
    
    def get_history(self, ticker: str) -> List[Dict]:
        """Get price history for a ticker"""
        return list(self.history.get(ticker, []))


class DataFetcher:
    """
    Ultra-reliable financial data fetcher with live simulation
    """

    def __init__(self, use_live_simulation: bool = False):
        self.rate_limiter = RateLimiter(max_calls_per_minute=30)
        self.source_used = "Yahoo Finance"
        
        # Enhanced caching with TTL
        self.price_cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 60  # 1 minute for live data
        
        self.historical_cache = {}
        self.historical_timestamp = {}
        self.historical_cache_duration = 3600  # 1 hour
        
        # Live simulation
        self.use_live_simulation = use_live_simulation
        self.live_simulator = LivePriceSimulator() if use_live_simulation else None
        
        # Default prices
        self.default_prices = self._load_default_prices()
        
        # Failed requests tracking
        self.failed_requests = {}
        self.failed_cooldown = 300  # 5 minutes
        
        # WebSocket-like event system
        self._price_subscribers = []
        
        logger.info(f"✅ DataFetcher initialized (Live sim: {use_live_simulation})")

    def _load_default_prices(self) -> Dict[str, float]:
        """Load extended default prices"""
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

    def start_live_simulation(self, tickers: List[str]):
        """Start live price simulation for tickers"""
        if self.live_simulator:
            # Get initial real prices if available
            initial_prices = self.get_current_prices(tickers, force_refresh=True)
            self.live_simulator.start(tickers, initial_prices)
            self.source_used = "Live Simulation"
            logger.info(f"🔄 Live simulation started for {len(tickers)} tickers")

    def stop_live_simulation(self):
        """Stop live price simulation"""
        if self.live_simulator:
            self.live_simulator.stop()
            logger.info("🔄 Live simulation stopped")

    def subscribe_prices(self, callback):
        """Subscribe to live price updates"""
        if self.live_simulator:
            self.live_simulator.register_callback(callback)

    def get_live_prices(self) -> Dict[str, float]:
        """Get current live prices"""
        if self.live_simulator:
            return self.live_simulator.get_current_prices()
        return {}

    def get_price_history(self, ticker: str) -> List[Dict]:
        """Get price history from live simulation"""
        if self.live_simulator:
            return self.live_simulator.get_history(ticker)
        return []

    def get_historical_data(self, tickers_tuple: Tuple[str, ...], period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data with enhanced caching"""
        cache_key = f"{tickers_tuple}_{period}"
        now = time.time()
        
        # Check cache
        if cache_key in self.historical_timestamp:
            if now - self.historical_timestamp[cache_key] < self.historical_cache_duration:
                logger.info(f"📦 Using cached historical data")
                return self.historical_cache[cache_key]
        
        tickers = list(tickers_tuple)
        logger.info(f"📥 Fetching fresh historical data for {len(tickers)} stocks")
        
        # Batch fetch with progress
        chunk_size = 5
        all_data = []
        total_chunks = (len(tickers) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            chunk_num = i // chunk_size + 1
            
            logger.info(f"  Fetching chunk {chunk_num}/{total_chunks}: {chunk}")
            
            try:
                self.rate_limiter.wait_if_needed()
                data = yf.download(
                    chunk,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    timeout=30
                )
                
                if not data.empty:
                    # Extract close prices
                    if 'Close' in data.columns:
                        if isinstance(data.columns, pd.MultiIndex):
                            close_data = data['Close']
                        else:
                            close_data = data[['Close']]
                        
                        for col in close_data.columns:
                            all_data.append(close_data[[col]])
                
                time.sleep(1)  # Delay between chunks
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch {chunk}: {e}")
                # Use synthetic data for failed chunks
                synthetic = self._generate_synthetic_data(chunk, period)
                all_data.append(synthetic)
            
            # Progress update
            logger.info(f"  ✓ Chunk {chunk_num}/{total_chunks} complete")
        
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
            self.source_used = "Yahoo Finance"
            
            logger.info(f"✅ Successfully fetched data for {len(combined.columns)} stocks")
            return combined
            
        except Exception as e:
            logger.error(f"❌ Data combination failed: {e}")
            return self._generate_synthetic_data(tickers, period)

    def _generate_synthetic_data(self, tickers: List[str], period: str) -> pd.DataFrame:
        """Generate realistic synthetic data with correlations"""
        logger.info(f"🔄 Generating synthetic data for {tickers}")
        self.source_used = "Synthetic (Fallback)"
        
        days = self._period_to_days(period)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
        
        # Generate correlated returns
        n_assets = len(tickers)
        corr_matrix = np.eye(n_assets) * 0.7 + 0.3
        L = np.linalg.cholesky(corr_matrix)
        
        returns = np.random.randn(days, n_assets) @ L.T * 0.02
        returns = returns + 0.0003  # Drift
        
        data = {}
        for i, ticker in enumerate(tickers):
            base_price = self.default_prices.get(ticker, 100)
            prices = base_price * np.exp(np.cumsum(returns[:, i]))
            prices = np.maximum(prices, base_price * 0.3)
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
        """Get current prices with enhanced caching"""
        # If using live simulation, return live prices
        if self.live_simulator and self.live_simulator.is_running:
            live_prices = self.live_simulator.get_current_prices()
            if live_prices:
                return live_prices
        
        now = time.time()
        prices = {}
        
        # Check cache
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
            
            # Batch fetch
            chunk_size = 5
            for i in range(0, len(stale_tickers), chunk_size):
                chunk = stale_tickers[i:i+chunk_size]
                
                try:
                    self.rate_limiter.wait_if_needed()
                    tickers_str = " ".join(chunk)
                    data = yf.download(
                        tickers_str,
                        period="1d",
                        interval="1m",
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                        timeout=10
                    )
                    
                    if not data.empty and 'Close' in data.columns:
                        if isinstance(data.columns, pd.MultiIndex):
                            close_data = data['Close']
                        else:
                            close_data = data[['Close']]
                        
                        for ticker in chunk:
                            if ticker in close_data.columns:
                                price = close_data[ticker].iloc[-1]
                                if price and price > 0:
                                    prices[ticker] = float(price)
                                    self.price_cache[ticker] = float(price)
                                    self.cache_timestamp[ticker] = now
                                    continue
                    
                    # Fallback for missing tickers
                    for ticker in chunk:
                        if ticker not in prices:
                            prices[ticker] = self.default_prices.get(ticker, 100)
                            self.price_cache[ticker] = prices[ticker]
                            self.cache_timestamp[ticker] = now
                            
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch prices for {chunk}: {e}")
                    for ticker in chunk:
                        if ticker not in prices:
                            prices[ticker] = self.default_prices.get(ticker, 100)
                
                time.sleep(1)  # Delay between chunks
        
        return prices

    def get_option_chain(self, ticker: str) -> Dict:
        """Fetch option chain with better formatting"""
        if '.' in ticker or ticker in self.failed_requests:
            return {}
        
        try:
            self.rate_limiter.wait_if_needed()
            stock = yf.Ticker(ticker)
            expirations = stock.options
            
            if not expirations:
                return {}
            
            # Get nearest 3 expirations
            chain_data = {}
            for exp in expirations[:3]:
                chain = stock.option_chain(exp)
                
                calls = []
                for _, row in chain.calls.head(10).iterrows():
                    calls.append({
                        'strike': float(row.get('strike', 0)),
                        'lastPrice': float(row.get('lastPrice', 0)),
                        'bid': float(row.get('bid', 0)),
                        'ask': float(row.get('ask', 0)),
                        'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                        'openInterest': int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0,
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
                        'openInterest': int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0,
                        'impliedVolatility': float(row.get('impliedVolatility', 0))
                    })
                
                chain_data[exp] = {
                    'calls': calls,
                    'puts': puts,
                    'underlying_price': float(chain.underlying['price']) if 'price' in chain.underlying else 0
                }
            
            return chain_data
            
        except Exception as e:
            logger.warning(f"Option chain error for {ticker}: {e}")
            self.failed_requests[ticker] = time.time() + 300
            return {}

    def get_data_source_info(self) -> str:
        """Get info about data source used"""
        if self.live_simulator and self.live_simulator.is_running:
            return "Live Simulation + Yahoo Finance"
        return self.source_used

    def get_market_summary(self) -> Dict:
        """Get market summary with key indices"""
        indices = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'Dow Jones',
            '^FTSE': 'FTSE 100',
            '^NSEI': 'NIFTY 50',
            '^BSESN': 'SENSEX',
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum'
        }
        
        summary = {}
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                price = info.get('regularMarketPrice') or info.get('currentPrice')
                if price:
                    change = info.get('regularMarketChangePercent', 0)
                    summary[name] = {
                        'symbol': symbol,
                        'price': float(price),
                        'change': float(change),
                        'direction': 'up' if change > 0 else 'down'
                    }
            except:
                pass
        
        return summary