"""
Monte Carlo Pipeline with Real-Time Updates & Advanced Features
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import threading
import queue
import time

logger = logging.getLogger(__name__)


class MonteCarloPipeline:
    """
    Advanced Monte Carlo pipeline with real-time capabilities
    """

    def __init__(
        self,
        n_assets: int = 5,
        n_simulations: int = 1000,
        filter_top_k: float = 0.1,
        seq_len: int = 60,
        use_gan: bool = False,
        use_live_data: bool = True
    ):
        self.n_assets = n_assets
        self.n_simulations = n_simulations
        self.filter_top_k = filter_top_k
        self.seq_len = seq_len
        self.use_gan = use_gan
        self.use_live_data = use_live_data
        
        # Initialize components
        from src.pipeline.data_fetcher import DataFetcher
        from src.pipeline.risk.metrics import RiskCalculator
        from src.pipeline.risk.confidence import ConfidenceIntervalCalculator
        
        self.data_fetcher = DataFetcher(use_live_simulation=use_live_data)
        self.risk_calc = RiskCalculator()
        self.ci_calc = ConfidenceIntervalCalculator()
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Models (lazy loaded)
        self.generator = None
        self.discriminator = None
        self.filter = None
        
        # Real-time components
        self.live_prices = {}
        self.price_history = {}
        self.update_queue = queue.Queue()
        self.is_running = False
        self.update_thread = None
        
        # Performance tracking
        self.simulation_history = []
        self.performance_metrics = {}
        
        logger.info(f"Pipeline initialized with {n_simulations} simulations, use_gan={use_gan}")

    def load_models(self):
        """Load or initialize GAN models"""
        if not self.use_gan:
            logger.info("GAN disabled, using standard Monte Carlo")
            return
            
        try:
            from src.pipeline.dfmgan.generator import (
                DFMGANConfig, ConditionalMCGenerator, 
                TimeGANDiscriminator, PathFilter
            )
            
            self.dfmgan_config = DFMGANConfig(
                n_assets=self.n_assets,
                seq_len=self.seq_len,
                n_simulations=self.n_simulations,
                filter_top_k=self.filter_top_k
            )
            
            self.generator = ConditionalMCGenerator(self.dfmgan_config).to(self.device)
            self.discriminator = TimeGANDiscriminator(self.dfmgan_config).to(self.device)
            self.filter = PathFilter(self.discriminator)
            
            logger.info("GAN models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load GAN models: {e}")
            self.use_gan = False

    def start_live_updates(self, tickers: List[str]):
        """Start live price updates"""
        if self.use_live_data:
            self.data_fetcher.start_live_simulation(tickers)
            self.data_fetcher.subscribe_prices(self._on_price_update)
            self.is_running = True
            
            # Start update processing thread
            if self.update_thread is None or not self.update_thread.is_alive():
                self.update_thread = threading.Thread(target=self._process_updates, daemon=True)
                self.update_thread.start()
            
            logger.info(f"📡 Live updates started for {tickers}")

    def stop_live_updates(self):
        """Stop live price updates"""
        self.is_running = False
        self.data_fetcher.stop_live_simulation()
        logger.info("📡 Live updates stopped")

    def _on_price_update(self, prices: Dict[str, float]):
        """Callback for price updates"""
        self.update_queue.put(prices)

    def _process_updates(self):
        """Process price updates in background"""
        while self.is_running:
            try:
                prices = self.update_queue.get(timeout=0.1)
                self.live_prices = prices
                
                # Update history
                for ticker, price in prices.items():
                    if ticker not in self.price_history:
                        self.price_history[ticker] = []
                    self.price_history[ticker].append({
                        'time': datetime.now(),
                        'price': price
                    })
                    # Keep last 500 points
                    if len(self.price_history[ticker]) > 500:
                        self.price_history[ticker] = self.price_history[ticker][-500:]
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Update processing error: {e}")

    def get_live_prices(self) -> Dict[str, float]:
        """Get current live prices"""
        return self.live_prices.copy() if self.live_prices else {}

    def get_price_history(self, ticker: str) -> List[Dict]:
        """Get price history for a ticker"""
        return self.price_history.get(ticker, [])

    def _estimate_covariance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """Robust covariance estimation with Ledoit-Wolf shrinkage"""
        try:
            from sklearn.covariance import LedoitWolf
            cov_estimator = LedoitWolf()
            cov = cov_estimator.fit(returns.values).covariance_
            return cov
        except:
            # Fallback to standard with shrinkage
            cov = returns.cov().values
            shrinkage = 0.2
            target = np.diag(np.diag(cov))
            cov = (1 - shrinkage) * cov + shrinkage * target
            
            # Ensure positive definite
            min_eig = np.linalg.eigvals(cov).min()
            if min_eig < 0:
                cov += np.eye(cov.shape[0]) * (-min_eig + 1e-6)
            return cov

    def _simulate_gbm_paths(
        self, 
        returns: pd.DataFrame, 
        last_prices: np.ndarray,
        n_steps: int = 60
    ) -> np.ndarray:
        """Enhanced GBM with jump diffusion"""
        n_paths = self.n_simulations
        n_assets = len(returns.columns)
        
        # Parameters
        mu = returns.mean().values
        cov = self._estimate_covariance_matrix(returns)
        
        # Jump parameters
        jump_prob = 0.05
        jump_mean = -0.02
        jump_std = 0.03
        
        # Cholesky
        L = np.linalg.cholesky(cov)
        dt = 1.0
        
        # Initialize
        paths = np.zeros((n_paths, n_steps, n_assets))
        paths[:, 0, :] = last_prices
        
        for t in range(1, n_steps):
            # Standard GBM
            Z = np.random.randn(n_paths, n_assets)
            correlated_Z = Z @ L.T
            drift = (mu - 0.5 * np.diag(cov)) * dt
            diffusion = np.sqrt(dt) * correlated_Z
            
            # Jump diffusion
            jumps = np.random.binomial(1, jump_prob, (n_paths, n_assets))
            jump_returns = jumps * np.random.normal(jump_mean, jump_std, (n_paths, n_assets))
            
            log_returns = drift + diffusion + jump_returns
            paths[:, t, :] = paths[:, t-1, :] * np.exp(log_returns)
        
        return paths

    def _filter_paths_advanced(self, paths: np.ndarray, returns: pd.DataFrame) -> np.ndarray:
        """Advanced path filtering with multiple criteria"""
        n_paths = len(paths)
        k = max(1, int(n_paths * self.filter_top_k))
        
        # Calculate multiple scores
        scores = np.zeros(n_paths)
        
        # 1. Mean reversion score
        final_returns = paths[:, -1, :] / paths[:, 0, :] - 1
        historical_returns = returns.mean().values
        scores += np.abs(final_returns - historical_returns).mean(axis=1)
        
        # 2. Volatility score
        path_vol = np.std(np.diff(np.log(paths + 1e-8), axis=1), axis=1).mean(axis=1)
        hist_vol = returns.std().values.mean()
        scores += np.abs(path_vol - hist_vol)
        
        # 3. Correlation score (paths should maintain historical correlations)
        for i in range(paths.shape[0]):
            path_corr = np.corrcoef(paths[i, -20:, :].T)
            scores[i] += np.abs(np.mean(path_corr - returns.iloc[-20:].corr().values))
        
        # Select best paths
        best_indices = np.argsort(scores)[:k]
        return paths[best_indices]

    def run_simulation(
        self, 
        tickers: List[str], 
        period: str = "2y",
        use_real_options: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with real data
        """
        start_time = datetime.now()
        logger.info(f"🚀 Starting simulation for {tickers}")
        
        try:
            # Start live updates if enabled
            if self.use_live_data and not self.is_running:
                self.start_live_updates(tickers)
            
            # 1. FETCH DATA
            if progress_callback:
                progress_callback(10, "Fetching historical data...")
            
            historical = self.data_fetcher.get_historical_data(tuple(tickers), period)
            
            if historical is None or historical.empty:
                raise ValueError("No historical data available")
            
            # 2. GET CURRENT PRICES
            if progress_callback:
                progress_callback(20, "Fetching current prices...")
            
            current_prices = self.data_fetcher.get_current_prices(tickers)
            
            # 3. PREPARE DATA
            last_prices = historical.iloc[-1].values
            returns = historical.pct_change().dropna()
            
            # 4. SIMULATE PATHS
            if progress_callback:
                progress_callback(30, "Generating price paths...")
            
            if self.use_gan and self.generator is not None:
                # Use GAN
                hist_tensor = torch.tensor(
                    historical.values[-self.seq_len:], 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    paths_tensor = self.generator(hist_tensor, self.n_simulations)
                    paths = paths_tensor.cpu().numpy()
            else:
                # Use GBM
                paths = self._simulate_gbm_paths(returns, last_prices, n_steps=60)
            
            # 5. FILTER PATHS
            if progress_callback:
                progress_callback(50, "Filtering paths...")
            
            filtered_paths = self._filter_paths_advanced(paths, returns)
            
            # 6. COMPUTE EXPECTED PRICES
            if progress_callback:
                progress_callback(60, "Computing expected prices...")
            
            expected_prices = {}
            confidence_intervals = {}
            price_distributions = {}
            
            for i, ticker in enumerate(tickers):
                if i >= filtered_paths.shape[2]:
                    continue
                    
                final_prices = filtered_paths[:, -1, i]
                
                expected_prices[ticker] = float(np.mean(final_prices))
                confidence_intervals[ticker] = [
                    float(np.percentile(final_prices, 2.5)),
                    float(np.percentile(final_prices, 97.5))
                ]
                price_distributions[ticker] = final_prices.tolist()
            
            # 7. CALCULATE RISK METRICS
            if progress_callback:
                progress_callback(70, "Calculating risk metrics...")
            
            paths_returns = np.diff(np.log(filtered_paths + 1e-8), axis=1)
            
            risk_metrics = {}
            for i, ticker in enumerate(tickers[:filtered_paths.shape[2]]):
                ticker_returns = paths_returns[:, :, i].flatten()
                
                var_95 = np.percentile(ticker_returns, 5)
                cvar_95 = np.mean(ticker_returns[ticker_returns <= var_95]) if np.any(ticker_returns <= var_95) else var_95
                
                expected_return = np.mean(ticker_returns) * 252
                volatility = np.std(ticker_returns) * np.sqrt(252)
                sharpe = (expected_return - 0.02) / (volatility + 1e-8)
                
                # Max drawdown
                price_path = filtered_paths[:, :, i]
                peak = np.maximum.accumulate(price_path, axis=1)
                drawdown = (price_path - peak) / (peak + 1e-8)
                max_dd = np.min(drawdown)
                
                # Sortino ratio
                downside_returns = ticker_returns[ticker_returns < 0]
                downside_risk = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
                sortino = (expected_return - 0.02) / (downside_risk + 1e-8)
                
                risk_metrics[ticker] = {
                    'var_95': float(var_95),
                    'cvar_95': float(cvar_95),
                    'expected_return': float(expected_return),
                    'volatility': float(volatility),
                    'sharpe': float(sharpe),
                    'sortino': float(sortino),
                    'max_drawdown': float(max_dd)
                }
            
            # 8. OPTION PRICING
            if progress_callback:
                progress_callback(80, "Fetching options data...")
            
            option_prices = {}
            if use_real_options:
                for ticker in tickers:
                    if '.' not in ticker:
                        chain = self.data_fetcher.get_option_chain(ticker)
                        if chain:
                            option_prices[ticker] = chain
            
            # 9. CALCULATE VARIANCE REDUCTION
            hist_var = np.var(returns.values.flatten())
            sim_var = np.var(np.diff(np.log(paths + 1e-8), axis=1).flatten())
            variance_reduction = max(0, (hist_var - sim_var) / (hist_var + 1e-8))
            
            # 10. PORTFOLIO STATISTICS
            if progress_callback:
                progress_callback(90, "Computing portfolio statistics...")
            
            portfolio_returns = paths_returns.mean(axis=2).flatten()
            portfolio_stats = {
                'mean_return': float(np.mean(portfolio_returns) * 252),
                'volatility': float(np.std(portfolio_returns) * np.sqrt(252)),
                'sharpe': float(np.mean(portfolio_returns) * 252 / (np.std(portfolio_returns) * np.sqrt(252) + 1e-8)),
                'var_95': float(np.percentile(portfolio_returns, 5)),
                'cvar_95': float(np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]))
            }
            
            # 11. COMPILE RESULTS
            computation_time = (datetime.now() - start_time).total_seconds()
            
            results = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "tickers": tickers,
                    "n_simulations": self.n_simulations,
                    "filtered_paths": len(filtered_paths),
                    "computation_time": computation_time,
                    "data_source": self.data_fetcher.get_data_source_info(),
                    "use_gan": self.use_gan,
                    "use_live_data": self.use_live_data
                },
                "current_prices": current_prices,
                "expected_prices": expected_prices,
                "confidence_intervals": confidence_intervals,
                "price_distributions": price_distributions,
                "risk_metrics": risk_metrics,
                "option_prices": option_prices,
                "portfolio_stats": portfolio_stats,
                "variance_reduction": float(variance_reduction),
                "path_sample": filtered_paths[:10].tolist()
            }
            
            # Store in history
            self.simulation_history.append({
                'timestamp': datetime.now(),
                'tickers': tickers,
                'results': results
            })
            
            if progress_callback:
                progress_callback(100, "Complete!")
            
            logger.info(f"✅ Simulation complete in {computation_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics from simulation history"""
        if not self.simulation_history:
            return {}
        
        metrics = {
            'total_runs': len(self.simulation_history),
            'avg_computation_time': np.mean([s['results']['metadata']['computation_time'] for s in self.simulation_history]),
            'avg_variance_reduction': np.mean([s['results']['variance_reduction'] for s in self.simulation_history])
        }
        
        return metrics