"""
Monte Carlo Pipeline with Real Data Integration
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MonteCarloPipeline:
    """
    Production Monte Carlo pipeline with real data
    """

    def __init__(
        self,
        n_assets: int = 5,
        n_simulations: int = 1000,
        filter_top_k: float = 0.1,
        seq_len: int = 60,
        use_gan: bool = False
    ):
        self.n_assets = n_assets
        self.n_simulations = n_simulations
        self.filter_top_k = filter_top_k
        self.seq_len = seq_len
        self.use_gan = use_gan
        
        # Initialize components
        from src.pipeline.data_fetcher import DataFetcher
        from src.pipeline.risk.metrics import RiskCalculator
        from src.pipeline.risk.confidence import ConfidenceIntervalCalculator
        
        self.data_fetcher = DataFetcher()
        self.risk_calc = RiskCalculator()
        self.ci_calc = ConfidenceIntervalCalculator()
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Models (lazy loaded)
        self.generator = None
        self.discriminator = None
        self.filter = None
        
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

    def _estimate_covariance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Robust covariance estimation
        """
        # Standard covariance
        cov = returns.cov().values
        
        # Shrinkage for stability
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
        """
        Geometric Brownian Motion simulation
        """
        n_paths = self.n_simulations
        n_assets = len(returns.columns)
        
        # Parameters
        mu = returns.mean().values
        cov = self._estimate_covariance_matrix(returns)
        
        # Cholesky decomposition
        L = np.linalg.cholesky(cov)
        
        # Time step (daily)
        dt = 1.0
        
        # Initialize price paths
        paths = np.zeros((n_paths, n_steps, n_assets))
        paths[:, 0, :] = last_prices
        
        for t in range(1, n_steps):
            # Random shocks
            Z = np.random.randn(n_paths, n_assets)*2
            
            # Correlated returns
            correlated_Z = Z @ L.T
            
            # Price update (GBM)
            drift = (mu - 0.5 * np.diag(cov)) * dt
            diffusion = np.sqrt(dt) * correlated_Z
            
            log_returns = drift + diffusion
            paths[:, t, :] = paths[:, t-1, :] * np.exp(log_returns)
        
        return paths

    def _filter_paths(self, paths: np.ndarray, returns: pd.DataFrame) -> np.ndarray:
        """
        Filter paths based on statistical plausibility
        """
        n_paths = len(paths)
        k = max(1, int(n_paths * self.filter_top_k))
        
        # Calculate path statistics
        final_returns = paths[:, -1, :] / paths[:, 0, :] - 1
        
        # Mean reversion score (paths should not diverge too much)
        mean_return = returns.mean().values
        return_deviation = np.abs(final_returns - mean_return).mean(axis=1)
        
        # Volatility score (paths should have realistic volatility)
        path_vol = np.std(np.diff(np.log(paths), axis=1), axis=1).mean(axis=1)
        hist_vol = returns.std().values.mean()
        vol_deviation = np.abs(path_vol - hist_vol)
        
        # Combined score (lower is better)
        scores = return_deviation + vol_deviation
        
        # Select best paths
        best_indices = np.argsort(scores)[:k]
        
        return paths[best_indices]

    def run_simulation(
        self, 
        tickers: List[str], 
        period: str = "2y",
        use_real_options: bool = False
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with real data
        """
        start_time = datetime.now()
        logger.info(f"🚀 Starting simulation for {tickers}")
        
        try:
            # 1. FETCH REAL DATA
            logger.info("📥 Fetching historical data...")
            historical = self.data_fetcher.get_historical_data(tuple(tickers), period)
            
            if historical is None or historical.empty:
                raise ValueError("No historical data available")
            
            logger.info(f"   ✓ Retrieved {len(historical)} days of data")
            
            # 2. GET CURRENT PRICES
            logger.info("💰 Fetching current prices...")
            current_prices = self.data_fetcher.get_current_prices(tickers)
            logger.info(f"   ✓ Current prices: {current_prices}")
            
            # 3. PREPARE DATA
            last_prices = historical.iloc[-1].values
            returns = historical.pct_change().dropna()
            
            # 4. SIMULATE PATHS
            logger.info(f"📊 Generating {self.n_simulations:,} price paths...")
            
            if self.use_gan and self.generator is not None:
                # Use GAN for path generation
                hist_tensor = torch.tensor(
                    historical.values[-self.seq_len:], 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    paths_tensor = self.generator(hist_tensor, self.n_simulations)
                    paths = paths_tensor.cpu().numpy()
            else:
                # Use standard GBM
                paths = self._simulate_gbm_paths(returns, last_prices, n_steps=60)
            
            logger.info(f"   ✓ Generated paths shape: {paths.shape}")
            
            # 5. FILTER PATHS
            logger.info(f"🎯 Filtering top {self.filter_top_k*100:.0f}% paths...")
            filtered_paths = self._filter_paths(paths, returns)
            logger.info(f"   ✓ Kept {len(filtered_paths)} most realistic paths")
            
            # 6. COMPUTE EXPECTED PRICES
            expected_prices = {}
            confidence_intervals = {}
            
            for i, ticker in enumerate(tickers):
                if i >= filtered_paths.shape[2]:
                    continue
                    
                final_prices = filtered_paths[:, -1, i]
                
                # Expected price
                expected_prices[ticker] = float(np.mean(final_prices))
                
                # Confidence intervals
                ci_lower = np.percentile(final_prices, 2.5)
                ci_upper = np.percentile(final_prices, 97.5)
                confidence_intervals[ticker] = [float(ci_lower), float(ci_upper)]
            
            # 7. CALCULATE RISK METRICS
            logger.info("📉 Calculating risk metrics...")
            
            # Convert paths to returns for risk metrics
            paths_returns = np.diff(np.log(filtered_paths + 1e-8), axis=1)
            
            risk_metrics = {}
            for i, ticker in enumerate(tickers[:filtered_paths.shape[2]]):
                ticker_returns = paths_returns[:, :, i].flatten()
                
                # Calculate metrics
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
                
                risk_metrics[ticker] = {
                    'var_95': float(var_95),
                    'cvar_95': float(cvar_95),
                    'expected_return': float(expected_return),
                    'volatility': float(volatility),
                    'sharpe': float(sharpe),
                    'max_drawdown': float(max_dd)
                }
            
            # 8. OPTION PRICING (if requested)
            option_prices = {}
            if use_real_options:
                for ticker in tickers:
                    chain = self.data_fetcher.get_option_chain(ticker)
                    if chain and chain.get('calls'):
                        option_prices[ticker] = {
                            'underlying_price': chain.get('underlying_price', 0),
                            'calls': chain.get('calls', [])[:5] if chain.get('calls') else [],
                            'puts': chain.get('puts', [])[:5] if chain.get('puts') else [],
                            'expiration': chain.get('expiration', 'N/A')
                        }
            
            # 9. CALCULATE VARIANCE REDUCTION
            if self.use_gan:
                hist_var = np.var(returns.values.flatten())
                sim_var = np.var(np.diff(np.log(paths), axis=1).flatten())
                variance_reduction = max(0, (hist_var - sim_var) / (hist_var + 1e-8))
            else:
                variance_reduction = 0.0
            
            # 10. COMPILE RESULTS
            computation_time = (datetime.now() - start_time).total_seconds()
            
            results = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "tickers": tickers,
                    "n_simulations": self.n_simulations,
                    "filtered_paths": len(filtered_paths),
                    "computation_time": computation_time,
                    "data_source": self.data_fetcher.get_data_source_info(),
                    "use_gan": self.use_gan
                },
                "current_prices": current_prices,
                "expected_prices": expected_prices,
                "confidence_intervals": confidence_intervals,
                "risk_metrics": risk_metrics,
                "option_prices": option_prices,
                "variance_reduction": float(variance_reduction),
                "path_sample": filtered_paths[:10].tolist()
            }
            
            logger.info(f"✅ Simulation complete in {computation_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise