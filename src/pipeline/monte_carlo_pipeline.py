"""
Monte Carlo Pipeline with real-time updates
"""

import logging
import os
import queue
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class MonteCarloPipeline:
    def __init__(
        self,
        n_assets: int = 5,
        n_simulations: int = 1000,
        filter_top_k: float = 0.1,
        seq_len: int = 60,
        use_gan: bool = False,
        use_live_data: bool = True,
    ):
        self.n_assets = n_assets
        self.n_simulations = n_simulations
        self.filter_top_k = filter_top_k
        self.seq_len = seq_len
        self.use_gan = use_gan
        self.use_live_data = use_live_data

        from src.pipeline.data_fetcher import get_data_fetcher
        from src.pipeline.risk.metrics import RiskCalculator
        from src.pipeline.risk.confidence import ConfidenceIntervalCalculator

        # shared, cached fetcher: one cache + one rate limiter for the whole app
        self.data_fetcher = get_data_fetcher(use_live_simulation=use_live_data)
        self.risk_calc = RiskCalculator()
        self.ci_calc = ConfidenceIntervalCalculator()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = None
        self.discriminator = None
        self.filter = None
        self.dfmgan_config = None

        self.live_prices: Dict[str, float] = {}
        self.price_history: Dict[str, list] = {}
        self.update_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self.update_thread = None

        self.simulation_history: list = []
        self.performance_metrics: dict = {}

    # ------------------------------------------------------------
    # GAN
    # ------------------------------------------------------------
    def load_models(self, n_assets: Optional[int] = None, checkpoint_path: Optional[str] = None):
        """
        Build GAN models. An UNTRAINED generator emits garbage, so the GAN is only used
        when a trained checkpoint ({'generator': state_dict, 'discriminator': state_dict}) is loaded.
        """
        if not self.use_gan:
            return
        if n_assets:
            self.n_assets = n_assets
        try:
            from src.pipeline.dfmgan.generator import (
                DFMGANConfig, ConditionalMCGenerator, TimeGANDiscriminator, PathFilter)

            self.dfmgan_config = DFMGANConfig(
                n_assets=self.n_assets, seq_len=self.seq_len,
                n_simulations=self.n_simulations, filter_top_k=self.filter_top_k)
            self.generator = ConditionalMCGenerator(self.dfmgan_config).to(self.device)
            self.discriminator = TimeGANDiscriminator(self.dfmgan_config).to(self.device)
            self.filter = PathFilter(self.discriminator)

            if checkpoint_path and os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                self.generator.load_state_dict(ckpt["generator"])
                self.discriminator.load_state_dict(ckpt["discriminator"])
                self.generator.eval()
                logger.info("Trained GAN checkpoint loaded")
            else:
                logger.warning("No trained GAN checkpoint -> falling back to GBM Monte Carlo")
                self.generator = None
                self.use_gan = False
        except Exception as e:
            logger.warning(f"Could not load GAN models: {e}")
            self.generator = None
            self.use_gan = False

    # ------------------------------------------------------------
    # LIVE UPDATES (explicit opt-in; no longer triggered by every simulation)
    # ------------------------------------------------------------
    def start_live_updates(self, tickers: List[str]):
        if not self.use_live_data or self.is_running:
            return
        self.data_fetcher.start_live_simulation(tickers)
        self.data_fetcher.subscribe_prices(self._on_price_update)
        self.is_running = True
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self._process_updates, daemon=True)
            self.update_thread.start()

    def stop_live_updates(self):
        self.is_running = False
        self.data_fetcher.stop_live_simulation()

    def _on_price_update(self, prices: Dict[str, float]):
        self.update_queue.put(prices)

    def _process_updates(self):
        while self.is_running:
            try:
                prices = self.update_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.live_prices = prices
                for ticker, price in prices.items():
                    hist = self.price_history.setdefault(ticker, [])
                    hist.append({"time": datetime.now(), "price": price})
                    if len(hist) > 500:
                        self.price_history[ticker] = hist[-500:]
            except Exception as e:
                logger.error(f"Update processing error: {e}")

    def get_live_prices(self) -> Dict[str, float]:
        return self.live_prices.copy()

    def get_price_history(self, ticker: str) -> List[Dict]:
        return self.price_history.get(ticker, [])

    # ------------------------------------------------------------
    # SIMULATION
    # ------------------------------------------------------------
    def _estimate_covariance_matrix(self, returns: pd.DataFrame) -> np.ndarray:
        try:
            from sklearn.covariance import LedoitWolf
            cov = LedoitWolf().fit(returns.values).covariance_
        except Exception:
            cov = returns.cov().values
            cov = 0.8 * cov + 0.2 * np.diag(np.diag(cov))

        n = cov.shape[0]
        jitter = max(1e-8, 1e-6 * np.trace(cov) / max(n, 1))
        for _ in range(5):
            try:
                np.linalg.cholesky(cov)
                return cov
            except np.linalg.LinAlgError:
                cov = cov + np.eye(n) * jitter
                jitter *= 10
        return cov + np.eye(n) * (np.abs(cov).max() * 1e-3 + 1e-6)

    def _simulate_gbm_paths(self, returns: pd.DataFrame, last_prices: np.ndarray,
                            n_steps: int = 60) -> np.ndarray:
        n_paths, n_assets = self.n_simulations, len(returns.columns)
        mu = returns.mean().values
        cov = self._estimate_covariance_matrix(returns)

        jump_prob, jump_mean, jump_std = 0.05, -0.02, 0.03
        # compensate so jumps don't add a hidden negative drift on top of mu
        jump_comp = jump_prob * (np.exp(jump_mean + 0.5 * jump_std ** 2) - 1)

        L = np.linalg.cholesky(cov)
        drift = mu - 0.5 * np.diag(cov) - jump_comp

        paths = np.zeros((n_paths, n_steps, n_assets))
        paths[:, 0, :] = last_prices
        for t in range(1, n_steps):
            diffusion = np.random.randn(n_paths, n_assets) @ L.T
            jumps = np.random.binomial(1, jump_prob, (n_paths, n_assets))
            jump_ret = jumps * np.random.normal(jump_mean, jump_std, (n_paths, n_assets))
            paths[:, t, :] = paths[:, t - 1, :] * np.exp(drift + diffusion + jump_ret)
        return paths

    def _filter_paths_advanced(self, paths: np.ndarray, returns: pd.DataFrame) -> np.ndarray:
        """Select the most 'historically plausible' paths. For display/sampling only —
        do NOT compute risk from these: the filter removes the tails."""
        n_paths, n_steps = len(paths), paths.shape[1]
        k = max(1, int(n_paths * self.filter_top_k))

        final_returns = paths[:, -1, :] / (paths[:, 0, :] + 1e-8) - 1
        hist_horizon = (1 + returns.mean().values) ** (n_steps - 1) - 1  # same scale as final_returns
        scores = np.abs(final_returns - hist_horizon).mean(axis=1)

        path_vol = np.std(np.diff(np.log(paths + 1e-8), axis=1), axis=1).mean(axis=1)
        scores = scores + np.abs(path_vol - returns.std().values.mean())

        if paths.shape[2] > 1:
            sample_idx = np.random.choice(n_paths, min(200, n_paths), replace=False)
            hist_corr = returns.iloc[-20:].corr().values
            for i in sample_idx:
                recent = paths[i, -20:, :]
                if recent.std(axis=0).min() < 1e-8:
                    continue
                pc = np.corrcoef(recent.T)
                if np.isnan(pc).any() or np.isnan(hist_corr).any():
                    continue
                scores[i] += np.abs(np.mean(pc - hist_corr))

        scores = np.nan_to_num(scores, nan=np.inf, posinf=np.inf, neginf=np.inf)
        return paths[np.argsort(scores)[:k]]

    def run_simulation(
        self,
        tickers: List[str],
        period: str = "2y",
        use_real_options: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        start_time = datetime.now()
        cb = progress_callback or (lambda *_: None)

        cb(10, "Fetching historical data...")
        historical = self.data_fetcher.get_historical_data(tuple(tickers), period)
        if historical is None or historical.empty:
            raise ValueError(
                "No historical data available. Provider errors: "
                f"{getattr(self.data_fetcher, 'last_errors', {})}")

        # Columns are what we really got. Re-key everything on them so a failed ticker
        # can never shift another ticker's data.
        available = [t for t in tickers if t in historical.columns]
        missing = [t for t in tickers if t not in historical.columns]
        historical = historical[available]
        tickers = available
        if missing:
            logger.warning(f"No data for {missing}; simulating {available}")

        cb(20, "Fetching current prices...")
        current_prices = self.data_fetcher.get_current_prices(tickers)

        returns = historical.pct_change().dropna()
        if returns.empty:
            raise ValueError("No valid returns computed from historical data")

        last_prices = []
        for t in tickers:
            hist_last = float(historical[t].iloc[-1])
            cur = current_prices.get(t)
            # use the live price only if it is consistent with the history (same currency/scale)
            last_prices.append(cur if cur and 0.75 < cur / hist_last < 1.25 else hist_last)
        last_prices = np.array(last_prices)

        cb(30, "Generating price paths...")
        n_assets = len(tickers)
        if self.use_gan and (self.generator is None or self.generator.config.n_assets != n_assets):
            self.load_models(n_assets=n_assets)

        if self.use_gan and self.generator is not None:
            hist_tensor = torch.tensor(historical.values[-self.seq_len:],
                                       dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                paths = self.generator(hist_tensor, self.n_simulations).cpu().numpy()
        else:
            paths = self._simulate_gbm_paths(returns, last_prices, n_steps=60)

        cb(50, "Filtering paths...")
        filtered_paths = self._filter_paths_advanced(paths, returns)

        # Statistics use ALL paths (filtered ones have artificially thin tails)
        cb(60, "Computing expected prices...")
        expected_prices, confidence_intervals, price_distributions = {}, {}, {}
        for i, ticker in enumerate(tickers):
            final = paths[:, -1, i]
            expected_prices[ticker] = float(np.mean(final))
            confidence_intervals[ticker] = [float(np.percentile(final, 2.5)),
                                            float(np.percentile(final, 97.5))]
            price_distributions[ticker] = final.tolist()

        cb(70, "Calculating risk metrics...")
        paths_returns = np.diff(np.log(paths + 1e-8), axis=1)
        risk_metrics = {}
        for i, ticker in enumerate(tickers):
            r = paths_returns[:, :, i].flatten()
            var_95 = np.percentile(r, 5)
            cvar_95 = np.mean(r[r <= var_95])
            exp_ret = np.mean(r) * 252
            vol = np.std(r) * np.sqrt(252)
            downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252)

            pp = paths[:, :, i]
            peak = np.maximum.accumulate(pp, axis=1)
            max_dd = float(np.min((pp - peak) / (peak + 1e-8)))

            risk_metrics[ticker] = {
                "var_95": float(var_95), "cvar_95": float(cvar_95),
                "expected_return": float(exp_ret), "volatility": float(vol),
                "sharpe": float((exp_ret - 0.02) / (vol + 1e-8)),
                "sortino": float((exp_ret - 0.02) / (downside + 1e-8)),
                "max_drawdown": max_dd,
            }

        cb(80, "Fetching options data...")
        option_prices = {}
        if use_real_options:
            for ticker in tickers:
                if "." not in ticker:
                    try:
                        chain = self.data_fetcher.get_option_chain(ticker)
                        if chain:
                            option_prices[ticker] = chain
                    except Exception as e:
                        logger.warning(f"Option fetch failed for {ticker}: {e}")

        hist_var = np.var(returns.values.flatten())
        sim_var = np.var(paths_returns.flatten())
        variance_reduction = max(0.0, (hist_var - sim_var) / (hist_var + 1e-8))

        cb(90, "Computing portfolio statistics...")
        port = paths_returns.mean(axis=2).flatten()
        var_p = np.percentile(port, 5)
        portfolio_stats = {
            "mean_return": float(np.mean(port) * 252),
            "volatility": float(np.std(port) * np.sqrt(252)),
            "sharpe": float(np.mean(port) * 252 / (np.std(port) * np.sqrt(252) + 1e-8)),
            "var_95": float(var_p),
            "cvar_95": float(np.mean(port[port <= var_p])),
        }

        computation_time = (datetime.now() - start_time).total_seconds()
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "tickers": tickers,
                "missing_tickers": missing,
                "n_simulations": self.n_simulations,
                "filtered_paths": len(filtered_paths),
                "computation_time": computation_time,
                "data_source": self.data_fetcher.get_data_source_info(),
                "use_gan": self.use_gan,
                "use_live_data": self.use_live_data,
            },
            "current_prices": current_prices,
            "expected_prices": expected_prices,
            "confidence_intervals": confidence_intervals,
            "price_distributions": price_distributions,
            "risk_metrics": risk_metrics,
            "option_prices": option_prices,
            "portfolio_stats": portfolio_stats,
            "variance_reduction": float(variance_reduction),
            "path_sample": filtered_paths[:10].tolist(),
        }
        self.simulation_history.append(
            {"timestamp": datetime.now(), "tickers": tickers, "results": results})
        cb(100, "Complete!")
        return results

    def get_performance_metrics(self) -> Dict:
        if not self.simulation_history:
            return {}
        return {
            "total_runs": len(self.simulation_history),
            "avg_computation_time": float(np.mean(
                [s["results"]["metadata"]["computation_time"] for s in self.simulation_history])),
            "avg_variance_reduction": float(np.mean(
                [s["results"]["variance_reduction"] for s in self.simulation_history])),
        }