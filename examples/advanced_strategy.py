"""
Advanced strategy examples built on the Monte Carlo pipeline.
Run from the project root:  python examples/advanced_strategy.py

Everything here is computed from the pipeline's real simulation output / real history.
No hardcoded returns, volatilities or correlations.
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sklearn.covariance import LedoitWolf

from src.pipeline import MonteCarloPipeline
from src.pipeline.portfolio_optimizer import PortfolioOptimizer as MeanVarianceOptimizer
from src.explainer.ollama_wrapper import LlamaExplainer

TRADING_DAYS = 252
RISK_FREE = 0.02


def historical_moments(pipeline, requested: List[str], used: List[str], period: str = "2y"):
    """Annualised mean log-return and Ledoit-Wolf covariance from real history.
    `requested` must be the tuple the pipeline asked the fetcher for (that is its cache key)."""
    hist = pipeline.data_fetcher.get_historical_data(tuple(requested), period)
    if hist is None or hist.empty:
        raise ValueError("No historical data available")
    hist = hist[used]
    rets = np.log(hist / hist.shift(1)).dropna()
    mu = rets.mean().values * TRADING_DAYS
    cov = LedoitWolf().fit(rets.values).covariance_ * TRADING_DAYS
    return mu, cov, hist


class PortfolioAllocator:
    """Mean-variance allocation. (Renamed: it clashed with src.pipeline.portfolio_optimizer.PortfolioOptimizer.)"""

    def __init__(self, pipeline: MonteCarloPipeline):
        self.pipeline = pipeline

    def optimize_allocation(self, requested: List[str], results: Dict, period: str = "2y",
                            n_portfolios: int = 5000, seed: int = 42) -> Dict:
        used = results["metadata"]["tickers"]
        if not used:
            return {"error": "No tickers with data"}

        mu, cov, _ = historical_moments(self.pipeline, requested, used, period)
        opt = MeanVarianceOptimizer(mu, cov, asset_names=used)

        max_sharpe = opt.optimize_max_sharpe(RISK_FREE)
        min_vol = opt.optimize_min_volatility()
        frontier = opt.generate_efficient_frontier(30)

        # random-portfolio cloud for plotting (correct covariance: diagonal = real variances)
        rng = np.random.default_rng(seed)
        W = rng.dirichlet(np.ones(len(used)), size=n_portfolios)
        port_ret = W @ mu
        port_vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
        cloud = pd.DataFrame({"return": port_ret, "risk": port_vol,
                              "sharpe": (port_ret - RISK_FREE) / (port_vol + 1e-12)})

        def named(sol: Dict) -> Dict:
            return {**sol, "weights": dict(zip(used, sol["weights"]))}

        return {"tickers": used, "max_sharpe": named(max_sharpe), "min_vol": named(min_vol),
                "frontier": frontier, "cloud": cloud}


class TailRiskAnalyzer:
    """Portfolio tail risk from the pipeline's simulated paths (joint across assets)."""

    def __init__(self, pipeline: MonteCarloPipeline):
        self.pipeline = pipeline

    def _start_prices(self, requested: List[str], results: Dict, used: List[str]) -> np.ndarray:
        sp = results.get("start_prices")
        if sp:
            return np.array([sp[t] for t in used], float)
        hist = self.pipeline.data_fetcher.get_historical_data(tuple(requested), "2y")
        return hist[used].iloc[-1].values.astype(float)

    def analyze(self, requested: List[str], results: Dict, weights: Dict[str, float]) -> Dict:
        used = results["metadata"]["tickers"]
        w = np.array([weights.get(t, 0.0) for t in used], float)
        if w.sum() <= 0:
            raise ValueError("Weights are all zero for the simulated tickers")
        w = w / w.sum()

        # price_distributions[t][k] is path k's final price -> aligned across tickers
        final = np.column_stack([results["price_distributions"][t] for t in used])
        p0 = self._start_prices(requested, results, used)
        port = (final / p0 - 1.0) @ w

        var_95 = float(np.percentile(port, 5))
        tail = port[port <= var_95]
        sample = results.get("path_sample")
        horizon = len(sample[0]) - 1 if sample else 59

        return {
            "horizon_days": horizon,
            "weights": dict(zip(used, w.round(4).tolist())),
            "expected_return": float(port.mean()),
            "VaR_95": var_95,
            "CVaR_95": float(tail.mean()) if len(tail) else var_95,
            "prob_loss": float((port < 0).mean()),
            "prob_loss_10pct": float((port < -0.10).mean()),
        }


def main() -> int:
    print("=" * 60)
    print("ADVANCED MONTE CARLO STRATEGIES")
    print("=" * 60)

    tickers = ["AAPL", "MSFT", "GOOGL"]
    pipeline = MonteCarloPipeline(n_assets=len(tickers), n_simulations=2000, use_live_data=False)
    pipeline.load_models()  # no-op unless use_gan=True and a trained checkpoint is supplied

    try:
        results = pipeline.run_simulation(tickers, period="2y")  # ONE simulation, reused below
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1

    missing = results["metadata"].get("missing_tickers", [])
    if missing:
        print(f"WARNING: no data for {missing}; continuing with {results['metadata']['tickers']}")

    # ---- allocation ----
    print("\nBEST PORTFOLIO (max Sharpe):")
    try:
        alloc = PortfolioAllocator(pipeline).optimize_allocation(tickers, results)
        if "error" in alloc:
            print("  failed:", alloc["error"])
        else:
            ms = alloc["max_sharpe"]
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ms.items()})
    except Exception as e:
        print(f"  allocation failed: {e}")

    # ---- tail risk ----
    print("\nTAIL RISK (equal weight, simulated horizon):")
    try:
        used = results["metadata"]["tickers"]
        risk = TailRiskAnalyzer(pipeline).analyze(tickers, results, {t: 1.0 / len(used) for t in used})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in risk.items()})
    except Exception as e:
        print(f"  tail-risk analysis failed: {e}")

    # ---- LLM: real numbers only ----
    print("\nAI ANALYSIS:")
    try:
        explainer = LlamaExplainer()
        print(explainer.explain_simulation_results(
            tickers=results["metadata"]["tickers"],
            expected_prices=results["expected_prices"],
            confidence_intervals=results["confidence_intervals"],
            risk_metrics=results["risk_metrics"],
            variance_reduction=results["variance_reduction"],
        ))
    except Exception as e:
        print(f"  LLM unavailable: {e}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())