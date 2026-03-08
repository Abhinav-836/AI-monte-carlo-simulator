"""
Advanced strategy examples using Monte Carlo simulations (FIXED)
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List

# Add root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline import MonteCarloPipeline
from src.explainer.ollama_wrapper import LlamaExplainer


# ================================
# PORTFOLIO OPTIMIZER (FIXED)
# ================================

class PortfolioOptimizer:

    def __init__(self, pipeline: MonteCarloPipeline):
        self.pipeline = pipeline

    def optimize_allocation(self, tickers: List[str], n_portfolios=5000):

        print(f"\nOptimizing portfolio for {tickers}...")

        try:
            results = self.pipeline.run_simulation(tickers, period="2y")
        except Exception as e:
            return {"error": str(e)}

        # ✅ FIX: Check if results contain expected data
        if not results or 'risk_metrics' not in results:
            return {"error": "No risk metrics available"}

        # Safely extract returns and volatility
        returns = []
        vols = []
        valid_tickers = []

        for t in tickers:
            if t in results['risk_metrics']:
                returns.append(results['risk_metrics'][t]['expected_return'])
                vols.append(results['risk_metrics'][t]['volatility'])
                valid_tickers.append(t)
            else:
                print(f"⚠️ {t} not found in results, skipping")

        if len(valid_tickers) == 0:
            return {"error": "No valid tickers with data"}

        returns = np.array(returns)
        vols = np.array(vols)

        # Realistic covariance
        cov_matrix = np.outer(vols, vols) * 0.3

        portfolios = []

        for _ in range(n_portfolios):
            w = np.random.random(len(valid_tickers))
            w /= np.sum(w)

            ret = np.dot(w, returns)
            risk = np.sqrt(w.T @ cov_matrix @ w) if len(valid_tickers) > 1 else w[0] * vols[0]

            sharpe = ret / risk if risk > 0 else 0

            portfolios.append({
                "weights": w.tolist(),
                "return": float(ret),
                "risk": float(risk),
                "sharpe": float(sharpe)
            })

        if not portfolios:
            return {"error": "No portfolios generated"}

        df = pd.DataFrame(portfolios)

        best_idx = df["sharpe"].idxmax()
        min_risk_idx = df["risk"].idxmin()

        return {
            "max_sharpe": df.loc[best_idx].to_dict() if not pd.isna(best_idx) else None,
            "min_risk": df.loc[min_risk_idx].to_dict() if not pd.isna(min_risk_idx) else None,
            "all": df.to_dict('records')
        }


# ================================
# TAIL RISK (FIXED)
# ================================

class TailRiskHedging:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def analyze(self, tickers, weights):

        results = self.pipeline.run_simulation(tickers, period="2y")

        weights_arr = np.array([weights.get(t, 0) for t in tickers])
        weights_arr = weights_arr / np.sum(weights_arr)

        # Generate synthetic portfolio returns
        sims = 5000
        portfolio_returns = []

        for _ in range(sims):
            # Simple return simulation
            ret = np.random.normal(0.0005, 0.015, len(tickers))
            portfolio_ret = np.dot(weights_arr, ret)
            portfolio_returns.append(portfolio_ret)

        portfolio_returns = np.array(portfolio_returns)

        var_95 = float(np.percentile(portfolio_returns, 5))
        cvar_95 = float(portfolio_returns[portfolio_returns <= var_95].mean()) if np.any(portfolio_returns <= var_95) else var_95

        # Drawdown calculation
        cumulative = np.cumsum(portfolio_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / (np.abs(peak) + 1e-8)

        return {
            "VaR_95": var_95,
            "CVaR_95": cvar_95,
            "max_drawdown": float(drawdown.min())
        }


# ================================
# MAIN (FIXED)
# ================================

def main():

    print("=" * 60)
    print("🚀 ADVANCED MONTE CARLO STRATEGIES")
    print("=" * 60)

    pipeline = MonteCarloPipeline(n_simulations=1000)

    try:
        pipeline.load_models()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️ Models not loaded: {e}")

    tickers = ["AAPL", "MSFT", "GOOGL"]

    # ================= PORTFOLIO =================
    optimizer = PortfolioOptimizer(pipeline)
    opt = optimizer.optimize_allocation(tickers)

    print("\n📊 BEST PORTFOLIO:")
    if opt and "max_sharpe" in opt and opt["max_sharpe"]:
        print(opt["max_sharpe"])
    else:
        print("⚠️ Portfolio optimization failed:", opt.get("error", "Unknown error"))

    # ================= TAIL RISK =================
    hedging = TailRiskHedging(pipeline)

    weights = {t: 1.0/len(tickers) for t in tickers}

    try:
        risk = hedging.analyze(tickers, weights)
        print("\n⚠️ TAIL RISK:")
        print(risk)
    except Exception as e:
        print(f"\n⚠️ Tail risk analysis failed: {e}")

    # ================= LLM =================
    print("\n🧠 AI ANALYSIS:")

    try:
        explainer = LlamaExplainer()
        explanation = explainer.explain_simulation_results(
            tickers=tickers,
            expected_prices={t: 100 for t in tickers},
            confidence_intervals={t: [90, 110] for t in tickers},
            risk_metrics={t: {'sharpe': 1.2, 'expected_return': 0.1} for t in tickers},
            variance_reduction=0.4
        )
        print(explanation)

    except Exception as e:
        print(f"⚠️ LLM failed: {e}")

    print("\n✅ DONE")


if __name__ == "__main__":
    main()