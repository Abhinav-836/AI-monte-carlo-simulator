"""
Basic usage example for the Monte Carlo Hybrid Simulator.
Run from the project root:  python examples/basic_usage.py   (add --ask for the Q&A demo)
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sklearn.covariance import LedoitWolf

from src.pipeline import MonteCarloPipeline
from src.pipeline.portfolio_optimizer import PortfolioOptimizer
from src.explainer.ollama_wrapper import LlamaExplainer

RESULTS_FILE = "simulation_results.json"


def safe_get(data, key, default=None):
    return data.get(key, default) if data else default


def to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def main() -> int:
    print("=" * 60)
    print("MONTE CARLO HYBRID SIMULATOR v2.0")
    print("=" * 60)

    try:
        print("\n1. Initializing pipeline...")
        requested = ["AAPL", "MSFT", "GOOGL"]
        pipeline = MonteCarloPipeline(n_assets=len(requested), n_simulations=2000,
                                      filter_top_k=0.1, use_live_data=False)
        pipeline.load_models()

        print(f"\n2. Running simulation for {', '.join(requested)}...")
        results = pipeline.run_simulation(tickers=requested, period="1y", use_real_options=False)

        metadata = safe_get(results, "metadata", {})
        expected_prices = safe_get(results, "expected_prices", {})
        confidence_intervals = safe_get(results, "confidence_intervals", {})
        risk_metrics = safe_get(results, "risk_metrics", {})
        portfolio_stats = safe_get(results, "portfolio_stats", {})
        tickers = safe_get(metadata, "tickers", [])  # tickers that actually had data

        missing = safe_get(metadata, "missing_tickers", [])
        if missing:
            print(f"   WARNING: no market data for {missing} (check API keys / provider limits)")

        print(f"   Simulation complete in {safe_get(metadata, 'computation_time', 0):.2f}s")
        print(f"   Generated {safe_get(metadata, 'n_simulations', 0)} paths "
              f"(stats use all paths; {safe_get(metadata, 'filtered_paths', 0)} kept as a display sample)")
        print(f"   Data source: {safe_get(metadata, 'data_source', 'unknown')}")

        print("\n3. Key results:")
        print("-" * 40)
        for t in tickers:
            exp = safe_get(expected_prices, t)
            ci = safe_get(confidence_intervals, t)
            risk = safe_get(risk_metrics, t, {})
            print(f"\n{t}:")
            if exp is None or ci is None:
                print("   no result")
                continue
            print(f"   Expected price: {exp:.2f}")
            print(f"   95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
            print(f"   Daily VaR (95%): {safe_get(risk, 'var_95', 0) * 100:.2f}%")
            print(f"   Sharpe: {safe_get(risk, 'sharpe', 0):.2f}")
            print(f"   Sortino: {safe_get(risk, 'sortino', 0):.2f}")
            print(f"   Annualised return: {safe_get(risk, 'expected_return', 0) * 100:.1f}%")

        print("\n4. Portfolio statistics (equal weight):")
        print("-" * 40)
        print(f"   Annualised return: {safe_get(portfolio_stats, 'mean_return', 0) * 100:.1f}%")
        print(f"   Volatility: {safe_get(portfolio_stats, 'volatility', 0) * 100:.1f}%")
        print(f"   Sharpe: {safe_get(portfolio_stats, 'sharpe', 0):.2f}")
        print(f"   Daily VaR (95%): {safe_get(portfolio_stats, 'var_95', 0) * 100:.2f}%")

        print("\n5. Portfolio optimization (historical mean / Ledoit-Wolf covariance):")
        print("-" * 40)
        if len(tickers) >= 1:
            hist = pipeline.data_fetcher.get_historical_data(tuple(requested), "1y")
            rets = np.log(hist[tickers] / hist[tickers].shift(1)).dropna()
            mu = rets.mean().values * 252
            cov = LedoitWolf().fit(rets.values).covariance_ * 252
            optimal = PortfolioOptimizer(mu, cov, asset_names=tickers).optimize_max_sharpe()
            for t, w in zip(tickers, optimal["weights"]):
                print(f"      {t}: {w * 100:.1f}%")
            print(f"   Expected return: {optimal['return'] * 100:.1f}%")
            print(f"   Volatility: {optimal['volatility'] * 100:.1f}%")
            print(f"   Sharpe: {optimal['sharpe']:.2f}")
            print("   (Max-Sharpe on 1y of history is very sensitive to estimation error; "
                  "treat it as an illustration.)")

        print("\n6. AI explanation...")
        try:
            explainer = LlamaExplainer()
            explanation = explainer.explain_simulation_results(
                tickers=tickers, expected_prices=expected_prices,
                confidence_intervals=confidence_intervals, risk_metrics=risk_metrics,
                variance_reduction=safe_get(results, "variance_reduction", 0),
                style="professional",
            )
            print("\n" + "=" * 60 + "\nAI FINANCIAL ANALYSIS\n" + "=" * 60)
            print(explanation)
        except Exception as e:
            print(f"   AI explanation unavailable ({e}). Raw metrics shown above.")

        print("\n7. Saving results...")
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=to_serializable)
        print(f"   Saved to {RESULTS_FILE}")

    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60 + "\nSIMULATION COMPLETE\n" + "=" * 60)
    return 0


def ask_question() -> int:
    print("\n" + "=" * 60 + "\nASK QUESTIONS ABOUT SIMULATION RESULTS\n" + "=" * 60)
    try:
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    except FileNotFoundError:
        print("No simulation results found. Run main() first.")
        return 1

    try:
        explainer = LlamaExplainer()
    except Exception as e:
        print(f"Explainer unavailable: {e}")
        return 1

    questions = [
        "What's the probability of a 10% drop in AAPL?",
        "Which stock has the best risk-adjusted return?",
        "How risky is MSFT relative to the others?",
        "What's the main risk factor I should watch?",
    ]
    for i, q in enumerate(questions, 1):
        print(f"\n{i}. Q: {q}")
        try:
            answer = explainer.explain_simulation_results(
                tickers=list(results.get("expected_prices", {}).keys()),
                expected_prices=results.get("expected_prices", {}),
                confidence_intervals=results.get("confidence_intervals", {}),
                risk_metrics=results.get("risk_metrics", {}),
                variance_reduction=results.get("variance_reduction", 0),
                user_question=q,
            )
            print(f"   A: {answer}")
        except Exception as e:
            print(f"   Could not get answer: {e}")
        print("-" * 40)
    return 0


if __name__ == "__main__":
    code = main()
    if code == 0 and "--ask" in sys.argv:
        code = ask_question()
    sys.exit(code)