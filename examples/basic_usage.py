"""
Basic usage example for Monte Carlo Hybrid Simulator
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.pipeline import MonteCarloPipeline
from src.explainer.ollama_wrapper import LlamaExplainer
import json
import traceback

def safe_get(data, key, default=None):
    """Safely get dictionary value"""
    return data.get(key, default) if data else default

def main():
    """Run a basic simulation and get LLM explanation"""
    
    print("=" * 60)
    print("MONTE CARLO HYBRID SIMULATOR")
    print("=" * 60)
    
    try:
        # Step 1: Initialize pipeline
        print("\n1. Initializing ML pipeline...")
        pipeline = MonteCarloPipeline(
            n_assets=3,
            n_simulations=1000,  # Reduced for speed
            filter_top_k=0.1      # Keep top 10% for better stats
        )
        pipeline.load_models()
        
        # Step 2: Run simulation
        print("\n2. Running simulation for AAPL, MSFT, GOOGL...")
        tickers = ["AAPL", "MSFT", "GOOGL"]
        
        results = pipeline.run_simulation(
            tickers=tickers,
            period="1y"  # Shorter period for faster fetching
        )
        
        # Safe access with defaults
        metadata = safe_get(results, 'metadata', {})
        expected_prices = safe_get(results, 'expected_prices', {})
        confidence_intervals = safe_get(results, 'confidence_intervals', {})
        risk_metrics = safe_get(results, 'risk_metrics', {})
        
        print(f"   ✓ Simulation complete in {safe_get(metadata, 'computation_time', 0):.2f}s")
        print(f"   ✓ Generated {safe_get(metadata, 'n_simulations', 0)} paths")
        print(f"   ✓ Filtered to {safe_get(metadata, 'filtered_paths', 0)} most realistic")
        print(f"   ✓ Variance reduction: {safe_get(results, 'variance_reduction', 0)*100:.1f}%")
        
        # Step 3: Display key results
        print("\n3. Key Results:")
        print("-" * 40)
        
        for ticker in tickers:
            exp = safe_get(expected_prices, ticker, 0)
            ci = safe_get(confidence_intervals, ticker, [0, 0])
            risk = safe_get(risk_metrics, ticker, {})
            
            print(f"\n{ticker}:")
            print(f"   Expected price: ${exp:.2f}")
            print(f"   95% CI: [${ci[0]:.2f}, ${ci[1]:.2f}]")
            print(f"   VaR (95%): {safe_get(risk, 'var_95', 0)*100:.1f}%")
            print(f"   Sharpe: {safe_get(risk, 'sharpe', 0):.2f}")
            print(f"   Expected return: {safe_get(risk, 'expected_return', 0)*100:.1f}%")
        
        # Step 4: Get LLM explanation with fallback
        print("\n4. Getting AI explanation...")
        
        explainer = LlamaExplainer()
        
        # Safe explanation with fallback
        try:
            explanation = explainer.explain_simulation_results(
                tickers=tickers,
                expected_prices=expected_prices,
                confidence_intervals=confidence_intervals,
                risk_metrics=risk_metrics,
                variance_reduction=safe_get(results, 'variance_reduction', 0)
            )
            
            print("\n" + "=" * 60)
            print("AI FINANCIAL ANALYSIS")
            print("=" * 60)
            print(explanation)
        except Exception as e:
            print(f"\n⚠️ AI explanation unavailable: {e}")
            print("Showing raw metrics instead.")
        
        # Step 5: Save results
        print("\n5. Saving results...")
        with open("simulation_results.json", "w") as f:
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            serializable_results = json.loads(
                json.dumps(results, default=convert_to_serializable)
            )
            json.dump(serializable_results, f, indent=2)
        
        print("   ✓ Results saved to simulation_results.json")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


def ask_question():
    """Example of asking a specific question"""
    
    print("\n" + "=" * 60)
    print("ASK A QUESTION ABOUT SIMULATION RESULTS")
    print("=" * 60)
    
    # Load results
    try:
        with open("simulation_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("No simulation results found. Run basic_usage() first.")
        return
    
    # Initialize explainer with fallback
    explainer = LlamaExplainer()
    
    # Ask questions
    questions = [
        "What's the probability of a 10% drop in AAPL?",
        "Which stock has the best risk-adjusted return?",
        "Should I buy MSFT based on this simulation?",
        "What's the main risk factor I should watch?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Q: {question}")
        
        try:
            answer = explainer.answer_question(results, question)
            print(f"   A: {answer}")
        except Exception as e:
            print(f"   ⚠️ Could not get answer: {e}")
        print("-" * 40)


if __name__ == "__main__":
    main()