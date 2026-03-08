"""
Integration tests for the full system
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline import MonteCarloPipeline
from src.pipeline.data_fetcher import DataFetcher


class TestIntegration(unittest.TestCase):
    """Test full system integration"""

    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)

        self.pipeline = MonteCarloPipeline(
            n_assets=2,
            n_simulations=5,  # Small for fast tests
            filter_top_k=0.2
        )

        # Mock DataFetcher
        class MockFetcher:
            def get_historical_data(self, tickers_tuple, period="1y"):
                tickers = list(tickers_tuple)
                dates = pd.date_range('2020-01-01', periods=50)
                data = {}
                for ticker in tickers:
                    data[ticker] = 100 + np.cumsum(np.random.randn(50))
                return pd.DataFrame(data, index=dates)

            def get_current_prices(self, tickers):
                return {t: float(100 + np.random.rand() * 50) for t in tickers}

            def get_data_source_info(self):
                return "Mock Data"

        self.pipeline.data_fetcher = MockFetcher()

        # Load models
        try:
            self.pipeline.load_models()
        except Exception as e:
            print(f"Model loading warning: {e}")

    def test_pipeline_execution(self):
        """Test that pipeline runs without errors"""
        tickers = ["AAPL", "MSFT"]
        results = self.pipeline.run_simulation(tickers, period="1mo")

        self.assertIsNotNone(results)
        self.assertIn('metadata', results)
        self.assertIn('expected_prices', results)

    def test_risk_metrics(self):
        """Test risk metrics calculation"""
        tickers = ["AAPL", "MSFT"]
        results = self.pipeline.run_simulation(tickers)

        self.assertIn('risk_metrics', results)
        self.assertEqual(len(results['risk_metrics']), 2)

        for ticker, metrics in results['risk_metrics'].items():
            self.assertIn('var_95', metrics)
            self.assertIn('expected_return', metrics)
            self.assertIn('sharpe', metrics)


if __name__ == '__main__':
    unittest.main()