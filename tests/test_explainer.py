"""
Unit tests for LLM explainer module
"""

import unittest
import sys
import os

# Add root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.explainer.ollama_wrapper import LlamaExplainer
from src.explainer.prompt_templates import PromptTemplates
from src.explainer.response_parser import ResponseParser


class TestPromptTemplates(unittest.TestCase):
    """Test prompt template generation"""

    def setUp(self):
        self.templates = PromptTemplates()
        self.tickers = ["AAPL", "MSFT"]
        self.expected_prices = {"AAPL": 175.32, "MSFT": 380.45}
        self.confidence_intervals = {
            "AAPL": [168.21, 182.43],
            "MSFT": [372.18, 388.72]
        }
        self.risk_metrics = {
            "AAPL": {
                "var_95": -0.023,
                "sharpe": 1.42,
                "expected_return": 0.12,
                "volatility": 0.25
            },
            "MSFT": {
                "var_95": -0.018,
                "sharpe": 1.38,
                "expected_return": 0.10,
                "volatility": 0.22
            }
        }

    def test_simulation_summary(self):
        prompt = self.templates.simulation_summary(
            self.tickers,
            self.expected_prices,
            self.confidence_intervals,
            self.risk_metrics,
            variance_reduction=0.42
        )
        self.assertIsInstance(prompt, str)
        self.assertIn("AAPL", prompt)
        self.assertIn("MSFT", prompt)

    def test_risk_analysis(self):
        prompt = self.templates.risk_analysis(
            self.tickers,
            self.risk_metrics,
            variance_reduction=0.42
        )
        self.assertIsInstance(prompt, str)

    def test_compare_assets(self):
        prompt = self.templates.compare_assets(
            self.tickers,
            self.expected_prices,
            self.risk_metrics
        )
        self.assertIsInstance(prompt, str)


class TestResponseParser(unittest.TestCase):
    """Test response parsing"""

    def setUp(self):
        self.parser = ResponseParser()

    def test_clean_response(self):
        messy = "  This is a test.  \n\n\nWith extra spaces.  "
        cleaned = self.parser.clean_response(messy)
        self.assertEqual(cleaned, "This is a test.\n\nWith extra spaces.")

    def test_extract_bullet_points(self):
        text = """
        • First point
        • Second point with details
        • Third point
        """
        bullets = self.parser.extract_bullet_points(text)
        self.assertEqual(len(bullets), 3)

    def test_extract_key_metrics(self):
        text = "The VaR is 2.3% and Sharpe ratio is 1.42. Price target $175.32."
        metrics = self.parser.extract_key_metrics(text)
        self.assertIsInstance(metrics, dict)


class TestLlamaExplorer(unittest.TestCase):
    """Test LLaMA wrapper (mock)"""

    def test_initialization(self):
        # Test that it initializes without error
        explainer = LlamaExplainer()
        self.assertIsNotNone(explainer)


if __name__ == '__main__':
    unittest.main()