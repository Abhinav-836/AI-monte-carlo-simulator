"""
Unit tests for the LLM explainer helpers.
Assertions are behavioural (not exact strings) and the LLM backend is never required.
"""

import os
import re
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.explainer.ollama_wrapper import LlamaExplainer
from src.explainer.prompt_templates import PromptTemplates
from src.explainer.response_parser import ResponseParser


class TestPromptTemplates(unittest.TestCase):

    def setUp(self):
        self.t = PromptTemplates()
        self.tickers = ["AAPL", "MSFT"]
        self.expected_prices = {"AAPL": 175.32, "MSFT": 380.45}
        self.ci = {"AAPL": [168.21, 182.43], "MSFT": [372.18, 388.72]}
        self.risk = {
            "AAPL": {"var_95": -0.023, "sharpe": 1.42, "expected_return": 0.12, "volatility": 0.25},
            "MSFT": {"var_95": -0.018, "sharpe": 1.38, "expected_return": 0.10, "volatility": 0.22},
        }

    def _check(self, prompt):
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)
        self.assertIsNone(re.search(r"\bnan\b|\bNone\b", prompt))

    def test_simulation_summary(self):
        p = self.t.simulation_summary(self.tickers, self.expected_prices, self.ci, self.risk,
                                      variance_reduction=0.42)
        self._check(p)
        self.assertIn("AAPL", p)
        self.assertIn("MSFT", p)

    def test_risk_analysis(self):
        self._check(self.t.risk_analysis(self.tickers, self.risk, variance_reduction=0.42))

    def test_compare_assets(self):
        self._check(self.t.compare_assets(self.tickers, self.expected_prices, self.risk))


class TestResponseParser(unittest.TestCase):

    def setUp(self):
        self.p = ResponseParser()

    def test_clean_response(self):
        out = self.p.clean_response("  This is a test.  \n\n\nWith extra spaces.  ")
        self.assertEqual(out, out.strip())
        self.assertNotIn("\n\n\n", out)
        self.assertNotIn("  \n", out)
        self.assertIn("This is a test.", out)
        self.assertIn("With extra spaces.", out)

    def test_extract_bullet_points(self):
        text = "\n        • First point\n        • Second point with details\n        • Third point\n        "
        bullets = self.p.extract_bullet_points(text)
        self.assertEqual(len(bullets), 3)
        self.assertIn("First point", bullets[0])

    def test_extract_key_metrics(self):
        m = self.p.extract_key_metrics("The VaR is 2.3% and Sharpe ratio is 1.42. Price target $175.32.")
        self.assertIsInstance(m, dict)


class TestLlamaExplainer(unittest.TestCase):

    def test_initialization(self):
        try:
            explainer = LlamaExplainer()
        except Exception as e:  # backend (Ollama) not reachable in CI / network is blocked in tests
            self.skipTest(f"LLM backend unavailable: {e}")
        self.assertIsNotNone(explainer)


if __name__ == "__main__":
    unittest.main()