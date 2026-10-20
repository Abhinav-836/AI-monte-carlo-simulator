"""
Integration tests: pipeline end-to-end on mock data + DataFetcher behaviour with mocked providers.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline import MonteCarloPipeline
from src.pipeline.data_fetcher import DataFetcher

TICKERS = ["AAPL", "MSFT", "GOOGL"]
LEVELS = {"AAPL": 50.0, "MSFT": 200.0, "GOOGL": 1000.0}  # very different scales on purpose


class MockFetcher:
    def __init__(self, missing=(), empty=False):
        self.missing = set(missing)
        self.empty = empty
        self.last_errors = {}

    def get_historical_data(self, tickers_tuple, period="1y"):
        if self.empty:
            return None
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2023-01-02", periods=150)
        cols = {t: LEVELS[t] * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
                for t in tickers_tuple if t not in self.missing}
        return pd.DataFrame(cols, index=idx)

    def get_current_prices(self, tickers, force_refresh=False):
        df = self.get_historical_data(tuple(TICKERS))
        return {t: float(df[t].iloc[-1]) for t in tickers if t in df.columns}

    def get_data_source_info(self):
        return "Mock Data"


def make_pipeline(fetcher):
    p = MonteCarloPipeline(n_assets=3, n_simulations=200, filter_top_k=0.1, use_live_data=False)
    p.data_fetcher = fetcher
    return p


class TestPipelineIntegration(unittest.TestCase):

    def test_pipeline_execution(self):
        r = make_pipeline(MockFetcher()).run_simulation(TICKERS, period="1y")
        self.assertEqual(r["metadata"]["tickers"], TICKERS)
        self.assertEqual(r["metadata"]["missing_tickers"], [])
        self.assertEqual(r["metadata"]["filtered_paths"], 20)
        self.assertEqual(set(r["expected_prices"]), set(TICKERS))
        self.assertEqual(len(r["price_distributions"]["AAPL"]), 200)  # stats use ALL paths

    def test_risk_metrics(self):
        r = make_pipeline(MockFetcher()).run_simulation(TICKERS)
        self.assertEqual(len(r["risk_metrics"]), 3)
        for t, m in r["risk_metrics"].items():
            for key in ("var_95", "cvar_95", "expected_return", "sharpe", "sortino", "max_drawdown"):
                self.assertIn(key, m)
                self.assertTrue(np.isfinite(m[key]), f"{t}.{key}")
            self.assertLessEqual(m["cvar_95"], m["var_95"])
            self.assertLessEqual(m["max_drawdown"], 0.0)

    def test_expected_price_near_own_last_price(self):
        fetcher = MockFetcher()
        r = make_pipeline(fetcher).run_simulation(TICKERS)
        hist = fetcher.get_historical_data(tuple(TICKERS))
        for t in TICKERS:
            ratio = r["expected_prices"][t] / hist[t].iloc[-1]
            self.assertLess(abs(ratio - 1), 0.3, f"{t} expected price is off its own scale")

    def test_missing_ticker_does_not_shift_other_tickers(self):
        # regression: a failed ticker removed a column, and positional indexing then gave
        # GOOGL the MSFT paths (prices ~5x too low)
        fetcher = MockFetcher(missing=["MSFT"])
        r = make_pipeline(fetcher).run_simulation(TICKERS)
        self.assertEqual(r["metadata"]["missing_tickers"], ["MSFT"])
        self.assertEqual(set(r["expected_prices"]), {"AAPL", "GOOGL"})
        self.assertEqual(set(r["risk_metrics"]), {"AAPL", "GOOGL"})
        hist = fetcher.get_historical_data(tuple(TICKERS))
        self.assertLess(abs(r["expected_prices"]["GOOGL"] / hist["GOOGL"].iloc[-1] - 1), 0.3)

    def test_no_data_raises(self):
        with self.assertRaises(ValueError):
            make_pipeline(MockFetcher(empty=True)).run_simulation(TICKERS)


class TestDataFetcher(unittest.TestCase):

    def setUp(self):
        self.f = DataFetcher()  # conftest removes API keys and blocks the network

    def test_twelve_symbol_conversion(self):
        self.assertEqual(DataFetcher._twelve_params("RELIANCE.NS"), {"symbol": "RELIANCE", "exchange": "NSE"})
        self.assertEqual(DataFetcher._twelve_params("AAPL"), {"symbol": "AAPL"})
        self.assertEqual(DataFetcher._twelve_params("BTC-USD"), {"symbol": "BTC/USD"})

    def test_alpha_symbol_and_period(self):
        self.assertEqual(DataFetcher._alpha_symbol("RELIANCE.NS"), "RELIANCE.BSE")
        self.assertEqual(DataFetcher._period_to_days("2y"), 504)
        self.assertEqual(DataFetcher._period_to_days("6mo"), 126)
        self.assertEqual(DataFetcher._period_to_days("30d"), 30)

    def test_clean_series_removes_tz_and_duplicates(self):
        idx = pd.DatetimeIndex(["2024-01-02 09:30", "2024-01-02 15:30", "2024-01-03 09:30"],
                               tz="America/New_York")
        s = DataFetcher._clean_series(pd.Series([10.0, 11.0, 12.0], index=idx), "X")
        self.assertEqual(len(s), 2)
        self.assertEqual(s.iloc[0], 11.0)
        self.assertIsNone(s.index.tz)

    def test_rate_limiter_blocks_after_limit(self):
        self.f.TD_PER_MINUTE = 2
        self.assertTrue(self.f._acquire_twelve_slot(max_wait=0.05))
        self.assertTrue(self.f._acquire_twelve_slot(max_wait=0.05))
        self.assertFalse(self.f._acquire_twelve_slot(max_wait=0.05))

    def test_unsupported_symbol_is_not_retried(self):
        self.f.twelve_data_key = "dummy"
        resp = mock.MagicMock()
        resp.json.return_value = {"code": 404, "message": "symbol not found"}
        with mock.patch("src.pipeline.data_fetcher.requests.get", return_value=resp) as get:
            self.assertIsNone(self.f._call_twelve_data("FOO.NS"))
            self.assertIsNone(self.f._call_twelve_data("FOO.NS"))
        self.assertEqual(get.call_count, 1)
        self.assertIn("FOO.NS", self.f._td_unsupported)

    def test_price_fallback_chain(self):
        with mock.patch.object(self.f, "_call_twelve_data", return_value=None), \
             mock.patch.object(self.f, "_call_finnhub", return_value=None), \
             mock.patch.object(self.f, "_call_yfinance", return_value=123.0):
            prices = self.f.get_current_prices(["X"])
        self.assertEqual(prices, {"X": 123.0})
        self.assertEqual(self.f.price_sources["X"], "yfinance")

    def test_stale_price_used_when_all_providers_fail(self):
        with mock.patch.object(self.f, "_call_twelve_data", return_value=50.0):
            self.f.get_current_prices(["X"])
        stamp = self.f.cache_timestamp["X"]
        none = lambda *_: None
        with mock.patch.object(self.f, "_call_twelve_data", none), mock.patch.object(self.f, "_call_finnhub", none), \
             mock.patch.object(self.f, "_call_yfinance", none), mock.patch.object(self.f, "_call_yahoo_chart", none), \
             mock.patch.object(self.f, "_call_alpha_vantage", none):
            prices = self.f.get_current_prices(["X"], force_refresh=True)
        self.assertEqual(prices["X"], 50.0)
        self.assertEqual(self.f.price_sources["X"], "Cached (stale)")
        self.assertEqual(self.f.cache_timestamp["X"], stamp)  # stays stale -> retried next time

    def test_partial_history_expires_fast_and_reports_failures(self):
        series = pd.Series(100 + np.arange(60.0), index=pd.bdate_range("2024-01-01", periods=60), name="AAA")
        with mock.patch.object(self.f, "_fetch_history_one",
                               side_effect=lambda t, d: series if t == "AAA" else None):
            df = self.f.get_historical_data(("AAA", "BBB"), "1y")
        self.assertEqual(list(df.columns), ["AAA"])
        self.assertEqual(self.f.failed_tickers, ["BBB"])
        key = f"{('AAA', 'BBB')}_1y"
        self.assertEqual(self.f.historical_ttl[key], self.f.partial_cache_duration)


if __name__ == "__main__":
    unittest.main()