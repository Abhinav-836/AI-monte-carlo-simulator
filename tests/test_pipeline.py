"""
Unit tests for the Monte Carlo pipeline components.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.feature_engineering import FeatureEngineer
from src.pipeline.dfmgan.generator import (
    DFMGANConfig, ConditionalMCGenerator, TimeGANDiscriminator, PathFilter)
from src.pipeline.dfmgan.trainer import DFMGANTrainer
from src.pipeline.risk.metrics import RiskCalculator
from src.pipeline.pemc.estimator import PEMCConfig, NeuralPredictor, PEMCEstimator
from src.pipeline.pemc.trainer import PEMCTrainer
from src.pipeline.pemc.option_models import OptionPricer, EuropeanOptionPricer, create_sampler
from src.pipeline.pemc.variance_analyzer import VarianceAnalyzer
from src.pipeline.portfolio_optimizer import PortfolioOptimizer


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2020-01-01", periods=120)
        self.prices = pd.DataFrame({
            "AAPL": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))),
            "MSFT": 200 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))),
        }, index=dates)
        self.engineer = FeatureEngineer()

    def test_features_created_and_finite(self):
        f = self.engineer.create_features(self.prices)
        self.assertGreater(len(f), 0)
        self.assertTrue(np.isfinite(f.values).all())

    def test_bounded_indicators_are_not_clipped(self):
        # regression: the old clip(-10, 10) flattened RSI / stochastic / etc.
        f = self.engineer.create_features(self.prices)
        self.assertGreater(f["AAPL_rsi"].abs().max(), 10)

    def test_no_lookahead(self):
        # features on a prefix must equal the same rows computed with future data present
        a = FeatureEngineer().create_features(self.prices.iloc[:90])
        b = FeatureEngineer().create_features(self.prices)
        cols = [c for c in a.columns if not c.endswith("_interact") and c in b.columns]
        self.assertGreater(len(cols), 100)
        pd.testing.assert_frame_equal(a[cols], b[cols].loc[a.index], rtol=1e-6, atol=1e-9)

    def test_sequences(self):
        f = self.engineer.create_features(self.prices)
        X, y = self.engineer.create_sequences(f, seq_length=20)
        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.shape[1], 20)
        self.assertEqual(len(X), len(y))

    def test_normalize_clips(self):
        f = self.engineer.create_features(self.prices)
        n = self.engineer.normalize_features(f, fit=True)
        self.assertLessEqual(float(n.abs().max().max()), 5.0)


class TestDFMGAN(unittest.TestCase):

    def setUp(self):
        self.config = DFMGANConfig(n_assets=2, seq_len=20, latent_dim=8, hidden_dim=16, n_simulations=10)
        self.historical = torch.randn(1, 15, 2).abs() + 1.0

    def test_generator_shape(self):
        paths = ConditionalMCGenerator(self.config)(self.historical, n_paths=5)
        self.assertEqual(tuple(paths.shape), (5, 20, 2))

    def test_layers_exist_at_init(self):
        # regression: layers used to be built lazily AFTER the optimizer, so they never trained
        names = [n for n, _ in ConditionalMCGenerator(self.config).named_parameters()]
        self.assertTrue(any("input_proj" in n for n in names))
        self.assertTrue(any("output_proj" in n for n in names))

    def test_wrong_asset_count_raises(self):
        with self.assertRaises(ValueError):
            ConditionalMCGenerator(self.config)(torch.randn(1, 15, 3))

    def test_critic_output_shape_and_no_sigmoid(self):
        critic = TimeGANDiscriminator(self.config)
        self.assertEqual(tuple(critic(torch.rand(3, 20, 2) + 1).shape), (3, 1))
        self.assertFalse(any(isinstance(m, nn.Sigmoid) for m in critic.modules()))

    def test_path_filter(self):
        gen, critic = ConditionalMCGenerator(self.config), TimeGANDiscriminator(self.config)
        paths = gen(self.historical, n_paths=10)
        filtered, scores = PathFilter(critic).filter_paths(paths, top_k=0.3)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(len(scores), 10)

    def test_path_filter_batch_of_one(self):
        # regression: squeeze() on a batch of 1 crashed torch.cat
        gen, critic = ConditionalMCGenerator(self.config), TimeGANDiscriminator(self.config)
        paths = gen(self.historical, n_paths=1)
        filtered, _ = PathFilter(critic).filter_paths(paths, top_k=1.0)
        self.assertEqual(len(filtered), 1)

    def test_trainer_updates_generator(self):
        gen, critic = ConditionalMCGenerator(self.config), TimeGANDiscriminator(self.config)
        before = gen.input_proj.weight.detach().clone()
        real = torch.rand(16, 20, 2) + 1.0
        hist = DFMGANTrainer(gen, critic, self.config).train(real, epochs=1, batch_size=8, n_critic=1)
        self.assertTrue(np.isfinite(hist["d_loss"]).all() and np.isfinite(hist["g_loss"]).all())
        self.assertFalse(torch.equal(before, gen.input_proj.weight))


class TestRiskMetrics(unittest.TestCase):

    def setUp(self):
        self.calc = RiskCalculator()
        self.returns = np.random.randn(200) * 0.02

    def test_var_and_sharpe(self):
        self.assertIsInstance(self.calc.calculate_var(self.returns, 0.95), float)
        self.assertIsInstance(self.calc.calculate_sharpe(self.returns), float)

    def test_sortino_uses_all_observations(self):
        r = np.array([0.01] * 9 + [-0.01])
        downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252)
        expected = (r.mean() * 252 - 0.02) / downside
        self.assertAlmostEqual(self.calc.calculate_sortino(r), expected, places=6)

    def test_calculate_all_metrics_works(self):
        # regression: was a @staticmethod taking `self`, so it crashed on every call
        paths = np.random.randn(50, 30, 2) * 0.01
        m = RiskCalculator.calculate_all_metrics(paths, ["A", "B"])
        self.assertEqual(set(m), {"A", "B"})
        self.assertLessEqual(m["A"]["max_drawdown"], 0.0)
        self.assertLessEqual(m["A"]["worst_drawdown"], m["A"]["max_drawdown"])
        self.assertLessEqual(m["A"]["cvar_95"], m["A"]["var_95"])

    def test_capture_aligns_on_recent_observations(self):
        b = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        r_long = np.concatenate([[9.0] * 5, b])  # old junk must be ignored
        self.assertAlmostEqual(self.calc.calculate_upside_capture(r_long, b), 1.0, places=6)
        self.assertAlmostEqual(self.calc.calculate_downside_capture(r_long, b), 1.0, places=6)


class TestPEMC(unittest.TestCase):

    def test_black_scholes_reference(self):
        self.assertAlmostEqual(OptionPricer.black_scholes(100, 100, 1.0, 0.05, 0.2), 10.4506, places=3)

    def test_european_mc_matches_black_scholes(self):
        bs = OptionPricer.black_scholes(100, 105, 1.0, 0.05, 0.2)
        pay, _ = EuropeanOptionPricer().payoff(20000, 100, 105, 1.0, 0.05, 0.2, n_steps=50)
        se = pay.std(ddof=1) / np.sqrt(len(pay))
        self.assertLess(abs(pay.mean() - bs), 4 * se)

    def test_sampler_shapes(self):
        # regression: 'european' used to crash (no .payoff method)
        dims = {"european": 7, "asian": 7, "barrier": 7, "variance_swap": 5}
        for name, dim in dims.items():
            with self.subTest(option=name):
                pay, feats = create_sampler(name)(200)
                self.assertEqual(pay.shape, (200,))
                self.assertEqual(feats.shape, (200, dim))
                self.assertTrue(np.isfinite(feats).all())

    def test_trainer_learns_signal(self):
        # regression: (B,) predictions vs (B,1) targets broadcast to (B,B) and the net learned only the mean
        X = np.random.randn(2000, 2).astype(np.float32)
        y = (2 * X[:, 0] + X[:, 1]).astype(np.float32)
        predictor = NeuralPredictor(input_dim=2)
        PEMCTrainer(predictor, PEMCConfig(), device="cpu").train(
            X, y, epochs=40, batch_size=64, early_stopping_patience=15)
        with torch.no_grad():
            pred = predictor(torch.as_tensor(X)).numpy()
        self.assertGreater(np.corrcoef(pred, y)[0, 1], 0.9)

    def test_trainer_rejects_swapped_arguments(self):
        X = np.random.randn(100, 2).astype(np.float32)
        y = X[:, 0]
        with self.assertRaises(ValueError):
            PEMCTrainer(NeuralPredictor(input_dim=2), PEMCConfig(), device="cpu").train(y, X, epochs=1)

    def test_estimator_unbiased_and_reduces_variance(self):
        def expensive(n):  # returns (payoffs, features) — the order PEMCEstimator expects
            x = np.random.randn(n, 2).astype(np.float32)
            return x[:, 0] + 0.1 * np.random.randn(n), x

        def cheap(n):
            return None, np.random.randn(n, 2).astype(np.float32)

        y, X = expensive(4000)  # FIX: was `X, y = ...`, which swapped payoffs and features
        predictor = NeuralPredictor(input_dim=2)
        PEMCTrainer(predictor, PEMCConfig(), device="cpu").train(X, y, epochs=40, batch_size=64)

        est = PEMCEstimator(predictor, PEMCConfig(n_coupled=2000, n_independent=20000, sampler_chunk=5000))
        value, var, m = est.estimate(expensive, cheap)
        self.assertLess(abs(value), 5 * np.sqrt(var) + 0.05)  # true mean is 0
        self.assertGreater(m["variance_reduction"], 0.5)
        self.assertEqual(m["samples_used"]["independent"], 20000)

    def test_theoretical_variance_reduction(self):
        va = VarianceAnalyzer()
        self.assertAlmostEqual(va.theoretical_variance_reduction(0.9, 1000, 100000), 0.8019, places=3)
        self.assertEqual(va.theoretical_variance_reduction(0.0, 1000, 100000), 0.0)


class TestPortfolioOptimizer(unittest.TestCase):

    def test_min_vol_two_assets(self):
        cov = np.diag([0.01, 0.04])
        w = PortfolioOptimizer(np.array([0.1, 0.1]), cov).optimize_min_volatility()["weights"]
        self.assertAlmostEqual(sum(w), 1.0, places=6)
        self.assertAlmostEqual(w[0], 0.8, delta=0.01)  # inverse-variance weights

    def test_risk_parity_inverse_vol(self):
        cov = np.diag([0.01, 0.04, 0.09])
        w = np.array(PortfolioOptimizer(np.full(3, 0.1), cov).optimize_risk_parity()["weights"])
        expected = (1 / np.sqrt(np.diag(cov)))
        expected /= expected.sum()
        np.testing.assert_allclose(w, expected, atol=0.02)

    def test_weights_valid_for_all_methods(self):
        mu = np.array([0.08, 0.12, 0.10])
        cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.06]])
        for method in ("max_sharpe", "min_vol", "max_return", "risk_parity", "max_diversification"):
            with self.subTest(method=method):
                res = PortfolioOptimizer.optimize_portfolio(mu, cov, method=method)
                w = np.array(res["weights"])
                self.assertAlmostEqual(w.sum(), 1.0, places=5)
                self.assertTrue((w >= -1e-9).all())


if __name__ == "__main__":
    unittest.main()