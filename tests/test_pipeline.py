"""
Unit tests for the Monte Carlo pipeline
"""

import unittest
import numpy as np
import pandas as pd
import torch
import sys
import os

# Add root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.feature_engineering import FeatureEngineer
from src.pipeline.dfmgan.generator import (
    DFMGANConfig,
    ConditionalMCGenerator,
    TimeGANDiscriminator,
    PathFilter
)
from src.pipeline.risk.metrics import RiskCalculator


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.engineer = FeatureEngineer()
        dates = pd.date_range(start='2020-01-01', periods=100)
        self.prices = pd.DataFrame({
            'AAPL': 100 + np.cumsum(np.random.randn(100)),
            'MSFT': 200 + np.cumsum(np.random.randn(100))
        }, index=dates)

    def test_feature_creation(self):
        features = self.engineer.create_features(self.prices)
        self.assertIsNotNone(features)
        self.assertGreater(len(features), 0)

    def test_sequence_creation(self):
        features = self.engineer.create_features(self.prices)
        X, y = self.engineer.create_sequences(features, seq_length=20)
        if len(X) > 0:
            self.assertEqual(len(X.shape), 3)


class TestDFMGAN(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.config = DFMGANConfig(
            n_assets=2,
            seq_len=20,
            latent_dim=8,
            hidden_dim=16,
            n_simulations=10
        )
        self.historical = torch.randn(1, 15, 2)

    def test_generator(self):
        generator = ConditionalMCGenerator(self.config)
        paths = generator(self.historical, n_paths=5)
        self.assertEqual(paths.shape[0], 5)

    def test_discriminator(self):
        discriminator = TimeGANDiscriminator(self.config)
        paths = torch.randn(3, 20, 2)
        scores = discriminator(paths)
        self.assertEqual(scores.shape, (3, 1))

    def test_path_filter(self):
        generator = ConditionalMCGenerator(self.config)
        discriminator = TimeGANDiscriminator(self.config)
        filter = PathFilter(discriminator)
        paths = generator(self.historical, n_paths=10)
        filtered, scores = filter.filter_paths(paths, top_k=0.3)
        self.assertEqual(len(filtered), 3)


class TestRiskMetrics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.calc = RiskCalculator()
        self.returns = np.random.randn(200) * 0.02

    def test_var(self):
        var = self.calc.calculate_var(self.returns, 0.95)
        self.assertIsInstance(var, float)

    def test_sharpe(self):
        sharpe = self.calc.calculate_sharpe(self.returns)
        self.assertIsInstance(sharpe, float)


if __name__ == "__main__":
    unittest.main()