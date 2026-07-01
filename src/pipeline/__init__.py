"""
Pipeline module for Monte Carlo simulations
"""

from src.pipeline.data_fetcher import DataFetcher
from src.pipeline.feature_engineering import FeatureEngineer, FeatureConfig
from src.pipeline.risk.metrics import RiskCalculator
from src.pipeline.risk.confidence import ConfidenceIntervalCalculator
from src.pipeline.monte_carlo_pipeline import MonteCarloPipeline
from src.pipeline.portfolio_optimizer import PortfolioOptimizer, FactorModel

# Try to import optional modules
try:
    from src.pipeline.dfmgan.generator import DFMGANConfig, ConditionalMCGenerator, TimeGANDiscriminator, PathFilter
    from src.pipeline.dfmgan.trainer import DFMGANTrainer
    DFMGAN_AVAILABLE = True
except ImportError:
    DFMGAN_AVAILABLE = False

try:
    from src.pipeline.pemc.estimator import PEMCConfig, NeuralPredictor, PEMCEstimator
    from src.pipeline.pemc.trainer import PEMCTrainer
    from src.pipeline.pemc.option_models import OptionPricer, AsianOptionPricer, BarrierOptionPricer, VarianceSwapPricer
    from src.pipeline.pemc.variance_analyzer import VarianceAnalyzer
    PEMC_AVAILABLE = True
except ImportError:
    PEMC_AVAILABLE = False

__all__ = [
    'DataFetcher',
    'FeatureEngineer',
    'FeatureConfig',
    'RiskCalculator',
    'ConfidenceIntervalCalculator',
    'MonteCarloPipeline',
    'PortfolioOptimizer',
    'FactorModel'
]

# Add optional modules to __all__ if available
if DFMGAN_AVAILABLE:
    __all__.extend([
        'DFMGANConfig',
        'ConditionalMCGenerator',
        'TimeGANDiscriminator',
        'PathFilter',
        'DFMGANTrainer'
    ])

if PEMC_AVAILABLE:
    __all__.extend([
        'PEMCConfig',
        'NeuralPredictor',
        'PEMCEstimator',
        'PEMCTrainer',
        'OptionPricer',
        'AsianOptionPricer',
        'BarrierOptionPricer',
        'VarianceSwapPricer',
        'VarianceAnalyzer'
    ])