"""
Strategies Subpackage
"""

from quant_lab.strategies.base import BaseStrategy
from quant_lab.strategies.moving_average import DualMovingAverageStrategy
from quant_lab.strategies.mean_reversion import BollingerMeanReversionStrategy
from quant_lab.strategies.momentum import TimeSeriesMomentumStrategy
from quant_lab.strategies.universe import (
    NoiseStrategy,
    StrategyUniverse,
)

__all__ = [
    "BaseStrategy",
    "DualMovingAverageStrategy",
    "BollingerMeanReversionStrategy",
    "TimeSeriesMomentumStrategy",
    "NoiseStrategy",
    "StrategyUniverse",
]
