"""
Time Series Momentum Strategy
=============================
Classic absolute momentum strategy that generates signals based on
past cumulative return signs.
"""

from typing import Union
import numpy as np
import pandas as pd

from quant_lab.strategies.base import BaseStrategy


class TimeSeriesMomentumStrategy(BaseStrategy):
    """
    Time Series Momentum (Trend Following) Strategy.
    
    Parameters
    ----------
    lookback : int, default 60
        Lookback horizon in trading bars.
    long_only : bool, default True
        If True, only takes long positions; otherwise long/short.
    """

    def __init__(self, lookback: int = 60, long_only: bool = True):
        super().__init__(
            name=f"TSMomentum({lookback})",
            params={"lookback": lookback, "long_only": long_only}
        )
        self.lookback = lookback
        self.long_only = long_only

    def generate_signals(self, prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(prices, pd.Series):
            s = prices.astype(float)
        else:
            s = pd.Series(prices, dtype=float)
            
        n = len(s)
        if n <= self.lookback:
            return np.zeros(n)
            
        # Cumulative return over lookback
        past_returns = (s / s.shift(self.lookback) - 1.0).values
        
        signals = np.zeros(n, dtype=float)
        bullish = past_returns > 0
        bearish = past_returns < 0
        
        signals[bullish] = 1.0
        if not self.long_only:
            signals[bearish] = -1.0
            
        signals[:self.lookback] = 0.0
        return signals
