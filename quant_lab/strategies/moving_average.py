"""
Dual Moving Average Strategy
============================
Trend following strategy that generates long/short/flat signals based on
moving average crossovers (SMA or EMA).
"""

from typing import Union
import numpy as np
import pandas as pd

from quant_lab.strategies.base import BaseStrategy


class DualMovingAverageStrategy(BaseStrategy):
    """
    Dual Moving Average Crossover Strategy.
    
    Parameters
    ----------
    fast_window : int
        Lookback period for fast moving average.
    slow_window : int
        Lookback period for slow moving average.
    kind : str, default 'sma'
        'sma' (simple moving average) or 'ema' (exponential moving average).
    long_only : bool, default True
        If True, signals are in {0, 1}. If False, signals are in {-1, 1}.
    """

    def __init__(
        self,
        fast_window: int = 10,
        slow_window: int = 50,
        kind: str = "sma",
        long_only: bool = True
    ):
        if fast_window >= slow_window:
            raise ValueError(f"fast_window ({fast_window}) must be < slow_window ({slow_window})")
        super().__init__(
            name=f"DualMA({fast_window},{slow_window},{kind})",
            params={
                "fast_window": fast_window,
                "slow_window": slow_window,
                "kind": kind,
                "long_only": long_only
            }
        )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.kind = kind.lower()
        self.long_only = long_only

    def generate_signals(self, prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(prices, pd.Series):
            s = prices.astype(float)
        else:
            s = pd.Series(prices, dtype=float)
            
        n = len(s)
        if n < self.slow_window:
            return np.zeros(n)
            
        if self.kind == "ema":
            fast_ma = s.ewm(span=self.fast_window, adjust=False).mean()
            slow_ma = s.ewm(span=self.slow_window, adjust=False).mean()
        else:
            fast_ma = s.rolling(window=self.fast_window, min_periods=self.fast_window).mean()
            slow_ma = s.rolling(window=self.slow_window, min_periods=self.slow_window).mean()
            
        signals = np.zeros(n, dtype=float)
        
        # Bullish condition: fast > slow
        bullish = (fast_ma > slow_ma).values
        # Bearish condition: fast < slow
        bearish = (fast_ma < slow_ma).values
        
        signals[bullish] = 1.0
        if not self.long_only:
            signals[bearish] = -1.0
            
        # Warmup period has 0 signal
        signals[:self.slow_window] = 0.0
        return signals
