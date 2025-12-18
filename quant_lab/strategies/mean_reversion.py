"""
Mean Reversion Strategies
=========================
Statistically grounded mean-reversion strategies using Bollinger Bands
and Z-scores.
"""

from typing import Union
import numpy as np
import pandas as pd

from quant_lab.strategies.base import BaseStrategy


class BollingerMeanReversionStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy.
    
    Parameters
    ----------
    window : int, default 20
        Lookback window for rolling mean and standard deviation.
    num_std : float, default 2.0
        Band width in terms of rolling standard deviations.
    long_only : bool, default True
        Whether to disallow short positions.
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        long_only: bool = True
    ):
        super().__init__(
            name=f"BollingerMR({window},{num_std:.1f})",
            params={
                "window": window,
                "num_std": num_std,
                "long_only": long_only
            }
        )
        self.window = window
        self.num_std = num_std
        self.long_only = long_only

    def generate_signals(self, prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
        if isinstance(prices, pd.Series):
            s = prices.astype(float)
        else:
            s = pd.Series(prices, dtype=float)
            
        n = len(s)
        if n < self.window:
            return np.zeros(n)
            
        rolling_mean = s.rolling(window=self.window, min_periods=self.window).mean()
        rolling_std = s.rolling(window=self.window, min_periods=self.window).std(ddof=1)
        
        upper_band = rolling_mean + self.num_std * rolling_std
        lower_band = rolling_mean - self.num_std * rolling_std
        
        signals = np.zeros(n, dtype=float)
        current_pos = 0.0
        
        # State-based signal generation with reversion to the mean
        for i in range(self.window, n):
            p = s.iloc[i]
            mean_val = rolling_mean.iloc[i]
            lower_val = lower_band.iloc[i]
            upper_val = upper_band.iloc[i]
            
            if p < lower_val:
                current_pos = 1.0  # Oversold -> enter long
            elif p > upper_val:
                current_pos = -1.0 if not self.long_only else 0.0  # Overbought
            elif current_pos == 1.0 and p >= mean_val:
                current_pos = 0.0  # Take profit on reversion to mean
            elif current_pos == -1.0 and p <= mean_val:
                current_pos = 0.0  # Cover short on reversion to mean
                
            signals[i] = current_pos
            
        signals[:self.window] = 0.0
        return signals
