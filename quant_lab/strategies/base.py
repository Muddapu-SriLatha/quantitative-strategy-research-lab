"""
Base Strategy Interface
=======================
Defines the abstract interface for all quantitative strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """
    Abstract base class for quantitative trading strategies.
    
    A strategy must implement `generate_signals` which transforms
    historical price data into a target position signal in [-1, +1].
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Generate target position signal for each bar.
        
        Parameters
        ----------
        prices : pd.Series or np.ndarray
            Price series up to the current bar.
            
        Returns
        -------
        signals : np.ndarray
            Target position signal (-1 = short, 0 = flat, +1 = long).
        """
        pass

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}(name='{self.name}', {param_str})"
