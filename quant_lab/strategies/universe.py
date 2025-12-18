"""
Strategy Universe and Parameter Sweep Generator
===============================================
Generates sets of strategies (parameter sweeps or multi-strategy portfolios)
and executes them across price data to produce a validated return matrix (T x N).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from quant_lab.engine.backtest import BacktestEngine, BacktestResult
from quant_lab.engine.transaction_costs import TransactionCostModel
from quant_lab.strategies.base import BaseStrategy
from quant_lab.strategies.moving_average import DualMovingAverageStrategy
from quant_lab.strategies.mean_reversion import BollingerMeanReversionStrategy
from quant_lab.strategies.momentum import TimeSeriesMomentumStrategy


class NoiseStrategy(BaseStrategy):
    """Synthetic random trading strategy representing uninformative noise."""
    def __init__(self, seed: int, name: Optional[str] = None):
        super().__init__(name=name or f"NoiseStrategy_{seed}", params={"seed": seed})
        self.seed = seed

    def generate_signals(self, prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
        n = len(prices)
        rng = np.random.default_rng(self.seed)
        # Random positions in {-1, 0, 1}
        raw_signals = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.35, 0.30, 0.35])
        return raw_signals


class StrategyUniverse:
    """
    Manages strategy collections and runs multi-strategy backtest suites.
    """

    def __init__(self, strategies: Optional[List[BaseStrategy]] = None):
        self.strategies: List[BaseStrategy] = strategies or []

    def add(self, strategy: BaseStrategy) -> "StrategyUniverse":
        self.strategies.append(strategy)
        return self

    @classmethod
    def create_moving_average_sweep(
        cls,
        fast_windows: Optional[List[int]] = None,
        slow_windows: Optional[List[int]] = None,
        kind: str = "sma",
        long_only: bool = True
    ) -> "StrategyUniverse":
        """Generate a parameter sweep across moving average windows."""
        if fast_windows is None:
            fast_windows = [5, 10, 15, 20, 25]
        if slow_windows is None:
            slow_windows = [40, 50, 60, 75, 100]

        universe = cls()
        for f in fast_windows:
            for s in slow_windows:
                if f < s:
                    strat = DualMovingAverageStrategy(
                        fast_window=f,
                        slow_window=s,
                        kind=kind,
                        long_only=long_only
                    )
                    universe.add(strat)
        return universe

    @classmethod
    def create_mixed_universe(
        cls,
        num_signal_strategies: int = 5,
        num_noise_strategies: int = 15,
        seed: int = 42
    ) -> "StrategyUniverse":
        """
        Create a universe with a subset of legitimate strategies and
        a larger pool of noise strategies to model selection bias.
        """
        universe = cls()
        # 1. Real strategies across different paradigms
        signal_strats = [
            DualMovingAverageStrategy(10, 50, kind="ema"),
            DualMovingAverageStrategy(15, 60, kind="sma"),
            DualMovingAverageStrategy(20, 100, kind="ema"),
            BollingerMeanReversionStrategy(window=20, num_std=2.0),
            TimeSeriesMomentumStrategy(lookback=60),
            TimeSeriesMomentumStrategy(lookback=120),
            BollingerMeanReversionStrategy(window=30, num_std=1.8),
        ]
        for s in signal_strats[:num_signal_strategies]:
            universe.add(s)

        # 2. Pure noise strategies
        for i in range(num_noise_strategies):
            universe.add(NoiseStrategy(seed=seed + i, name=f"Noise_{i+1}"))

        return universe

    def run_all(
        self,
        prices: Union[pd.Series, np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None,
        cost_model: Optional[TransactionCostModel] = None,
        net: bool = True
    ) -> Tuple[np.ndarray, List[str], List[BacktestResult]]:
        """
        Execute all strategies in the universe and return the (T x N) return matrix.
        
        Parameters
        ----------
        prices : pd.Series or np.ndarray
            Price history.
        dates : pd.DatetimeIndex, optional
            Timestamp index.
        cost_model : TransactionCostModel, optional
            Trading friction parameters.
        net : bool, default True
            If True, returns matrix contains net-of-cost returns.
            If False, returns gross returns.
            
        Returns
        -------
        returns_matrix : np.ndarray, shape (T, N)
            Columns are strategies, rows are time steps.
        strategy_names : List[str]
            Names of all strategies.
        results : List[BacktestResult]
            Full backtest result objects for each strategy.
        """
        if not self.strategies:
            raise ValueError("Strategy universe is empty.")

        engine = BacktestEngine(cost_model=cost_model)
        names = []
        results = []
        columns = []

        for strat in self.strategies:
            signals = strat.generate_signals(prices)
            res = engine.run(
                prices=prices,
                signals=signals,
                dates=dates,
                strategy_name=strat.name
            )
            names.append(strat.name)
            results.append(res)
            columns.append(res.net_returns if net else res.gross_returns)

        returns_matrix = np.column_stack(columns)
        return returns_matrix, names, results
