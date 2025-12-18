"""
Backtesting Engine Module
=========================
Reusable quantitative backtesting engine featuring:
- Strict protection against look-ahead bias (signals generated at t executed at t+1)
- Comprehensive transaction cost and slippage modelling
- Accurate portfolio accounting and drawdown tracking
- Comprehensive performance diagnostics
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

from quant_lab.engine.metrics import calculate_all_metrics
from quant_lab.engine.transaction_costs import TransactionCostModel


@dataclass
class BacktestResult:
    """Encapsulates backtest execution output and analytics."""
    strategy_name: str
    dates: pd.DatetimeIndex
    prices: np.ndarray
    signals: np.ndarray
    positions: np.ndarray
    turnover: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    gross_equity: np.ndarray
    net_equity: np.ndarray
    gross_metrics: Dict[str, float]
    net_metrics: Dict[str, float]
    cost_drag_total: float
    total_trades: int
    cost_model: TransactionCostModel

    def summary(self) -> pd.DataFrame:
        """Return a side-by-side comparison of Gross vs Net performance."""
        keys = [
            "annualized_return_cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "profit_factor"
        ]
        labels = [
            "CAGR Return",
            "Annual Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Calmar Ratio",
            "Win Rate",
            "Profit Factor"
        ]
        data = {
            "Metric": labels,
            "Gross (Pre-Cost)": [self.gross_metrics.get(k, 0.0) for k in keys],
            "Net (Post-Cost)": [self.net_metrics.get(k, 0.0) for k in keys],
        }
        df = pd.DataFrame(data)
        # Format percentage columns
        pct_indices = [0, 1, 4, 6]
        for col in ["Gross (Pre-Cost)", "Net (Post-Cost)"]:
            formatted = []
            for idx, val in enumerate(df[col]):
                if idx in pct_indices:
                    formatted.append(f"{val:.2%}")
                else:
                    formatted.append(f"{val:.3f}")
            df[col] = formatted
        return df


class BacktestEngine:
    """
    Core vector and bar-level backtesting engine.
    
    Guarantees look-ahead safety by shifting signals by 1 period:
        position[t] = signal[t - 1]
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        initial_equity: float = 100_000.0,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        self.cost_model = cost_model or TransactionCostModel()
        self.initial_equity = initial_equity
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def run(
        self,
        prices: Union[pd.Series, np.ndarray],
        signals: Union[pd.Series, np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None,
        strategy_name: str = "Quantitative Strategy"
    ) -> BacktestResult:
        """
        Execute backtest with strict execution lag and cost deduction.
        
        Parameters
        ----------
        prices : pd.Series or np.ndarray
            Asset close prices or values.
        signals : pd.Series or np.ndarray
            Strategy target position signal generated at close of each bar (e.g. -1, 0, +1).
        dates : pd.DatetimeIndex, optional
            Timestamp index for bars.
        strategy_name : str
            Identifier for the strategy.
        """
        # Validate and align inputs
        if isinstance(prices, pd.Series):
            if dates is None:
                dates = prices.index
            prices_arr = prices.values.astype(float)
        else:
            prices_arr = np.asarray(prices, dtype=float)
            
        if isinstance(signals, pd.Series):
            signals_arr = signals.values.astype(float)
        else:
            signals_arr = np.asarray(signals, dtype=float)
            
        n = len(prices_arr)
        if len(signals_arr) != n:
            raise ValueError(f"Prices length ({n}) != signals length ({len(signals_arr)})")
        if n < 3:
            raise ValueError("Need at least 3 price bars to backtest.")
            
        if dates is None:
            dates = pd.date_range("2020-01-01", periods=n, freq="B")
            
        # 1. Underlying market returns: r_asset[t] = P[t] / P[t-1] - 1
        asset_returns = np.zeros(n)
        asset_returns[1:] = (prices_arr[1:] / prices_arr[:-1]) - 1.0
        
        # 2. Strict look-ahead protection:
        # Signal generated at bar t is executed at bar t+1.
        # Position held during period t is based on signal at t-1.
        positions = np.zeros(n)
        positions[1:] = signals_arr[:-1]  # 1-bar lag
        
        # 3. Strategy gross returns: position[t] * asset_returns[t]
        gross_returns = positions * asset_returns
        
        # 4. Turnover calculation: |pos[t] - pos[t-1]|
        prev_positions = np.zeros(n)
        prev_positions[1:] = positions[:-1]
        turnover = np.abs(positions - prev_positions)
        
        # 5. Cost deduction per bar
        cost_drag = self.cost_model.calculate_cost(turnover, portfolio_equity=self.initial_equity)
        net_returns = gross_returns - cost_drag
        
        # 6. Cumulative equity series (wealth index starting at 1.0)
        gross_equity = np.cumprod(1.0 + gross_returns)
        net_equity = np.cumprod(1.0 + net_returns)
        
        # 7. Metrics evaluation
        gross_metrics = calculate_all_metrics(
            gross_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year
        )
        net_metrics = calculate_all_metrics(
            net_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year
        )
        
        total_trades = int(np.sum(turnover > 1e-6))
        total_drag = float(np.sum(cost_drag))
        
        return BacktestResult(
            strategy_name=strategy_name,
            dates=dates,
            prices=prices_arr,
            signals=signals_arr,
            positions=positions,
            turnover=turnover,
            gross_returns=gross_returns,
            net_returns=net_returns,
            gross_equity=gross_equity,
            net_equity=net_equity,
            gross_metrics=gross_metrics,
            net_metrics=net_metrics,
            cost_drag_total=total_drag,
            total_trades=total_trades,
            cost_model=self.cost_model
        )
