"""
Transaction Cost Modelling Module
=================================
Models realistic institutional and retail friction costs:
- Proportional commission (basis points / percentage of traded value)
- Bid-ask spread costs (half-spread incurred per turnover)
- Fixed ticket fee per trade ($ per execution)
- Nonlinear market impact / slippage (proportional to trade size or square-root law)
"""

from dataclasses import dataclass
from typing import Union
import numpy as np
import pandas as pd


@dataclass
class TransactionCostModel:
    """
    Configurable transaction friction model.
    
    Parameters
    ----------
    commission_bps : float, default 5.0
        Brokerage commission in basis points (1 bp = 0.0001 = 0.01%).
    half_spread_bps : float, default 2.0
        Half the bid-ask spread in basis points.
    slippage_bps : float, default 3.0
        Baseline execution slippage in basis points.
    fixed_fee_per_trade : float, default 0.0
        Fixed flat ticket fee in currency units (e.g. $1.00).
    market_impact_coeff : float, default 0.0
        Quadratic/square-root market impact multiplier.
    """
    commission_bps: float = 5.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 3.0
    fixed_fee_per_trade: float = 0.0
    market_impact_coeff: float = 0.0

    @property
    def total_linear_bps(self) -> float:
        """Total linear basis points deducted per unit of portfolio turnover."""
        return self.commission_bps + self.half_spread_bps + self.slippage_bps

    @property
    def total_linear_rate(self) -> float:
        """Total linear friction as a decimal rate (e.g. 10 bps -> 0.0010)."""
        return self.total_linear_bps * 1e-4

    def calculate_cost(
        self,
        turnover: Union[float, np.ndarray, pd.Series],
        portfolio_equity: float = 100_000.0
    ) -> Union[float, np.ndarray]:
        """
        Compute total cost as a fraction of portfolio equity.
        
        Parameters
        ----------
        turnover : float or array-like
            Fraction of portfolio rebalanced (|w_t - w_{t-1}|).
        portfolio_equity : float, default 100_000.0
            Nominal portfolio equity for fixed fee calculation.
        """
        turnover_arr = np.asarray(turnover, dtype=float)
        
        # Linear rate: commission + spread + linear slippage
        linear_costs = self.total_linear_rate * turnover_arr
        
        # Nonlinear market impact: coeff * turnover^1.5 or turnover^2
        if self.market_impact_coeff > 0:
            impact_costs = self.market_impact_coeff * (turnover_arr ** 1.5)
        else:
            impact_costs = 0.0
            
        # Fixed fee per trade converted to return drag
        if self.fixed_fee_per_trade > 0 and portfolio_equity > 0:
            trade_occurred = (turnover_arr > 1e-6).astype(float)
            fixed_costs = trade_occurred * (self.fixed_fee_per_trade / portfolio_equity)
        else:
            fixed_costs = 0.0
            
        total = linear_costs + impact_costs + fixed_costs
        if np.isscalar(turnover):
            return float(total)
        return total


def apply_transaction_costs(
    gross_returns: np.ndarray,
    positions: np.ndarray,
    cost_model: TransactionCostModel,
    initial_equity: float = 100_000.0
) -> np.ndarray:
    """
    Deduct realistic transaction costs from a gross return series based on position changes.
    
    Parameters
    ----------
    gross_returns : np.ndarray
        Gross strategy returns before costs.
    positions : np.ndarray
        Target portfolio weight/position at each bar.
    cost_model : TransactionCostModel
        Friction parameters.
    initial_equity : float
        Starting capital.
        
    Returns
    -------
    net_returns : np.ndarray
        Returns net of all trading friction.
    """
    gross = np.asarray(gross_returns, dtype=float)
    pos = np.asarray(positions, dtype=float)
    
    # Turnover on each bar: change in position from previous bar
    prev_pos = np.roll(pos, 1)
    prev_pos[0] = 0.0  # Initial allocation cost from cash
    turnover = np.abs(pos - prev_pos)
    
    # Calculate costs
    cost_drag = cost_model.calculate_cost(turnover, portfolio_equity=initial_equity)
    net_returns = gross - cost_drag
    return net_returns
