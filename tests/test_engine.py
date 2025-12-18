"""
Unit Tests for Backtesting Engine and Friction Model
===================================================
Tests look-ahead bias prevention and transaction cost accounting.
"""

import numpy as np
import pandas as pd
import pytest

from quant_lab.engine.backtest import BacktestEngine
from quant_lab.engine.transaction_costs import TransactionCostModel


def test_look_ahead_bias_protection():
    """
    Critical verification that signal generated at bar t does NOT capture
    bar t's return, preventing look-ahead leakage.
    """
    # Price path: 100, 100, 100, 150 (huge +50% jump at index 3), 150, 150
    prices = np.array([100.0, 100.0, 100.0, 150.0, 150.0, 150.0])
    n = len(prices)
    
    # Strategy that cheats by looking at current price and trying to go long at index 3:
    # signals = [0, 0, 0, 1, 1, 1]
    signals = np.zeros(n)
    signals[3:] = 1.0  # Go long starting when price hits 150
    
    # Run backtest with zero transaction costs
    engine = BacktestEngine(cost_model=TransactionCostModel(0, 0, 0, 0))
    res = engine.run(prices=prices, signals=signals)
    
    # Asset returns:
    # t=0: 0.0
    # t=1: 0.0
    # t=2: 0.0
    # t=3: 150/100 - 1 = +0.50 (The jump!)
    # t=4: 150/150 - 1 = 0.0
    # t=5: 150/150 - 1 = 0.0
    
    # Due to 1-period execution lag:
    # position at t=3 MUST BE 0.0 (signal was generated at close of t=3)
    # position at t=4 is 1.0
    assert res.positions[3] == 0.0, "Position at jump bar must be 0 to avoid look-ahead bias!"
    assert res.positions[4] == 1.0, "Position at post-jump bar must be 1.0!"
    
    # Gross return at t=3 must be 0.0 (the strategy did NOT capture the jump retroactively)
    assert res.gross_returns[3] == 0.0
    assert res.gross_equity[-1] == 1.0, "Total wealth must be unchanged because price was flat after the jump!"


def test_transaction_cost_deduction():
    """
    Verify exact transaction cost deduction for known position turnover.
    """
    # 4 bars, flat prices
    prices = np.array([100.0, 100.0, 100.0, 100.0])
    # Signals: go from 0 -> 1 at t=1, 1 -> 0 at t=2
    signals = np.array([1.0, 0.0, 0.0, 0.0])
    # positions with lag:
    # t=0: pos = 0
    # t=1: pos = 1 (turnover = 1.0)
    # t=2: pos = 0 (turnover = 1.0)
    # t=3: pos = 0 (turnover = 0.0)
    
    # Cost model: 10 bps commission + 5 bps spread + 5 bps slippage = 20 bps total = 0.0020
    cost_model = TransactionCostModel(commission_bps=10.0, half_spread_bps=5.0, slippage_bps=5.0)
    engine = BacktestEngine(cost_model=cost_model)
    res = engine.run(prices=prices, signals=signals)
    
    # Turnover:
    # t=0: 0
    # t=1: |1 - 0| = 1.0 -> cost = 0.0020
    # t=2: |0 - 1| = 1.0 -> cost = 0.0020
    # t=3: 0.0
    expected_net_returns = np.array([0.0, -0.0020, -0.0020, 0.0])
    assert np.allclose(res.net_returns, expected_net_returns, atol=1e-6)
    assert np.allclose(res.cost_drag_total, 0.0040, atol=1e-6)
    assert res.total_trades == 2
